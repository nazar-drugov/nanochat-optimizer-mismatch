"""
Training-step throughput of the pretraining model (spec §8.8 benchmark).

Times compiled forward+backward micro-batches and optimizer steps separately on
random tokens, with the model and optimizer built as scripts/base_train.py builds
them, for each (E, dispatch, optimizer) combination, and projects the time of one
training step of --total-batch-size tokens on one GPU. One JSON line per combination.
The dense SwiGLU model (same active FLOPs as top-1 MoE) is the reference.

Example (B200 grid at depth 10):
    python -m scripts.bench_moe_throughput --depth 10 --experts 1,4,8,16 \
        --dispatch grouped,loop --optimizers adamw,muon --device-batch-size 32
"""

import argparse
import json
import time

import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, make_gpt_config, resolve_moe_dispatch
from nanochat.optim import MUON_VARIANTS


def build(args, ffn_type, n_experts, dispatch):
    config = make_gpt_config(depth=args.depth, aspect_ratio=args.aspect_ratio, head_dim=args.head_dim,
                             max_seq_len=args.max_seq_len, vocab_size=args.vocab_size, ffn_type=ffn_type,
                             n_experts=n_experts, moe_dispatch=dispatch)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cuda")
    model.init_weights(seed=0)
    return model


def bench_one(args, ffn_type, n_experts, dispatch, optimizer_name):
    torch._dynamo.reset()
    from torch._dynamo.utils import counters
    counters.clear()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build(args, ffn_type, n_experts, dispatch)
    optimizer = model.setup_optimizer(optimizer_name=optimizer_name, muon_variant=args.muon_variant,
                                      weight_decay=0.1, max_grad_norm=1.0, track_grad_norm=True)
    compiled = torch.compile(model, dynamic=False)
    B, T = args.device_batch_size, args.max_seq_len
    gen = torch.Generator(device="cuda").manual_seed(0)
    targets = torch.randint(0, args.vocab_size, (B, T), device="cuda", generator=gen)

    def micro_batch():
        idx = torch.randint(0, args.vocab_size, (B, T), device="cuda", generator=gen)  # new routes every micro-batch
        compiled(idx, targets).backward()

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        return time.time() - t0

    t0 = time.time()
    for _ in range(args.warmup):
        micro_batch()
        optimizer.step()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    warmup_s = time.time() - t0
    fwd_bwd, opt = [], []
    for _ in range(args.steps):
        fwd_bwd.append(timed(micro_batch))
        opt.append(timed(optimizer.step))
        model.zero_grad(set_to_none=True)
    median = lambda xs: sorted(xs)[len(xs) // 2]
    fb_s, opt_s = median(fwd_bwd), median(opt)
    # One optimizer step per --total-batch-size tokens, as in training on one GPU
    accum = max(1, args.total_batch_size // (B * T))
    step_s = accum * fb_s + opt_s
    tokens = accum * B * T
    resolved = resolve_moe_dispatch(model.config, "cuda", COMPUTE_DTYPE) if ffn_type == "moe" else None
    reserved = torch.cuda.max_memory_reserved()
    return {
        "ffn_type": ffn_type, "E": n_experts, "dispatch": resolved, "optimizer": optimizer_name,
        "depth": args.depth, "device_batch_size": B, "seq_len": T, "grad_accum": accum,
        "fwd_bwd_ms": round(fb_s * 1e3, 2), "opt_step_ms": round(opt_s * 1e3, 2),
        "ms_per_step": round(step_s * 1e3, 1), "tok_per_s": round(tokens / step_s),
        "model_tflops": round(model.estimate_flops() * tokens / step_s / 1e12, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gb": round(reserved / 2**30, 2),
        # On WSL/Windows, CUDA silently spills to host memory instead of raising OOM
        "exceeds_gpu_memory": reserved > torch.cuda.get_device_properties(0).total_memory,
        "warmup_s": round(warmup_s, 1),
        "graph_breaks": sum(counters["graph_break"].values()),
        "unique_graphs": counters["stats"].get("unique_graphs", 0),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=50257, help="GPT-2 tokenizer")
    parser.add_argument("--experts", type=str, default="1,4,8,16")
    parser.add_argument("--dispatch", type=str, default="grouped", help="comma list of grouped|loop")
    parser.add_argument("--optimizers", type=str, default="adamw,muon")
    parser.add_argument("--muon-variant", choices=MUON_VARIANTS, default="moonlight")
    parser.add_argument("--no-dense", action="store_true", help="skip the dense SwiGLU reference")
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--total-batch-size", type=int, default=262144,
                        help="tokens per optimizer step; ms_per_step = accum * fwd_bwd + opt_step on one GPU")
    parser.add_argument("--warmup", type=int, default=10, help="untimed steps (compilation)")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", type=str, default=None, help="append JSON lines to this file")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "this benchmark needs a CUDA GPU"
    torch.backends.cuda.matmul.allow_tf32 = True

    combos = []
    for optimizer_name in args.optimizers.split(","):
        if not args.no_dense:
            combos.append(("swiglu", 1, "loop", optimizer_name))
        for n_experts in map(int, args.experts.split(",")):
            for dispatch in args.dispatch.split(","):
                combos.append(("moe", n_experts, dispatch, optimizer_name))
    dense_ms = {}
    for ffn_type, n_experts, dispatch, optimizer_name in combos:
        try:
            row = bench_one(args, ffn_type, n_experts, dispatch, optimizer_name)
        except torch.OutOfMemoryError:
            row = {"ffn_type": ffn_type, "E": n_experts, "dispatch": dispatch, "optimizer": optimizer_name, "error": "OOM"}
        if row.get("ffn_type") == "swiglu" and "ms_per_step" in row:
            dense_ms[optimizer_name] = row["ms_per_step"]
        if "ms_per_step" in row and optimizer_name in dense_ms:
            row["x_dense"] = round(row["ms_per_step"] / dense_ms[optimizer_name], 3)
        line = json.dumps(row)
        print(line, flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
