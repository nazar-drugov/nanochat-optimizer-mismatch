"""
Train model. From root directory of the project, run as:

python -m scripts.base_train --train-tokens=3000238080

or distributed as:

torchrun --nproc_per_node=2 -m scripts.base_train --train-tokens=3000238080

The total batch size defaults to 262,144 tokens (see the note at --total-batch-size);
--train-tokens must be a multiple of it.

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --train-tokens=10240
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import argparse
import hashlib
from dataclasses import asdict

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, make_gpt_config
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes, tokenizer_dir_for, TOKENIZERS
from nanochat.checkpoint_manager import (
    save_checkpoint, load_checkpoint, validate_checkpoint_seed, make_base_experiment_name, validate_experiment_name,
    patch_missing_expert_bias, train_shard_inventory, validate_checkpoint_training_data, validate_checkpoint_tokenizer,
)
from nanochat.routing_probe import routing_probe, save_routing_snapshot, load_routing_snapshot, snapshot_change_rates
from nanochat.training_stats import accumulate_router_stats, expert_gradient_squared_norms, add_router_log_fields, write_train_stats_jsonl, update_batch_fingerprint
from nanochat.train_config import (
    add_moe_args, add_optimizer_args, resolve_optimizer_args, resolve_moe_args,
    muon_momentum_at, wsd_lr_multiplier, write_run_record, tokenizer_identity,
)
from nanochat.dataset import resolve_data_dir, list_parquet_files, list_split_files
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument(
    "--run",
    type=str,
    default=None,
    help="shared W&B and checkpoint name (default: generated automatically)",
)
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--seed", type=int, default=42, help="seed for model initialization")
# Model architecture
add_moe_args(parser)
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
# Data
parser.add_argument("--tokenizer", choices=TOKENIZERS, default="gpt2",
                    help="gpt2: GPT-2 BPE via tiktoken (team decision 2026-11-23; matches Ludziejewski et al.); nanochat: <base_dir>/tokenizer")
parser.add_argument("--dataset", type=str, default=None, help="pretraining dataset name (default: the registry default, FineWeb-Edu-100B)")
parser.add_argument("--data-dir", type=str, default=None, help="directory of nanochat-format parquet shards (overrides --dataset)")
parser.add_argument("--val-data-dir", type=str, default=None,
                    help="directory containing one validation parquet (e.g. D0); default: last shard in --data-dir")
# Training horizon
parser.add_argument("--train-tokens", type=int, required=True, help="total number of training tokens")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
# Default 256K tokens (2^18). Rationale (2026-11-23): Ludziejewski et al. (arXiv:2502.05172)
# end their batch ramp at 256K for models of this size (~118M active), and DeepSeek LLM's
# fit B_opt = 0.292 * C^0.327 gives ~240K for depth-10 at 3B tokens. Larger batches waste
# tokens under a fixed token budget and are reported to favor Muon over AdamW, which would
# widen the gap between AdamW- and Muon-pretrained base models. Keep it fixed across every
# cell and run the LR sweep at this batch: the optimal LR shifts with batch size.
parser.add_argument("--total-batch-size", type=int, default=262144, help="total batch size in tokens across all ranks (default 256K; keep identical across cells, see note above)")
add_optimizer_args(parser, weight_decay_default=0.1)
# Schedule: warmup, constant, then linear decay over the last warmdown_ratio of steps (spec §5)
parser.add_argument("--warmup-tokens", type=int, default=130_000_000,
                    help="tokens of linear LR warmup (default 130M, as in Ludziejewski et al. Sec. 5.1.3); rounded up to whole steps")
parser.add_argument("--warmup-steps", type=int, default=None, help="LR warmup in optimizer steps (overrides --warmup-tokens)")
parser.add_argument("--warmdown-ratio", type=float, default=0.2, help="ratio of iterations for the final linear LR decay")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--train-stats-every", type=int, default=100, help="log training statistics every N optimizer updates")
parser.add_argument("--train-stats-jsonl", type=str, default=None, help="optional rank-0 JSONL file for W&B training log records")
# Output
args = parser.parse_args()

if args.train_tokens <= 0:
    parser.error("--train-tokens must be greater than zero")
if args.total_batch_size <= 0:
    parser.error("--total-batch-size must be greater than zero")
if args.train_tokens % args.total_batch_size != 0:
    parser.error("--train-tokens must be divisible by --total-batch-size")
if args.train_stats_every <= 0:
    parser.error("--train-stats-every must be greater than zero")

# Fill defaults that depend on other options (Muon LR per variant, z-loss per balancer, warmup)
if args.warmup_steps is None:
    args.warmup_steps = -(-args.warmup_tokens // args.total_batch_size)  # ceil
optimizer_kwargs = resolve_optimizer_args(args)
moe_config_kwargs = resolve_moe_args(args)
data_dir = resolve_data_dir(args.dataset, args.data_dir)
val_data_dir = resolve_data_dir(data_dir=args.val_data_dir) if args.val_data_dir else data_dir
try:
    val_files = list_parquet_files(val_data_dir)
except FileNotFoundError as exc:
    parser.error(str(exc))
if not val_files or (args.val_data_dir and len(val_files) != 1):
    parser.error(f"Validation directory {val_data_dir} must contain "
                 + ("exactly one" if args.val_data_dir else "at least one") + " parquet file")
val_data_file = val_files[-1]
val_sha256 = hashlib.sha256()
with open(val_data_file, "rb") as val_file:
    for chunk in iter(lambda: val_file.read(1024 * 1024), b""):
        val_sha256.update(chunk)
val_data_sha256 = val_sha256.hexdigest()
# Names and sizes of the training shards, saved in checkpoints so a resumed run
# can detect added, removed or changed shards (the saved data position is only
# "file number, row group" and is meaningless for different files).
train_shards = train_shard_inventory(list_split_files("train", data_dir))

total_batch_size = args.total_batch_size
total_tokens = args.train_tokens
num_iterations = total_tokens // total_batch_size
if args.warmup_steps < 0 or not 0 <= args.warmdown_ratio <= 1:
    parser.error("--warmup-steps must be >= 0 and --warmdown-ratio in [0, 1]")
if args.warmup_steps + round(args.warmdown_ratio * num_iterations) > num_iterations:
    parser.error(f"warmup ({args.warmup_steps} steps) plus decay ({round(args.warmdown_ratio * num_iterations)} steps) "
                 f"exceeds the run ({num_iterations} steps); lower --warmup-tokens/--warmup-steps or --warmdown-ratio")

user_config = vars(args).copy()  # for logging

# -----------------------------------------------------------------------------
# Automatically generate a name for a run
default_experiment_name = make_base_experiment_name(args)
if args.run is None:
    experiment_name = default_experiment_name
else:
    try:
        experiment_name = validate_experiment_name(args.run)
    except ValueError as exc:
        parser.error(str(exc))

user_config["experiment_name"] = experiment_name

# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type, seed=args.seed)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    print0(f"GPU: {gpu_device_name}")
eval_batch_tokens = args.device_batch_size * args.max_seq_len * ddp_world_size
if args.eval_every > 0 and (args.eval_tokens < eval_batch_tokens or args.eval_tokens % eval_batch_tokens):
    parser.error(f"--eval-tokens must be a positive multiple of {eval_batch_tokens} "
                 "(device batch size × sequence length × world size)")
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
print0(f"Validation data: {val_data_file} (SHA-256 {val_data_sha256})")

# wandb logging init
wandb_run = (
    wandb.init(
        project="nanochat",
        name=experiment_name,
        config=user_config,
    )
    if master_process
    else DummyWandb()
)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer(args.tokenizer)
token_bytes = get_token_bytes(device=device, name=args.tokenizer)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")
tokenizer_id = tokenizer_identity(tokenizer_dir_for(args.tokenizer))  # file SHA-256s, saved in checkpoints and checked on resume

# -----------------------------------------------------------------------------
# Initialize the Model

# Build the model, move to device, init the weights
model_config = make_gpt_config(
    depth=args.depth,
    aspect_ratio=args.aspect_ratio,
    head_dim=args.head_dim,
    max_seq_len=args.max_seq_len,
    vocab_size=vocab_size,
    **moe_config_kwargs,
)
with torch.device("meta"):
    model = GPT(model_config) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
parameter_counts = model.parameter_counts()
print0("Parameter counts:\n" + "\n".join(f"  {k}: {v:,}" for k, v in parameter_counts.items()))
moe_layer_indices = model.moe_layer_indices
if moe_layer_indices:
    print0(f"MoE layers: {moe_layer_indices}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights(seed=args.seed) # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
checkpoint_dir = os.path.join(
    base_dir,
    "base_checkpoints",
    experiment_name,
)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    validate_checkpoint_seed(meta_data, args.seed)
    validate_checkpoint_training_data(meta_data, train_shards)
    validate_checkpoint_tokenizer(meta_data, args.tokenizer, tokenizer_id)
    saved_val_sha256 = meta_data.get("val_data_sha256")
    if saved_val_sha256 is None and args.val_data_dir:
        raise ValueError("Checkpoint has no validation SHA-256; refusing to resume with --val-data-dir")
    if saved_val_sha256 is not None and (saved_val_sha256 != val_data_sha256 or meta_data.get("val_data_file") != val_data_file):
        raise ValueError("Validation file or SHA-256 differs from the checkpoint; refusing to resume")
    patch_missing_expert_bias(model, model_data)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Use the explicit optimizer settings without parameter- or FLOP-based scaling.
print0(f"Total batch size: {total_batch_size:,} tokens")
print0(f"Number of iterations: {num_iterations:,}")
print0(f"Total training tokens: {total_tokens:,}")

# -----------------------------------------------------------------------------
# Initialize the selected optimizer. Embeddings and the lm_head use AdamW in every
# cell; matrices use --optimizer-name; routers use --router-optimizer.
optimizer = orig_model.setup_optimizer(**optimizer_kwargs)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# Full run record next to the checkpoints (spec §0 rule 1)
if master_process:
    record_name = "config.json" if not resuming else f"config_resume_{args.resume_from_step:06d}.json"
    write_run_record(
        os.path.join(checkpoint_dir, record_name),
        args=args, model=orig_model, optimizer=optimizer,
        device_type=device_type, world_size=ddp_world_size, compute_dtype=COMPUTE_DTYPE,
        data={"train_data_dir": data_dir, "train_shards": train_shards, "val_data_dir": val_data_dir,
              "val_data_file": val_data_file, "val_data_sha256": val_data_sha256,
              "tokenizer_name": args.tokenizer,
              "tokenizer": tokenizer_id},
        extra={"experiment_name": experiment_name, "stage": "pretrain", "num_iterations": num_iterations},
    )

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict, data_dir=data_dir)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device, data_dir=val_data_dir)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# Learning rate schedule (linear warmup, constant, linear decay over the last warmdown_ratio)
warmdown_iters = round(args.warmdown_ratio * num_iterations)
def get_lr_multiplier(it):
    return wsd_lr_multiplier(it, num_iterations, args.warmup_steps, warmdown_iters, args.final_lr_frac)

# Muon momentum: constant by default (spec), or nanochat's schedule
def get_muon_momentum(it):
    return muon_momentum_at(it, num_iterations, args.warmdown_ratio, args)

# Cumulative per-(MoE layer, expert) token counts over the whole run (all ranks)
n_moe_layers = len(moe_layer_indices)
cumulative_expert_counts = torch.zeros((n_moe_layers, model_config.n_experts), dtype=torch.long, device=device) if n_moe_layers else None
if resuming and cumulative_expert_counts is not None and meta_data.get("cumulative_expert_counts") is not None:
    cumulative_expert_counts.copy_(torch.tensor(meta_data["cumulative_expert_counts"], dtype=torch.long))

# Running hash of every training batch this rank consumes (see training_stats).
# Saved per rank in checkpoints; identical across optimizer cells and across resumes.
data_fingerprint = torch.zeros((), dtype=torch.long, device=device)
fingerprint_weights = None
if resuming and meta_data.get("data_fingerprints") is not None:
    data_fingerprint.fill_(meta_data["data_fingerprints"][ddp_rank])

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Rank 0 owns the fixed probe and its snapshots. On resume, reload the exact
# tokens used by the checkpoint rather than drawing a fresh validation batch.
previous_snapshot = None
probe_tokens = None
if args.ffn_type == "moe" and master_process:
    if resuming:
        previous_path = os.path.join(checkpoint_dir, f"routing_{step:06d}.pt")
        previous_snapshot = load_routing_snapshot(previous_path, expected_step=step, n_experts=model_config.n_experts)
        probe_tokens = previous_snapshot["probe_tokens"].to(device)
    else:
        # The validation loader reuses its output buffer, so keep our own copy.
        probe_tokens = next(build_val_loader())[0].clone()

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    tokens_seen = total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "tokens_seen": tokens_seen,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "tokens_seen": tokens_seen,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend=tokenizer.get_bos_token_id())
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if step != args.resume_from_step and (last_step or (step > 0 and args.save_every > 0 and step % args.save_every == 0)):
        if is_ddp_initialized():
            gathered_fingerprints = [torch.zeros_like(data_fingerprint) for _ in range(ddp_world_size)]
            dist.all_gather(gathered_fingerprints, data_fingerprint)
            data_fingerprints = [int(fp.item()) for fp in gathered_fingerprints]
        else:
            data_fingerprints = [int(data_fingerprint.item())]
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "train_tokens": total_tokens,
                "tokenizer": args.tokenizer,
                "tokenizer_identity": tokenizer_id,
                "data_dir": data_dir,
                "train_shards": train_shards,
                "val_data_dir": val_data_dir,
                "val_data_file": val_data_file,
                "val_data_sha256": val_data_sha256,
                "parameter_counts": parameter_counts,
                "cumulative_expert_counts": None if cumulative_expert_counts is None else cumulative_expert_counts.cpu().tolist(),
                "data_fingerprints": data_fingerprints,  # per rank, over all batches trained so far
                # x/y have already been prefetched. Replay them first on resume,
                # then continue from the saved document buffer and cursor.
                "dataloader_state_dict": {
                    **dataloader_state_dict,
                    "pending_inputs": x.detach().cpu().tolist(),
                    "pending_targets": y.detach().cpu().tolist(),
                },
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

        if args.ffn_type == "moe" and master_process:
            current_choices = routing_probe(orig_model, probe_tokens)
            current_snapshot = {
                "step": step,
                "probe_tokens": probe_tokens.cpu(),
                "choices": current_choices,
            }
            if previous_snapshot is not None:
                change_rates = snapshot_change_rates(previous_snapshot, current_snapshot)
                wandb_run.log({
                    "step": step,
                    "tokens_seen": tokens_seen,
                    **{
                        f"probe/moe/layer_{layer_idx}/change_rate": rate.item()
                        for layer_idx, rate in enumerate(change_rates)
                    },
                })

            snapshot_path = os.path.join(checkpoint_dir, f"routing_{step:06d}.pt")
            save_routing_snapshot(snapshot_path, step, probe_tokens, current_choices, n_experts=model_config.n_experts)
            previous_snapshot = current_snapshot

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()

    metrics_sums = {}
    router_stats_sums = None
    for micro_step in range(grad_accum_steps):
        data_fingerprint, fingerprint_weights = update_batch_fingerprint(data_fingerprint, x, fingerprint_weights)
        loss, metrics, router_stats = model(x, y, return_metrics=True)

        # Divide the loss by the number of gradient accumulation steps
        scaled_loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        for key, value in metrics.items():
            metrics_sums[key] = metrics_sums.get(key, 0) + value.detach() / grad_accum_steps
        if router_stats:
            router_stats_sums = accumulate_router_stats(router_stats_sums, router_stats, grad_accum_steps)

        # Backpropagate through the loss
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    if scaler is not None:
        scaler.unscale_(optimizer)

    # Sample raw expert gradients after accumulation and unscaling, before the
    # optimizer updates parameters and zero_grad clears the gradients.
    expert_grad_norms = None
    if args.ffn_type == "moe" and step % args.train_stats_every == 0:
        expert_grad_sq = expert_gradient_squared_norms(orig_model, device)
        # The optimizer reduces gradients inside step(), so this diagnostic is
        # the RMS local gradient norm across ranks, not the optimizer update norm.
        if is_ddp_initialized():
            dist.all_reduce(expert_grad_sq, op=dist.ReduceOp.SUM)
            expert_grad_sq /= ddp_world_size
        expert_grad_norms = expert_grad_sq.sqrt()

    if scaler is not None:
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)

    # MoE load for this step, summed over micro-batches and ranks. It drives the
    # bias balancer (identical update on every rank) and the run's cumulative counts.
    if router_stats_sums is not None:
        step_expert_counts = router_stats_sums.pop("expert_counts")
        if is_ddp_initialized():
            dist.all_reduce(step_expert_counts, op=dist.ReduceOp.SUM)
        cumulative_expert_counts += step_expert_counts
        orig_model.update_expert_bias(step_expert_counts)
    train_loss_f = metrics_sums["total"].item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % args.train_stats_every == 0:
        if router_stats_sums is not None and is_ddp_initialized():
            for value in router_stats_sums.values():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                value /= ddp_world_size
        log_data = {
            "step": step,
            "tokens_seen": tokens_seen + total_batch_size,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/epoch": epoch,
        }
        log_data.update({
            f"train/{key}_loss": value.item()
            for key, value in metrics_sums.items()
        })
        if optimizer.last_grad_norm is not None:
            log_data["train/grad_norm"] = optimizer.last_grad_norm.item()  # pre-clip global norm
            log_data["train/grad_clip_coef"] = optimizer.last_clip_coef.item()  # < 1 means clipped
        add_router_log_fields(
            log_data, router_stats_sums, expert_grad_norms,
            layer_indices=moe_layer_indices,
            expert_bias=orig_model.expert_bias_state() if model_config.balancer == "bias" else None,
        )
        wandb_run.log(log_data)
        write_train_stats_jsonl(args.train_stats_jsonl, log_data, master_process)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print and log a few more stats
peak_memory_mib = get_max_memory() / 1024 / 1024
print0(f"Peak memory usage: {peak_memory_mib:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
wandb_run.log({
    "step": step,
    "tokens_seen": total_batch_size * step,
    "total_training_time": total_training_time,
    "train/peak_memory_mib": peak_memory_mib,
})

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
