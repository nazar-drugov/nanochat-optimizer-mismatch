"""
Language-model fine-tuning of a pretrained base model, with the learning-forgetting and
routing measurements of EXPERIMENT_SPEC.md §2, §6 and §7.

One process runs every cell (fine-tuning optimizer x learning rate x seed) for one base
model and one fine-tuning dataset, reusing the compiled model; each cell restarts from
the base weights with fresh optimizer state (spec §6). Cells that already have a
result.json are skipped, so a resubmitted job continues where it stopped.

Fine-tuning length (--branch-tokens): a cell can run several lengths at once. One trunk runs
to the longest length; at the end of each shorter length's constant phase the model, optimizer
state and data position are copied, that length's decay is run and measured, and the trunk
resumes from the copy. Warmup is fixed in steps (from --train-tokens), so every length is the
same as a separate run of that length with the same seed. Each length gets its own directory,
suffixed -T<steps>.

Checkpoints (--checkpoint-spacing log, the default): steps 2, 4, 8, ... below the end of the
constant phase, the end of the constant phase, the middle of the decay and the last step
(240 steps: 2, 4, 8, 16, 32, 64, 128, 192, 216, 240).

Per cell (output: <out-root>/<base-run>/<dataset>/<optimizer>-lr<lr>-s<seed>/):
  config.json      fully resolved settings, base model, data and eval-set identity
  metrics.jsonl    one row per (checkpoint, eval set, layer); layer -1 = whole model
  metrics.parquet  the same rows, written when the cell finishes
  ckpt_<step>.npz  per-document loss sums (normal, replay, reverse), per-document flip
                   counts per layer, and the route table on D0 (uint8, compressed)
  result.json      status, final learning / forgetting, curves
Base model (once per base and dataset, in .../base/): the same measurements at step 0,
its route tables, separation S and the two-halves reference.

Example:
  python -m scripts.finetune --base-run e8-main-adamw-s42 --ft-data-dir <dir> --ft-name wikitext103 \
      --d0-file <scratch>/heldout/fineweb-edu-100b/d0.parquet --lrs adamw=1e-4,2e-4 muon=2e-4,4e-4 --seeds 1 2
"""

import argparse
import copy
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nanochat.checkpoint_manager import build_model
from nanochat.common import COMPUTE_DTYPE, autodetect_device_type, compute_init, get_base_dir, print0
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.dataset import list_parquet_files
from nanochat.eval_set import EvalSet, build_eval_set, per_doc_sums
from nanochat.pretrain_sweep import (
    FINETUNE_BASE_CHECKPOINTS, FINETUNE_BASE_CHECKPOINTS_DIR, FINETUNE_BASE_STEP, base_run_seed,
)
from nanochat.finetune_metrics import (
    expert_relative_change, flip_cause_counts, load_shares, maxvio, route_comparison, separation, top2_margin,
    token_weighted_mean,
)
from nanochat.train_config import (
    add_optimizer_args, file_sha256, git_state, replace_atomic, wsd_lr_multiplier, write_json_atomic,
    write_run_record,
)
from nanochat.training_stats import update_batch_fingerprint

CONVENTIONS = {
    "forgetting": "mean over scored D0 tokens of L(theta_1, r_1) - L(theta_0, r_0), nats",
    "learning": "mean over scored D1 tokens of L(theta_0, r_0) - L(theta_1, r_1), nats",
    "replay": "F_route = L(theta_1, r_1) - L(theta_1, r_0); F_fixed = L(theta_1, r_0) - L(theta_0, r_0); "
              "reverse = L(theta_0, r_1) - L(theta_0, r_0); all on D0",
    "forced_route_gate": "the current model's own softmax probability of the forced expert",
    "route": "expert actually used, argmax(softmax(z) + selection bias), eval mode, dropless",
    "flip_cause_bias_side": "weight side (the selection bias is part of the router state)",
    "optimizer_state": "fresh in every cell",
    "data_order": "seed k gives the same batch sequence in every cell (paired design)",
}


# ---------------------------------------------------------------------------
# Evaluation

@torch.no_grad()
def forward_losses(model, x, y, forced=None, capture=None):
    losses = model(x, y, loss_reduction="none", forced_routes=forced, routing_capture=capture)
    return losses.float().view(y.shape)  # the model returns them flattened


@torch.no_grad()
def evaluate_base(model0, eval_set, batch_size, device):
    """Base-model pass: per-token losses, routes, router margins and per-doc sums."""
    model0.eval()
    layers = model0.moe_layer_indices
    losses, routes, margins = [], {l: [] for l in layers}, {l: [] for l in layers}
    for _, x, y in eval_set.batches(batch_size):
        cap = []
        losses.append(forward_losses(model0, x.to(device), y.to(device), capture=cap).cpu())
        for entry in cap:
            routes[entry["layer"]].append(entry["choices"].to(torch.uint8).cpu())
            margins[entry["layer"]].append(top2_margin(entry["logits"]).cpu())
    loss = torch.cat(losses)
    sums, counts = per_doc_sums(loss, eval_set.doc_ids, eval_set.n_docs)
    return {"loss": loss, "doc_sums": sums, "doc_counts": counts,
            "routes": {l: torch.cat(r) for l, r in routes.items()},
            "margin": {l: float(torch.cat(m).mean()) for l, m in margins.items()}}


def router_state(model, layer):
    mlp = model.transformer.h[layer].mlp
    bias = mlp.expert_bias.detach() if model.config.balancer == "bias" else None
    return mlp.router.weight.detach(), bias


@torch.no_grad()
def evaluate_checkpoint(model1, model0, eval_set, base, batch_size, device, full):
    """Fine-tuned model on one eval set. full=True (D0) adds replay, reverse order and flip cause."""
    model1.eval()
    model0.eval()
    layers = model1.moe_layer_indices
    n_experts = model1.config.n_experts
    sums = {k: torch.zeros(eval_set.n_docs, dtype=torch.float64) for k in ("normal", "replay", "reverse")}
    doc_flips = torch.zeros(eval_set.n_docs, len(layers), dtype=torch.float64)
    routes1 = {l: [] for l in layers}
    cause = {l: {"flips": 0, "input_driven": 0, "weight_driven": 0, "either": 0, "interaction": 0} for l in layers}
    for start, x, y in eval_set.batches(batch_size):
        x, y = x.to(device), y.to(device)
        rows = slice(start, start + x.size(0))
        ids = eval_set.doc_ids[rows]
        r0 = {l: base["routes"][l][rows].to(device=device, dtype=torch.long) for l in layers}
        cap1 = []
        loss = forward_losses(model1, x, y, capture=cap1).cpu()
        sums["normal"] += per_doc_sums(loss, ids, eval_set.n_docs)[0]
        r1 = {}
        for i, entry in enumerate(cap1):
            l = entry["layer"]
            r1[l] = entry["choices"]
            routes1[l].append(entry["choices"].to(torch.uint8).cpu())
            flipped = (entry["choices"] != r0[l]).cpu()
            doc_flips[:, i] += per_doc_sums(flipped.float(), ids, eval_set.n_docs)[0]
        if not full:
            continue
        sums["replay"] += per_doc_sums(forward_losses(model1, x, y, forced=r0).cpu(), ids, eval_set.n_docs)[0]
        sums["reverse"] += per_doc_sums(forward_losses(model0, x, y, forced=r1).cpu(), ids, eval_set.n_docs)[0]
        cap0 = []
        forward_losses(model0, x, y, capture=cap0)
        for e0, e1 in zip(cap0, cap1):
            l = e0["layer"]
            w0, b0 = router_state(model0, l)
            w1, b1 = router_state(model1, l)
            counts = flip_cause_counts(e0["inputs"], e1["inputs"], w0, w1, r0[l], r1[l], b0, b1)
            for key, value in counts.items():
                cause[l][key] += value
    routes1 = {l: torch.cat(r) for l, r in routes1.items()}
    scored = eval_set.doc_ids >= 0
    per_layer = {}
    for i, l in enumerate(layers):
        stats = route_comparison(base["routes"][l][scored], routes1[l][scored], n_experts)
        if full:
            stats.update({f"cause_{k}": v for k, v in cause[l].items()})
        per_layer[l] = stats
    return {"doc_sums": sums, "doc_flips": doc_flips, "routes": routes1, "per_layer": per_layer}


# ---------------------------------------------------------------------------
# One fine-tuning cell

def lr_tag(lr):
    return f"{lr:.3g}"


def cell_rows(meta, step, tokens, eval_name, model_level, per_layer):
    rows = [{**meta, "step": step, "tokens": tokens, "eval_set": eval_name, "layer": -1, **model_level}]
    for layer, stats in per_layer.items():
        rows.append({**meta, "step": step, "tokens": tokens, "eval_set": eval_name, "layer": layer, **stats})
    return rows


def append_rows(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def checkpoint_steps(steps, decay, spacing="log", n_checkpoints=10):
    """Measurement steps of a run of `steps` steps whose last `decay` steps decay the LR."""
    if spacing == "linear":
        return sorted({round(i * steps / n_checkpoints) for i in range(1, n_checkpoints + 1)})
    stable_end = steps - decay
    points = {2 ** k for k in range(1, max(stable_end, 2).bit_length() + 1) if 2 ** k < stable_end}
    points |= {stable_end, stable_end + decay // 2, steps}
    return sorted(p for p in points if 1 <= p <= steps)


class Branch:
    """One fine-tuning length within a cell."""

    def __init__(self, args, steps, suffix):
        self.steps = steps
        self.decay = round(args.warmdown_ratio * steps)
        self.branch_point = steps - self.decay  # last step at the trunk's (constant) LR
        self.ckpts = checkpoint_steps(steps, self.decay, args.checkpoint_spacing, args.n_checkpoints)
        self.suffix = suffix
        self.curve, self.status, self.steps_completed, self.done = [], "completed", 0, False


class ReplayLoader:
    """Wraps the data loader so a branch's batches can be handed to the trunk again.

    The loader yields views of one persistent buffer that it overwrites on every call, so a
    recorded batch must be copied; holding the yielded tensors would replay only the last batch."""

    def __init__(self, loader):
        self.loader, self.pending, self.recording = loader, deque(), None

    def __next__(self):
        item = self.pending.popleft() if self.pending else next(self.loader)[:2]
        if self.recording is not None:
            self.recording.append(tuple(t.clone() for t in item))
        return item

    def record(self):
        self.recording = []

    def rewind(self):
        self.pending = deque(self.recording + list(self.pending))
        self.recording = None


def train_steps(args, ctx, optimizer, loader, state, first, last, branch, on_checkpoint):
    """Run steps first..last on the schedule of `branch`; returns False if the loss diverged."""
    model, compiled = ctx["model"], ctx["compiled"]
    for step in range(first, last + 1):
        lrm = wsd_lr_multiplier(step - 1, branch.steps, ctx["warmup"], branch.decay, 0.0)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        loss_sum, step_counts = 0.0, 0
        for _ in range(ctx["accum"]):
            x, y = next(loader)
            state["fingerprint"], state["fp_weights"] = update_batch_fingerprint(state["fingerprint"], x,
                                                                                 state["fp_weights"])
            loss, _, router_stats = compiled(x, y, return_metrics=True)
            (loss / ctx["accum"]).backward()
            loss_sum += loss.detach()
            if router_stats:
                step_counts = step_counts + torch.stack([s["expert_counts"] for s in router_stats])
        optimizer.step()
        model.zero_grad(set_to_none=True)
        if torch.is_tensor(step_counts):
            state["counts"] += step_counts
            # as in pretraining: one bias update per optimizer step from the step's summed counts
            if model.config.balancer == "bias" and args.balancer_mode == "active":
                model.update_expert_bias(step_counts)
        state["step"] = step
        train_loss = float(loss_sum) / ctx["accum"]
        if not math.isfinite(train_loss):
            print0(f"training loss {train_loss} at step {step}; stopping")
            return False
        on_checkpoint(step, train_loss)
    return True


def run_cell(args, cell, ctx):
    """Fine-tune once from the base weights (one trunk with its decay branches, or a single
    run) and measure every checkpoint."""
    optimizer_name, lr, seed = cell
    name = f"{optimizer_name}-lr{lr_tag(lr)}-s{seed}"
    lengths = sorted({ctx["main_steps"], *ctx["branch_steps"]})
    branches = [Branch(args, n, f"-T{n}" if len(lengths) > 1 else "") for n in lengths]
    run_dirs = {b.steps: ctx["out_dir"] / (name + b.suffix) for b in branches}
    if all((d / "result.json").exists() for d in run_dirs.values()):
        for b in branches:  # never skip on a result of a different cell that shares the directory name
            done = json.loads((run_dirs[b.steps] / "result.json").read_text())
            if (done.get("lr"), done.get("steps"), done.get("warmup_steps")) != (lr, b.steps, ctx["warmup"]):
                raise RuntimeError(f"{run_dirs[b.steps]}/result.json has lr {done.get('lr')!r}, steps "
                                   f"{done.get('steps')}, warmup {done.get('warmup_steps')}; this cell has "
                                   f"lr {lr!r}, steps {b.steps}, warmup {ctx['warmup']}")
        print0(f"Skipping {name}: result.json exists")
        return
    for run_dir in run_dirs.values():  # an interrupted cell restarts from scratch
        if run_dir.exists():
            for f in run_dir.iterdir():
                f.unlink()
        run_dir.mkdir(parents=True, exist_ok=True)
    model, device = ctx["model"], ctx["device"]
    with torch.no_grad():  # restore the base weights and balancer state
        for key, tensor in model.state_dict().items():
            tensor.copy_(ctx["base_state"][key])
    model.train()
    pre = ctx["pretrain_defaults"]
    optimizer = model.setup_optimizer(
        optimizer_name=optimizer_name, router_optimizer="same",
        embedding_lr=pre.embedding_lr * args.non_matrix_lr_scale,
        unembedding_lr=pre.unembedding_lr * args.non_matrix_lr_scale,
        adamw_matrix_lr=lr, muon_matrix_lr=lr, muon_variant=pre.muon_variant,
        adamw_beta1=pre.adamw_beta1, adamw_beta2=pre.adamw_beta2, adamw_eps=pre.adamw_eps,
        muon_momentum=pre.muon_momentum, muon_ns_steps=pre.muon_ns_steps,
        weight_decay=args.weight_decay, router_weight_decay=0.0,
        max_grad_norm=args.grad_clip if args.grad_clip > 0 else None, track_grad_norm=True,
    )
    trunk = branches[-1]
    meta = {b.steps: {"run": run_dirs[b.steps].name, "base_run": ctx["base_run"],
                      "pretrain_optimizer": ctx["pretrain_optimizer"], "ft_dataset": args.ft_name,
                      "ft_optimizer": optimizer_name, "lr": lr, "seed": seed, "ft_steps": b.steps,
                      "n_experts": model.config.n_experts} for b in branches}
    for b in branches:
        write_run_record(run_dirs[b.steps] / "config.json", args=args, model=model, optimizer=optimizer,
                         device_type=device.type, world_size=1, compute_dtype=COMPUTE_DTYPE,
                         data={"ft_data_dir": args.ft_data_dir, "ft_train_files": ctx["ft_train_files"],
                               "eval_sets": ctx["eval_identity"]},
                         extra={"stage": "finetune", "cell": meta[b.steps], "base": ctx["base_identity"],
                                "steps": b.steps, "grad_accum": ctx["accum"], "warmup_steps": ctx["warmup"],
                                "decay_steps": b.decay, "checkpoint_steps": b.ckpts,
                                "branch": {"lengths": lengths, "branch_point": b.branch_point,
                                           "trunk_steps": trunk.steps} if len(branches) > 1 else None,
                                "conventions": CONVENTIONS, "balancer_mode": args.balancer_mode})
    loader = ReplayLoader(tokenizing_distributed_data_loader_with_state_bos_bestfit(
        ctx["tokenizer"], args.device_batch_size, ctx["seq_len"], split="train", device=device,
        data_dir=args.ft_data_dir, shuffle_seed=seed))
    n_layers = len(model.moe_layer_indices)
    state = {"step": 0, "counts": torch.zeros(n_layers, model.config.n_experts, dtype=torch.long, device=device),
             "fingerprint": torch.zeros((), dtype=torch.long, device=device), "fp_weights": None}
    t0 = time.time()

    def measure_for(targets, step, train_loss):
        """Measure once and record the checkpoint in every branch in `targets`."""
        if not targets:
            return
        first = targets[0]
        rows, entry = measure(args, ctx, model, step, step * args.total_batch_size, meta[first.steps],
                              run_dirs[first.steps], state["counts"])
        entry["train_loss"] = train_loss
        for b in targets:
            b.curve.append(dict(entry))
            if b is not first:
                os.link(run_dirs[first.steps] / f"ckpt_{step:06d}.npz", run_dirs[b.steps] / f"ckpt_{step:06d}.npz")
            append_rows(run_dirs[b.steps] / "metrics.jsonl",
                        [{**r, "run": meta[b.steps]["run"], "ft_steps": b.steps} for r in rows])
        print0(f"{name} step {step} ({', '.join(str(b.steps) for b in targets)}): learning {entry['learning']:.4f} "
               f"forgetting {entry['forgetting']:.4f} F_route {entry['F_route']:.4f}")
        model.train()

    def finish(b, status):
        b.status, b.steps_completed = status, state["step"] if status == "completed" else state["step"] - 1
        b.done = True
        run_dir = run_dirs[b.steps]
        rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()] \
            if (run_dir / "metrics.jsonl").exists() else []
        if rows:  # columns from every row (from_pylist would keep only the first row's keys)
            keys = dict.fromkeys(k for row in rows for k in row)
            pq.write_table(pa.Table.from_pydict({k: [row.get(k) for row in rows] for k in keys}),
                           run_dir / "metrics.parquet")
        final = b.curve[-1] if b.curve else {}
        write_json_atomic(run_dir / "result.json", {
            "status": status, **meta[b.steps], "steps": b.steps, "steps_completed": b.steps_completed,
            "warmup_steps": ctx["warmup"], "decay_steps": b.decay,
            "final_learning": final.get("learning"), "final_forgetting": final.get("forgetting"),
            "curve": b.curve, "data_fingerprint": int(state["fingerprint"]), "wall_time_s": time.time() - t0,
            "cumulative_expert_counts": state["counts"].cpu().tolist(),
        })

    def trunk_checkpoint(step, train_loss):
        # trunk steps up to a branch's branch point are shared with that branch
        measure_for([b for b in branches if step in b.ckpts and (b is trunk or step <= b.branch_point)],
                    step, train_loss)

    def branch_checkpoint_for(b):
        return lambda step, train_loss: measure_for([b] if step in b.ckpts else [], step, train_loss)

    position = 0
    for b in branches[:-1]:
        if not train_steps(args, ctx, optimizer, loader, state, position + 1, b.branch_point, trunk, trunk_checkpoint):
            for rest in branches:
                if not rest.done:
                    finish(rest, "diverged")
            return
        position = b.branch_point
        snapshot = {"model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "counts": state["counts"].clone(), "fingerprint": state["fingerprint"].clone()}
        loader.record()
        ok = train_steps(args, ctx, optimizer, loader, state, position + 1, b.steps, b, branch_checkpoint_for(b))
        finish(b, "completed" if ok else "diverged")
        loader.rewind()
        with torch.no_grad():
            for key, tensor in model.state_dict().items():
                tensor.copy_(snapshot["model"][key])
        optimizer.load_state_dict(snapshot["optimizer"])
        model.zero_grad(set_to_none=True)
        state.update(step=position, counts=snapshot["counts"], fingerprint=snapshot["fingerprint"])
        del snapshot
    ok = train_steps(args, ctx, optimizer, loader, state, position + 1, trunk.steps, trunk, trunk_checkpoint)
    finish(trunk, "completed" if ok else "diverged")


def measure(args, ctx, model, step, tokens, meta, run_dir, cumulative_counts):
    """All checkpoint measurements; returns metrics rows and the curve entry."""
    base0, base1 = ctx["base_d0"], ctx["base_d1"]
    d0 = evaluate_checkpoint(model, ctx["model0"], ctx["d0"], base0, args.eval_batch_size, ctx["device"], full=True)
    d1 = evaluate_checkpoint(model, ctx["model0"], ctx["d1"], base1, args.eval_batch_size, ctx["device"], full=False)
    c0, c1 = base0["doc_counts"], base1["doc_counts"]
    l1_d0 = token_weighted_mean(d0["doc_sums"]["normal"], c0)
    l0_d0 = token_weighted_mean(base0["doc_sums"], c0)
    replay = token_weighted_mean(d0["doc_sums"]["replay"], c0)
    reverse = token_weighted_mean(d0["doc_sums"]["reverse"], c0)
    l1_d1 = token_weighted_mean(d1["doc_sums"]["normal"], c1)
    l0_d1 = token_weighted_mean(base1["doc_sums"], c1)
    entry = {"step": step, "tokens": tokens, "loss_d0": l1_d0, "loss_d1": l1_d1,
             "forgetting": l1_d0 - l0_d0, "learning": l0_d1 - l1_d1,
             "F_route": l1_d0 - replay, "F_fixed": replay - l0_d0, "reverse_order": reverse - l0_d0}
    rho = expert_relative_change(model, ctx["base_state"], model.config.n_experts)
    layers = model.moe_layer_indices
    for i, l in enumerate(layers):
        d0["per_layer"][l]["rho"] = rho[l]
        d0["per_layer"][l]["ft_tokens_per_expert"] = cumulative_counts[i].cpu().tolist()
    rows = cell_rows(meta, step, tokens, "D0", {k: v for k, v in entry.items() if k not in ("step", "tokens")},
                     d0["per_layer"])
    rows += cell_rows(meta, step, tokens, "D1", {"loss": l1_d1, "learning": entry["learning"]}, d1["per_layer"])
    np.savez_compressed(
        run_dir / f"ckpt_{step:06d}.npz",
        d0_doc_loss=d0["doc_sums"]["normal"].numpy(), d0_doc_loss_replay=d0["doc_sums"]["replay"].numpy(),
        d0_doc_loss_reverse=d0["doc_sums"]["reverse"].numpy(), d0_doc_flips=d0["doc_flips"].numpy(),
        d1_doc_loss=d1["doc_sums"]["normal"].numpy(), d1_doc_flips=d1["doc_flips"].numpy(),
        d0_routes=torch.stack([d0["routes"][l] for l in layers]).numpy(),
        layers=np.array(layers))
    return rows, entry


# ---------------------------------------------------------------------------

def parse_lrs(values):
    lrs = {}
    for value in values:
        name, _, rest = value.partition("=")
        if name not in ("adamw", "muon") or not rest:
            raise SystemExit(f"--lrs expects optimizer=lr1,lr2,... got {value!r}")
        lrs[name] = [float(v) for v in rest.split(",")]
    return lrs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-run", required=True,
                        help="seed-42 pretraining run with the default shared checkpoint directory")
    parser.add_argument("--base-checkpoints-dir", default=str(FINETUNE_BASE_CHECKPOINTS_DIR),
                        help=f"default: {FINETUNE_BASE_CHECKPOINTS_DIR}")
    parser.add_argument("--base-step", type=int, default=FINETUNE_BASE_STEP,
                        help=f"default: final checkpoint of the main runs, step {FINETUNE_BASE_STEP}")
    parser.add_argument("--ft-data-dir", required=True, help="train_*.parquet + val_*.parquet (scripts/make_finetune_data.py)")
    parser.add_argument("--ft-name", required=True, help="dataset name used in outputs, e.g. wikitext103")
    parser.add_argument("--d0-file", required=True, help="held-out pretraining-distribution parquet (D0)")
    parser.add_argument("--eval-tokens", type=int, default=1_048_576, help="tokens scored per eval set (D0 and D1)")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-packing", choices=("bestfit", "contiguous"), default="bestfit",
                        help="bestfit: the pretraining evaluation's rows, every token with its document's "
                             "context (nanochat/eval_set.py); contiguous: the original stream packing")
    parser.add_argument("--lrs", nargs="+", required=True, help="matrix LR grid per optimizer, e.g. adamw=1e-4,2e-4 muon=2e-4")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--train-tokens", type=int, default=15_728_640, help="fine-tuning tokens per cell (spec: 10-20M)")
    parser.add_argument("--total-batch-size", type=int, default=65_536)
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="same in every cell (spec: 2-5%%)")
    parser.add_argument("--warmdown-ratio", type=float, default=0.2, help="linear decay to 0 over the last fraction")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="spec: 0 in fine-tuning")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--non-matrix-lr-scale", type=float, default=0.1,
                        help="embedding and lm_head AdamW LRs = pretraining defaults x this, identical in every cell")
    parser.add_argument("--checkpoint-spacing", choices=("log", "linear"), default="log",
                        help="log: 2, 4, 8, ... in the constant phase, its end, mid-decay, last step")
    parser.add_argument("--n-checkpoints", type=int, default=10, help="with --checkpoint-spacing linear")
    parser.add_argument("--branch-tokens", type=int, nargs="*", default=[],
                        help="further fine-tuning lengths run as decay branches of one trunk (see above)")
    parser.add_argument("--balancer-mode", choices=("active", "frozen"), default="active",
                        help="frozen: balance coefficient 0 (aux) or no bias updates (bias)")
    parser.add_argument("--out-root", default=None, help="default: <base_dir>/finetune")
    parser.add_argument("--device-type", default="")
    args = parser.parse_args(argv)
    lrs = parse_lrs(args.lrs)
    for name, grid in lrs.items():  # cell directories are named by lr_tag (3 significant digits)
        if len({lr_tag(lr) for lr in grid}) != len(grid):
            parser.error(f"--lrs {name}: learning rates must differ in their first 3 significant digits")
    for tokens in (args.train_tokens, *args.branch_tokens):
        if tokens % args.total_batch_size:
            parser.error("--train-tokens and --branch-tokens must be multiples of --total-batch-size")

    base_dir = Path(args.base_checkpoints_dir) / args.base_run
    base_step = args.base_step
    using_shared_checkpoints = Path(args.base_checkpoints_dir).resolve() == FINETUNE_BASE_CHECKPOINTS_DIR.resolve()
    if using_shared_checkpoints and args.base_run not in FINETUNE_BASE_CHECKPOINTS:
        parser.error(f"shared checkpoints are pinned to {tuple(FINETUNE_BASE_CHECKPOINTS)}")
    model_path = base_dir / f"model_{base_step:06d}.pt"
    meta_path = base_dir / f"meta_{base_step:06d}.json"
    if not model_path.is_file() or not meta_path.is_file():
        parser.error(f"missing base checkpoint or metadata: {model_path}, {meta_path}")
    checkpoint_meta = json.loads(meta_path.read_text())
    user_config = checkpoint_meta.get("user_config", {})
    if checkpoint_meta.get("step") != base_step:
        parser.error(f"checkpoint step mismatch in {meta_path}")
    if using_shared_checkpoints:
        expected_optimizer, expected_sha256 = FINETUNE_BASE_CHECKPOINTS[args.base_run]
        if (base_step != FINETUNE_BASE_STEP or user_config.get("seed") != base_run_seed(args.base_run)
                or user_config.get("optimizer_name") != expected_optimizer):
            parser.error(f"unexpected base checkpoint metadata in {meta_path}")
    model_sha256 = file_sha256(model_path)
    if using_shared_checkpoints and model_sha256 != expected_sha256:
        parser.error(f"base checkpoint SHA-256 mismatch: {model_path}")
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type, seed=0)
    model, tokenizer, base_meta = build_model(str(base_dir), base_step, device, phase="train")
    model0, _, _ = build_model(str(base_dir), base_step, device, phase="eval")
    if model.config.ffn_type != "moe":
        parser.error("fine-tuning measurements need an MoE base model (E=1 runs use ffn_type=moe with n_experts=1)")
    if args.balancer_mode == "frozen" and model.config.balancer == "aux":
        model.config.balance_coef = 0.0
    seq_len = base_meta["max_seq_len"]
    if args.total_batch_size % (args.device_batch_size * seq_len):
        parser.error("--total-batch-size must be a multiple of device batch size x sequence length")
    pretrain_parser = argparse.ArgumentParser()
    add_optimizer_args(pretrain_parser, weight_decay_default=0.1)
    pretrain_defaults = pretrain_parser.parse_args([])

    out_dir = Path(args.out_root or Path(get_base_dir()) / "finetune") / args.base_run / args.ft_name
    (out_dir / "base").mkdir(parents=True, exist_ok=True)
    ft_files = list_parquet_files(args.ft_data_dir)
    d0 = build_eval_set("D0", args.d0_file, tokenizer, seq_len, args.eval_tokens, packing=args.eval_packing)
    d1 = build_eval_set("D1", ft_files[-1], tokenizer, seq_len, args.eval_tokens, packing=args.eval_packing)
    eval_identity = {s.name: {"file": s.source_file, "sha256": s.source_sha256, "rows": s.inputs.size(0),
                              "seq_len": seq_len, "scored_tokens": s.n_scored, "documents": s.n_docs,
                              "packing": s.packing} for s in (d0, d1)}
    for s in (d0, d1):
        # atomic: concurrent jobs of the same base model and dataset write the same base files
        replace_atomic(out_dir / "base" / f"eval_set_{s.name}.pt", s.save)

    # Base model at step 0 (once per base model and dataset)
    base_d0 = evaluate_base(model0, d0, args.eval_batch_size, device)
    base_d1 = evaluate_base(model0, d1, args.eval_batch_size, device)
    layers = model.moe_layer_indices
    E = model.config.n_experts
    half = d0.n_docs // 2
    scored = d0.doc_ids >= 0
    first_half, second_half = scored & (d0.doc_ids < half), d0.doc_ids >= half
    base_summary = {"base_run": args.base_run, "base_step": base_step, "ft_dataset": args.ft_name,
                    "loss_d0": token_weighted_mean(base_d0["doc_sums"], base_d0["doc_counts"]),
                    "loss_d1": token_weighted_mean(base_d1["doc_sums"], base_d1["doc_counts"]), "layers": {}}
    for l in layers:
        s0 = load_shares(base_d0["routes"][l][scored], E)
        s1 = load_shares(base_d1["routes"][l][d1.doc_ids >= 0], E)
        base_summary["layers"][l] = {
            "load_d0": s0.tolist(), "load_d1": s1.tolist(), "maxvio_d0": maxvio(s0), "maxvio_d1": maxvio(s1),
            "separation": separation(s0, s1),
            "separation_two_halves_d0": separation(load_shares(base_d0["routes"][l][first_half], E),
                                                   load_shares(base_d0["routes"][l][second_half], E)),
            "margin_d0": base_d0["margin"][l], "margin_d1": base_d1["margin"][l]}
    if model.config.balancer == "bias":
        base_summary["expert_bias"] = model.expert_bias_state().cpu().tolist()
    write_json_atomic(out_dir / "base" / "base_metrics.json", base_summary)
    replace_atomic(out_dir / "base" / "base_arrays.npz", lambda f: np.savez_compressed(
        f, d0_doc_loss=base_d0["doc_sums"].numpy(), d0_doc_tokens=base_d0["doc_counts"].numpy(),
        d1_doc_loss=base_d1["doc_sums"].numpy(), d1_doc_tokens=base_d1["doc_counts"].numpy(),
        d0_routes=torch.stack([base_d0["routes"][l] for l in layers]).numpy(),
        d1_routes=torch.stack([base_d1["routes"][l] for l in layers]).numpy(), layers=np.array(layers)))
    print0(f"Base: D0 loss {base_summary['loss_d0']:.4f}, D1 loss {base_summary['loss_d1']:.4f}")

    ctx = {
        "model": model, "compiled": torch.compile(model, dynamic=False) if device.type == "cuda" else model,
        "model0": model0, "device": device, "tokenizer": tokenizer, "seq_len": seq_len,
        "base_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "pretrain_defaults": pretrain_defaults, "out_dir": out_dir, "base_run": args.base_run,
        "pretrain_optimizer": base_meta["user_config"].get("optimizer_name"),
        "base_identity": {"dir": str(base_dir), "step": base_step, "model_file": str(model_path),
                          "model_sha256": model_sha256, "val_bpb": base_meta.get("val_bpb"),
                          "user_config": base_meta["user_config"], "git": git_state()},
        "ft_train_files": {Path(f).name: file_sha256(f) for f in ft_files[:-1]},
        "eval_identity": eval_identity, "d0": d0, "d1": d1, "base_d0": base_d0, "base_d1": base_d1,
        "main_steps": args.train_tokens // args.total_batch_size,
        "branch_steps": [t // args.total_batch_size for t in args.branch_tokens],
        # warmup is fixed in steps by the main length, so every branch length shares it
        "warmup": math.ceil(args.warmup_ratio * (args.train_tokens // args.total_batch_size)),
        "accum": args.total_batch_size // (args.device_batch_size * seq_len),
    }
    for optimizer_name, grid in lrs.items():
        for lr in grid:
            for seed in args.seeds:
                run_cell(args, (optimizer_name, lr, seed), ctx)


if __name__ == "__main__":
    main()
