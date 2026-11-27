"""
Shared command-line options, setting resolution and run records for the
optimizer-mismatch experiments. Pretraining (scripts/base_train.py), chat SFT
(scripts/chat_sft.py) and language-model fine-tuning (scripts/lm_finetune.py)
all build their optimizer through these helpers, so a setting means the same
thing at every stage.
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch

from nanochat.gpt import BALANCERS, FFN_TYPES, MOE_DISPATCH, resolve_routing_loss
from nanochat.optim import MUON_VARIANTS

# Default matrix learning rate per Muon variant. "moonlight" uses Moonlight's
# update scaling, which matches AdamW's update RMS, so its learning rate is on
# the AdamW scale. The study still sweeps Muon's LR separately from AdamW's
# rather than copying it: the sweep is centered on Hernanz's 4e-3 for Muon and
# 2e-3 * E^-0.25 for AdamW. "keller" uses the original aspect-ratio
# scaling, whose learning rates are ~10x larger. Every default is only the
# center of a sweep; record the value actually used.
MUON_MATRIX_LR_DEFAULTS = {"moonlight": 4e-3, "keller": 0.02}
ADAMW_MATRIX_LR_DEFAULT = 2e-3
REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Command-line options

def add_moe_args(parser):
    group = parser.add_argument_group("feed-forward / MoE")
    group.add_argument("--ffn-type", choices=FFN_TYPES, default="dense",
                       help="dense: ReLU^2 MLP (nanochat); swiglu: dense SwiGLU (= one expert); moe: top-1 MoE of SwiGLU experts")
    group.add_argument("--n-experts", type=int, default=4, help="experts per MoE layer (E=1 is the dense control)")
    group.add_argument("--moe-layers", type=str, default="all",
                       help='which blocks are MoE: "all", "every:K" (blocks K-1, 2K-1, ...), or a list like "0,2,5"; other blocks use a dense SwiGLU MLP')
    group.add_argument("--balancer", choices=BALANCERS, default="aux",
                       help="aux: Switch auxiliary loss; bias: auxiliary-loss-free selection bias; none")
    group.add_argument("--balance-coef", type=float, default=0.01, help="auxiliary balance loss weight (balancer=aux)")
    group.add_argument("--z-coef", type=float, default=None,
                       help="router z-loss weight (default: 0.001 with balancer=aux, 0 otherwise)")
    group.add_argument("--disable-routing-loss", action="store_true",
                       help="E=1 MoE dense control only: set balance and z-loss coefficients to zero, retain routing diagnostics")
    group.add_argument("--bias-update-rate", type=float, default=0.001, help="bias balancer step size u (balancer=bias)")
    group.add_argument("--moe-dispatch", choices=MOE_DISPATCH, default="auto",
                       help="auto: grouped GEMM for bf16 on CUDA, else loop; grouped; loop (reference). Same maths, different speed")
    return group


def add_optimizer_args(parser, *, weight_decay_default, require_optimizer=False):
    group = parser.add_argument_group("optimizer")
    group.add_argument("--optimizer-name", choices=("adamw", "muon"), default=None if require_optimizer else "adamw",
                       required=require_optimizer, help="optimizer for attention and feed-forward matrices")
    group.add_argument("--router-optimizer", choices=("same", "adamw", "muon"), default="same",
                       help="optimizer for MoE routers; 'same' (the spec's decision) follows --optimizer-name")
    group.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (AdamW, every cell)")
    group.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for the lm_head (AdamW, every cell)")
    group.add_argument("--router-lr", type=float, default=None,
                       help="router learning rate (default: the matrix LR of the optimizer the router uses)")
    group.add_argument("--adamw-matrix-lr", type=float, default=ADAMW_MATRIX_LR_DEFAULT,
                       help="learning rate for transformer matrices in AdamW mode (used as given, no width scaling)")
    group.add_argument("--adamw-beta1", type=float, default=0.9, help="AdamW beta1 for every AdamW parameter group")
    group.add_argument("--adamw-beta2", type=float, default=0.95, help="AdamW beta2 for every AdamW parameter group")
    group.add_argument("--adamw-eps", type=float, default=1e-8, help="AdamW epsilon for every AdamW parameter group")
    group.add_argument("--muon-variant", choices=MUON_VARIANTS, default="moonlight",
                       help="moonlight: Nesterov + Newton-Schulz + Moonlight 0.2*sqrt(max(m,n)) scaling + decoupled WD (arXiv:2502.16982); "
                            "keller: original aspect-ratio scaling sqrt(max(1, m/n))")
    group.add_argument("--muon-matrix-lr", type=float, default=None,
                       help=f"learning rate for transformer matrices in Muon mode (default per variant: {MUON_MATRIX_LR_DEFAULTS})")
    group.add_argument("--muon-momentum", type=float, default=0.95, help="Muon momentum (constant schedule)")
    group.add_argument("--muon-momentum-schedule", choices=("constant", "nanochat"), default="constant",
                       help="constant: --muon-momentum throughout; nanochat: 0.85 -> 0.97 over 400 steps, -> 0.90 in warmdown")
    group.add_argument("--muon-ns-steps", type=int, default=5, help="Muon Newton-Schulz iteration count")
    group.add_argument("--weight-decay", type=float, default=weight_decay_default,
                       help="decoupled weight decay on transformer matrices, for AdamW and Muon alike")
    group.add_argument("--router-weight-decay", type=float, default=0.0, help="decoupled weight decay on MoE routers")
    group.add_argument("--grad-clip", type=float, default=1.0,
                       help="global gradient-norm clipping threshold (<= 0 disables; the pre-clip norm is logged either way)")
    return group


# -----------------------------------------------------------------------------
# Resolution of defaults that depend on other settings

def resolve_optimizer_args(args):
    """Fill defaults that depend on other options, in place, and return the
    keyword arguments for GPT.setup_optimizer."""
    if args.muon_matrix_lr is None:
        args.muon_matrix_lr = MUON_MATRIX_LR_DEFAULTS[args.muon_variant]
    return dict(
        optimizer_name=args.optimizer_name,
        router_optimizer=args.router_optimizer,
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        router_lr=args.router_lr,
        adamw_matrix_lr=args.adamw_matrix_lr,
        adamw_beta1=args.adamw_beta1,
        adamw_beta2=args.adamw_beta2,
        adamw_eps=args.adamw_eps,
        muon_matrix_lr=args.muon_matrix_lr,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
        muon_variant=args.muon_variant,
        weight_decay=args.weight_decay,
        router_weight_decay=args.router_weight_decay,
        max_grad_norm=args.grad_clip if args.grad_clip and args.grad_clip > 0 else None,
        track_grad_norm=True,
    )


def resolve_moe_args(args):
    """Fill resolved routing-loss coefficients in place and return GPTConfig keywords."""
    if args.z_coef is None:
        args.z_coef = 0.001 if args.balancer == "aux" else 0.0
    args.balance_coef, args.z_coef = resolve_routing_loss(
        args.ffn_type, args.n_experts, args.disable_routing_loss, args.balance_coef, args.z_coef,
    )
    return dict(
        ffn_type=args.ffn_type,
        n_experts=args.n_experts,
        moe_layers=args.moe_layers,
        balancer=args.balancer,
        balance_coef=args.balance_coef,
        z_coef=args.z_coef,
        disable_routing_loss=args.disable_routing_loss,
        bias_update_rate=args.bias_update_rate,
        moe_dispatch=args.moe_dispatch,
    )


def muon_momentum_at(step, num_iterations, warmdown_ratio, args):
    """Muon momentum for an optimizer step."""
    if args.muon_momentum_schedule == "constant":
        return args.muon_momentum
    # nanochat schedule: warm up to 0.97 over 400 steps, down to 0.90 during LR warmdown
    warmdown_iters = round(warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if step < 400:
        frac = step / 400
        return (1 - frac) * 0.85 + frac * 0.97
    if warmdown_iters > 0 and step >= warmdown_start:
        progress = (step - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    return 0.97


def wsd_lr_multiplier(step, num_iterations, warmup_iters, decay_iters, final_lr_frac):
    """Warmup-stable-decay: linear warmup, constant, linear decay to final_lr_frac."""
    if warmup_iters > 0 and step < warmup_iters:
        return (step + 1) / warmup_iters
    if decay_iters <= 0 or step <= num_iterations - decay_iters:
        return 1.0
    progress = (num_iterations - step) / decay_iters
    return progress * 1.0 + (1 - progress) * final_lr_frac


# -----------------------------------------------------------------------------
# Run records

def _git(*cmd):
    try:
        return subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_state():
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "dirty_files": None if not status else status.splitlines(),
    }


def file_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_identity(tokenizer_dir):
    tokenizer_dir = Path(tokenizer_dir)
    return {
        "dir": str(tokenizer_dir),
        "files": {path.name: file_sha256(path) for path in sorted(tokenizer_dir.glob("*")) if path.is_file()}
                 if tokenizer_dir.is_dir() else None,
    }


def optimizer_group_summary(optimizer):
    groups = []
    for group in optimizer.param_groups:
        entry = {key: value for key, value in group.items() if key != "params"}
        if "betas" in entry:
            entry["betas"] = list(entry["betas"])
        entry["num_tensors"] = len(group["params"])
        entry["num_params"] = sum(p.numel() for p in group["params"])
        entry["shapes"] = sorted({tuple(p.shape) for p in group["params"]})
        groups.append(entry)
    return {
        "class": type(optimizer).__name__,
        "max_grad_norm": getattr(optimizer, "max_grad_norm", None),
        "groups": groups,
    }


def environment_summary(device_type, world_size):
    env = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_type": device_type,
        "world_size": world_size,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if device_type == "cuda" and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def write_run_record(path, *, args, model, optimizer, device_type, world_size, compute_dtype, data, extra=None):
    """Write config.json: every resolved setting needed to reproduce the run.

    Includes the full CLI arguments, model config and MoE layers, parameter
    counts under each convention, every optimizer parameter group, seed, data
    and tokenizer identity, precision, git commit and hardware.
    """
    record = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": vars(args),
        "model_config": asdict(model.config),
        "moe_layer_indices": model.moe_layer_indices,
        "parameter_counts": model.parameter_counts(),
        "optimizer": optimizer_group_summary(optimizer),
        "compute_dtype": str(compute_dtype),
        "data": data,
        "git": git_state(),
        "environment": environment_summary(device_type, world_size),
    }
    if extra:
        record.update(extra)
    write_json_atomic(path, record)
    return record


def replace_atomic(path, write):
    """Call write(file object) on a temporary file next to `path`, then rename it over `path`, so
    readers never see a half-written file. The temporary name is unique, so concurrent writers of
    the same path (e.g. several fine-tuning jobs writing one base directory) do not collide."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    umask = os.umask(0)
    os.umask(umask)
    try:
        os.chmod(tmp, 0o666 & ~umask)  # mkstemp makes 0600; keep the usual permissions (other accounts read results)
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_json_atomic(path, data):
    """Write JSON so that readers never see a half-written file."""
    text = json.dumps(data, indent=2, default=str).encode()
    replace_atomic(path, lambda f: f.write(text))
