"""
The E = 8 pretraining plan (PRETRAINING_PROTOCOL.md, 2026-11-24): a short learning-rate
sweep, the selection of each optimizer's LR from it, the 3B-token base models, and the
Slurm job scripts that run them through runs/pretrain_study.sh.

Pure functions only; scripts/pretrain_sweep.py is the command-line front end.

Sweep: depth 10, top-1 MoE with E = 8, 499,908,608 tokens (1,907 steps at 262,144),
AdamW at 1e-3/2e-3/4e-3 and Muon at 2e-3/4e-3/8e-3, warmup 4.33% of the run (the same
fraction as 130M tokens of the 3B run), linear decay over the last 20%.
Selection: lowest final validation bpb (the held-out validation shard, minus D0's
documents); the LR is the vertex of the parabola through the best point and its two
neighbours in log2(LR), then multiplied by (3B / 0.5B)^-0.32 for the 3B-token runs.
If the best point is at the edge of the grid, the sweep is extended by one point (x2).
"""

import json
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

TOTAL_BATCH_SIZE = 262_144
FULL_TOKENS = 3_000_238_080           # 11,445 steps
SWEEP_TOKENS = 499_908_608            # 1,907 steps
REFERENCE_WARMUP_TOKENS = 130_000_000  # warmup of the 3B-token runs (Ludziejewski et al. Sec. 5.1.3)
TRANSFER_EXPONENT = -0.32             # LR ∝ T^-0.32 (Bjorck et al. / Ren et al., as used by Hernanz)
DEPTH = 10
HEADLINE_EXPERTS = 8
DEFAULT_SEED = 42
DEVICE_BATCH_SIZE = 32                # x 2048 = 65,536 tokens; 4 micro-batches per step on one GPU
VAL_EVAL_TOKENS = 10_485_760          # validation shard tokens per evaluation (160 micro-batches)
HELDOUT_EVAL_TOKENS = 1_048_576       # D0 tokens per evaluation (logged only)
EVAL_EVERY = 500
CHECKPOINT_EVERY = 1526               # 400M tokens: checkpoints at 0.8B and 1.6B, where 1B/2B decay branches would start
SWEEP_LRS = {"adamw": (1e-3, 2e-3, 4e-3), "muon": (2e-3, 4e-3, 8e-3)}
OPTIMIZERS = ("adamw", "muon")
LR_FLAG = {"adamw": "--adamw-matrix-lr", "muon": "--muon-matrix-lr"}
RUN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")  # what runs/pretrain_study.sh accepts


def num_steps(train_tokens, total_batch_size=TOTAL_BATCH_SIZE):
    if train_tokens % total_batch_size:
        raise ValueError(f"{train_tokens} tokens is not a whole number of {total_batch_size}-token steps")
    return train_tokens // total_batch_size


def warmup_steps(train_tokens, total_batch_size=TOTAL_BATCH_SIZE):
    """Warmup as the same fraction of the run as 130M tokens is of the 3B-token run:
    83 steps for the 0.5B sweep, 496 for 3B (= ceil(130M / 262,144))."""
    fraction = REFERENCE_WARMUP_TOKENS / FULL_TOKENS
    return math.ceil(fraction * num_steps(train_tokens, total_batch_size) - 1e-9)


def transfer_factor(from_tokens=SWEEP_TOKENS, to_tokens=FULL_TOKENS):
    return (to_tokens / from_tokens) ** TRANSFER_EXPONENT


def format_lr(lr):
    return f"{lr:.3g}"


@dataclass(frozen=True)
class RunSpec:
    stage: str            # "sweep" or "main"
    optimizer: str
    lr: float
    seed: int = DEFAULT_SEED
    n_experts: int = HEADLINE_EXPERTS

    @property
    def train_tokens(self):
        return SWEEP_TOKENS if self.stage == "sweep" else FULL_TOKENS

    @property
    def name(self):
        lr = f"-lr{format_lr(self.lr)}" if self.stage == "sweep" else ""
        return f"e{self.n_experts}-{self.stage}-{self.optimizer}{lr}-s{self.seed}"

    def base_train_args(self):
        """Arguments for runs/pretrain_study.sh train (the launcher adds the data,
        validation, held-out and log paths and the run name)."""
        return [
            f"--depth={DEPTH}", "--ffn-type=moe", f"--n-experts={self.n_experts}",
            f"--optimizer-name={self.optimizer}", f"{LR_FLAG[self.optimizer]}={self.lr!r}",
            f"--seed={self.seed}",
            f"--train-tokens={self.train_tokens}", f"--total-batch-size={TOTAL_BATCH_SIZE}",
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            f"--warmup-steps={warmup_steps(self.train_tokens)}", "--warmdown-ratio=0.2",
            f"--eval-every={EVAL_EVERY}", f"--eval-tokens={VAL_EVAL_TOKENS}",
            f"--heldout-eval-tokens={HELDOUT_EVAL_TOKENS}",
            "--core-metric-every=-1", "--sample-every=-1",
            f"--save-every={-1 if self.stage == 'sweep' else CHECKPOINT_EVERY}",
        ]


def sweep_runs(seed=DEFAULT_SEED, lrs=None):
    lrs = SWEEP_LRS if lrs is None else lrs
    return [RunSpec("sweep", optimizer, lr, seed) for optimizer in OPTIMIZERS for lr in lrs[optimizer]]


def main_runs(full_lrs, seeds=(DEFAULT_SEED,)):
    """3B-token E = 8 base models at the selected (transferred) LR of each optimizer."""
    return [RunSpec("main", optimizer, full_lrs[optimizer], seed) for seed in seeds for optimizer in OPTIMIZERS]


# ---------------------------------------------------------------------------
# Results and LR selection

def read_result(checkpoint_dir):
    """result.json written by base_train at the end of a run, or None if absent."""
    path = Path(checkpoint_dir) / "result.json"
    return json.loads(path.read_text()) if path.is_file() else None


def sweep_points(runs, checkpoints_root):
    """{optimizer: [(lr, final_val_bpb or None if diverged)]}; raises if a run has no result."""
    points, missing = {}, []
    for run in runs:
        result = read_result(Path(checkpoints_root) / run.name)
        if result is None:
            missing.append(run.name)
            continue
        if result["settings"]["matrix_lr"] != run.lr or result["settings"]["optimizer_name"] != run.optimizer:
            raise ValueError(f"{run.name}: result.json settings do not match the planned run")
        bpb = result["final_val_bpb"] if result["status"] == "completed" else None
        points.setdefault(run.optimizer, []).append((run.lr, bpb))
    if missing:
        raise FileNotFoundError(f"no result.json yet for: {', '.join(missing)}")
    return points


def parabola_vertex(xs, ys):
    """x of the vertex of the parabola through three points (any spacing)."""
    (x0, x1, x2), (y0, y1, y2) = xs, ys
    numerator = (x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)
    denominator = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
    if denominator == 0:
        return x1
    return x1 - 0.5 * numerator / denominator


def select_lr(points):
    """Choose one optimizer's LR from its sweep points [(lr, bpb or None if diverged)].

    Returns a dict with status "selected" (sweep_lr, full_lr, method) or "extend"
    (next_lr: one more point beyond the edge where the best point lies).
    """
    if len({lr for lr, _ in points}) != len(points):
        raise ValueError("duplicate learning rates in the sweep")
    points = sorted(points)
    finite = [(i, bpb) for i, (_, bpb) in enumerate(points) if bpb is not None and math.isfinite(bpb)]
    if not finite:
        raise ValueError("every sweep point diverged; extend the grid downwards by hand")
    best = min(finite, key=lambda item: item[1])[0]
    lrs = [lr for lr, _ in points]
    out = {"points": [{"lr": lr, "final_val_bpb": bpb} for lr, bpb in points], "best_grid_lr": lrs[best]}
    if best == 0:
        return {**out, "status": "extend", "next_lr": lrs[0] / 2}
    if best == len(points) - 1:
        return {**out, "status": "extend", "next_lr": lrs[-1] * 2}
    neighbours = [points[best - 1][1], points[best + 1][1]]
    if any(bpb is None or not math.isfinite(bpb) for bpb in neighbours):
        # A diverged neighbour leaves no parabola; keep the best grid point.
        sweep_lr, method = lrs[best], "best grid point (a neighbour diverged)"
    else:
        xs = [math.log2(lrs[i]) for i in (best - 1, best, best + 1)]
        ys = [points[i][1] for i in (best - 1, best, best + 1)]
        sweep_lr, method = 2 ** parabola_vertex(xs, ys), "parabola vertex in log2(LR) through the best point and its neighbours"
    return {**out, "status": "selected", "method": method, "sweep_lr": sweep_lr,
            "transfer_factor": transfer_factor(), "full_lr": sweep_lr * transfer_factor()}


# ---------------------------------------------------------------------------
# Slurm job scripts

@dataclass(frozen=True)
class SlurmSettings:
    partition: str = "mit_normal_gpu"
    gres: str = "gpu:b200:1"
    cpus: int = 16
    mem: str = "128G"


def slurm_header(job_name, time_limit, repo_dir, scratch_base=None, slurm=SlurmSettings()):
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -p {slurm.partition}",
        f"#SBATCH --gres={slurm.gres}",
        f"#SBATCH -c {slurm.cpus}",
        f"#SBATCH --mem={slurm.mem}",
        f"#SBATCH -t {time_limit}",
        "#SBATCH -o %x_%j.out",
        "set -uo pipefail",
        f"cd {shlex.quote(str(repo_dir))}",
    ]
    if scratch_base:
        lines.append(f"export NANOCHAT_SCRATCH_BASE={shlex.quote(str(scratch_base))}")
    lines += [
        'SCRATCH="${NANOCHAT_SCRATCH_BASE:-$HOME/orcd/scratch/public/nanochat-optimizer-mismatch}"',
        'POOL="${NANOCHAT_POOL_BASE:-$HOME/orcd/pool/nanochat-optimizer-mismatch}"',
        "failed=0",
    ]
    return lines


# Small result files (records, metrics, logs; not weights) are copied from scratch to pool
# storage at the end of every job, so they survive scratch purges.
PERSIST_LINES = [
    'mkdir -p "$POOL/results"',
    "for d in base_checkpoints logs finetune; do",
    '    [[ -d "$SCRATCH/$d" ]] || continue',
    "    rsync -a --prune-empty-dirs --include='*/' --include='*.json' --include='*.jsonl' --include='*.log' \\",
    "          --include='*.parquet' --exclude='*' \"$SCRATCH/$d\" \"$POOL/results/\" \\",
    '        || echo "WARNING: could not copy $d results to $POOL/results" >&2',
    "done",
]


def render_job(job_name, runs, time_limit, repo_dir, scratch_base=None, slurm=SlurmSettings()):
    """A Slurm batch script that runs `runs` one after another on one GPU.

    A run that already has result.json (completed or diverged) is skipped, so a
    resubmitted job continues where it stopped. A failed run does not stop the
    others; the job exits non-zero if any run failed. Result files are then copied
    to pool storage.
    """
    for run in runs:
        if not RUN_NAME_PATTERN.match(run.name):
            raise ValueError(f"invalid run name {run.name!r}")
    lines = slurm_header(job_name, time_limit, repo_dir, scratch_base, slurm) + [
        'LAUNCHER="${NANOCHAT_LAUNCHER:-runs/pretrain_study.sh}"',
        "export NPROC_PER_NODE=1",
        "run_one() {",
        '    local name="$1"; shift',
        '    if [[ -f "$SCRATCH/base_checkpoints/$name/result.json" ]]; then',
        '        echo "Skipping $name: result.json exists"; return 0',
        "    fi",
        '    RUN_NAME="$name" bash "$LAUNCHER" train "$@" || { echo "FAILED: $name" >&2; failed=1; }',
        "}",
    ]
    for run in runs:
        lines.append("run_one " + " ".join(shlex.quote(arg) for arg in [run.name, *run.base_train_args()]))
    lines += PERSIST_LINES
    lines.append('exit "$failed"')
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Fine-tuning jobs (scripts/finetune.py)

FINETUNE_SEEDS = (1, 2, 3, 4, 5)
# Fine-tuning LR grid per fine-tuning optimizer: fractions of that optimizer's selected
# 3B-token pretraining LR, spaced x2 (EXPERIMENT_SPEC.md §6: 5-point grid per cell).
FINETUNE_LR_FRACTIONS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)


def finetune_lr_grid(selection, fractions=FINETUNE_LR_FRACTIONS):
    return {o: [selection[o]["full_lr"] * f for f in fractions] for o in OPTIMIZERS}


def render_finetune_job(job_name, base_run, ft_name, ft_data_dir, d0_file, optimizer, lrs, time_limit, repo_dir,
                        seeds=FINETUNE_SEEDS, scratch_base=None, slurm=SlurmSettings()):
    """One process runs every (LR, seed) cell of one base model, dataset and fine-tuning optimizer."""
    if not RUN_NAME_PATTERN.match(base_run):
        raise ValueError(f"invalid run name {base_run!r}")
    args = ["--base-run", base_run, "--ft-data-dir", str(ft_data_dir), "--ft-name", ft_name, "--d0-file", str(d0_file),
            "--lrs", f"{optimizer}=" + ",".join(repr(lr) for lr in lrs), "--seeds", *map(str, seeds)]
    lines = slurm_header(job_name, time_limit, repo_dir, scratch_base, slurm) + [
        'PYTHON="${NANOCHAT_PYTHON:-.venv/bin/python}"',
        'export NANOCHAT_BASE_DIR="$SCRATCH"',
        'export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HOME/.cache/nanochat/tiktoken_cache}"',
        'export WANDB_MODE=disabled',
        '"$PYTHON" -m scripts.finetune ' + " ".join(shlex.quote(a) for a in args) + " || failed=1",
    ]
    lines += PERSIST_LINES
    lines.append('exit "$failed"')
    return "\n".join(lines) + "\n"
