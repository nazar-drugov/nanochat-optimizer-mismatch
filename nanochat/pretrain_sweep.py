"""
Pretraining plan (PRETRAINING_PROTOCOL.md): short learning-rate sweeps, local
selection of each optimizer's LR at each E, 3B-token transfer checks and bases,
and Slurm job scripts through runs/pretrain_study.sh.

Pure functions only; scripts/pretrain_sweep.py is the command-line front end.

Sweep: depth 10, top-1 MoE with E = 8, 499,908,608 tokens (1,907 steps at 262,144),
AdamW at 1e-3/2e-3/4e-3 and Muon at 2e-3/4e-3/8e-3, warmup 4.33% of the run (the same
fraction as 130M tokens of the 3B run), linear decay over the last 20%.
Selection: lowest final validation bpb (the held-out validation shard, minus D0's
documents); the LR is the vertex of the parabola through the best point and its two
neighbours in log2(LR), then multiplied by (3B / 0.5B)^-0.32 for the 3B-token runs.
If the best point is at the edge of the grid, the sweep is extended by one point (x2).
Transfer check (2026-11-25): 3B-token runs at x0.5 and x2 the transferred LR (the main
run of the same seed is x1 for E8); the parabola vertex through the three shows how
far the transferred LR is from the 3B optimum. New E=1/4/16 use five-point initial
grids, separate x1 transfer directories with full checkpoint cadence, and an
explicit logical-base manifest if the checked center is retained.
"""

import json
import hashlib
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
EXPERT_COUNTS = (1, 4, 8, 16)
DEFAULT_SEED = 42
DEVICE_BATCH_SIZE = 32                # x 2048 = 65,536 tokens; 4 micro-batches per step on one GPU
VAL_EVAL_TOKENS = 10_485_760          # validation shard tokens per evaluation (160 micro-batches)
HELDOUT_EVAL_TOKENS = 1_048_576       # D0 tokens per evaluation (logged only)
EVAL_EVERY = 500
CHECKPOINT_EVERY = 1526               # 400M tokens: checkpoints at 0.8B and 1.6B, where 1B/2B decay branches would start
SWEEP_LRS = {"adamw": (1e-3, 2e-3, 4e-3), "muon": (2e-3, 4e-3, 8e-3)}
TRANSFER_CHECK_FACTORS = (0.5, 2.0)   # x the transferred 3B LR; x1 is the main run
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


def sweep_lr_grid(n_experts=HEADLINE_EXPERTS):
    """Actual historic E8 grid; independent five-point grids for new E."""
    if n_experts not in EXPERT_COUNTS:
        raise ValueError(f"unsupported expert count {n_experts!r}")
    if n_experts == HEADLINE_EXPERTS:
        return SWEEP_LRS
    centers = {"adamw": .002 * n_experts ** -.25,
               "muon": .004 * 2 ** (-EXPERT_COUNTS.index(n_experts) / 4)}
    return {o: tuple(centers[o] * f for f in (.25, .5, 1, 2, 4)) for o in OPTIMIZERS}


@dataclass(frozen=True)
class RunSpec:
    stage: str            # "sweep" (0.5B), "main" (3B base model) or "xfer" (3B transfer check)
    optimizer: str
    lr: float
    seed: int = DEFAULT_SEED
    n_experts: int = HEADLINE_EXPERTS
    retain_checkpoints: bool = False  # new-E transfer center can serve as final base

    def __post_init__(self):
        if self.stage not in ("sweep", "main", "xfer") or self.optimizer not in OPTIMIZERS:
            raise ValueError("invalid pretraining stage or optimizer")
        if self.n_experts not in EXPERT_COUNTS or self.seed < 0:
            raise ValueError("invalid expert count or seed")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("learning rate must be positive and finite")

    @property
    def train_tokens(self):
        return SWEEP_TOKENS if self.stage == "sweep" else FULL_TOKENS

    @property
    def keeps_checkpoints(self):
        return self.stage == "main" or self.retain_checkpoints

    @property
    def name(self):
        lr = "" if self.stage == "main" else f"-lr{format_lr(self.lr)}"
        return f"e{self.n_experts}-{self.stage}-{self.optimizer}{lr}-s{self.seed}"

    def base_train_args(self):
        """Arguments for runs/pretrain_study.sh train (the launcher adds the data,
        validation, held-out and log paths and the run name)."""
        return [
            f"--depth={DEPTH}", "--ffn-type=moe", f"--n-experts={self.n_experts}",
            *(["--disable-routing-loss"] if self.n_experts == 1 else []),
            f"--optimizer-name={self.optimizer}", f"{LR_FLAG[self.optimizer]}={self.lr!r}",
            f"--seed={self.seed}",
            f"--train-tokens={self.train_tokens}", f"--total-batch-size={TOTAL_BATCH_SIZE}",
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            f"--warmup-steps={warmup_steps(self.train_tokens)}", "--warmdown-ratio=0.2",
            f"--eval-every={EVAL_EVERY}", f"--eval-tokens={VAL_EVAL_TOKENS}",
            f"--heldout-eval-tokens={HELDOUT_EVAL_TOKENS}",
            "--core-metric-every=-1", "--sample-every=-1",
            f"--save-every={CHECKPOINT_EVERY if self.keeps_checkpoints else -1}",
        ]


def checked_run_names(runs):
    """Reject different exact LRs that collide under the historic name rounding."""
    if len(set(r.name for r in runs)) != len(runs):
        raise ValueError("duplicate run names (possibly rounded LR collision)")
    return runs


def sweep_runs(seed=DEFAULT_SEED, lrs=None, n_experts=HEADLINE_EXPERTS):
    lrs = sweep_lr_grid(n_experts) if lrs is None else lrs
    return checked_run_names([RunSpec("sweep", optimizer, lr, seed, n_experts)
                              for optimizer in OPTIMIZERS for lr in lrs[optimizer]])


def main_runs(full_lrs, seeds=(DEFAULT_SEED,), n_experts=HEADLINE_EXPERTS):
    """3B-token bases at the selected LR of each optimizer."""
    return checked_run_names([RunSpec("main", optimizer, full_lrs[optimizer], seed, n_experts)
                              for seed in seeds for optimizer in OPTIMIZERS])


def transfer_check_runs(full_lrs, seed=DEFAULT_SEED, factors=None, n_experts=HEADLINE_EXPERTS):
    """3B-token runs at `factors` x each optimizer's transferred LR (the main run is x1).
    `factors` is one tuple for every optimizer or {optimizer: tuple} (e.g. to extend the
    check past an edge for one optimizer)."""
    # New E uses a separately named full-budget center trial, never a provisional
    # main path. E8 keeps the historical center-as-main workflow.
    if factors is None:
        factors = TRANSFER_CHECK_FACTORS
    per_optimizer = factors if isinstance(factors, dict) else {o: factors for o in OPTIMIZERS}
    return checked_run_names([
        RunSpec("xfer", optimizer, full_lrs[optimizer] * f, seed, n_experts,
                retain_checkpoints=(n_experts != HEADLINE_EXPERTS and f == 1))
        for optimizer in OPTIMIZERS
        for f in sorted(set(per_optimizer[optimizer]) | ({1.0} if n_experts != HEADLINE_EXPERTS else set()))
    ])


# ---------------------------------------------------------------------------
# Results and LR selection

def read_result(checkpoint_dir):
    """result.json written by base_train at the end of a run, or None if absent."""
    path = Path(checkpoint_dir) / "result.json"
    return json.loads(path.read_text()) if path.is_file() else None


def validate_pretrain_result(run, result):
    """Check resolved results, strictly for every new-E experiment.

    The old E8 result reader retains compatibility with its historical records.
    """
    expected = {"matrix_lr": run.lr, "optimizer_name": run.optimizer}
    if run.n_experts != HEADLINE_EXPERTS:
        expected.update(n_experts=run.n_experts, seed=run.seed, ffn_type="moe", depth=DEPTH,
                        total_batch_size=TOTAL_BATCH_SIZE, train_tokens=run.train_tokens,
                        warmup_steps=warmup_steps(run.train_tokens), warmdown_ratio=.2,
                        weight_decay=.1, max_grad_norm=1.,
                        disable_routing_loss=run.n_experts == 1,
                        balancer="aux",
                        balance_coef=0. if run.n_experts == 1 else .01,
                        z_coef=0. if run.n_experts == 1 else .001)
    settings = result.get("settings", {})
    if any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{run.name}: result.json settings do not match the planned run")
    if result.get("status") not in ("completed", "diverged"):
        raise ValueError(f"{run.name}: invalid or unfinished result")
    if run.n_experts != HEADLINE_EXPERTS:
        if result.get("run") != run.name or result.get("num_iterations") != num_steps(run.train_tokens):
            raise ValueError(f"{run.name}: wrong run identity or target length")
        if result["status"] == "completed" and (
            result.get("steps_trained") != num_steps(run.train_tokens) or
            result.get("tokens_trained") != run.train_tokens or
            not isinstance(result.get("final_val_bpb"), (int, float)) or
            not math.isfinite(result["final_val_bpb"]) or result["final_val_bpb"] <= 0
        ):
            raise ValueError(f"{run.name}: incomplete completed-run result or non-finite validation")


def sweep_points(runs, checkpoints_root):
    """{optimizer: [(lr, final_val_bpb or None if diverged)]}; raises if a run has no result."""
    checked_run_names(runs)
    if len({(r.n_experts, r.seed, r.train_tokens) for r in runs}) > 1:
        raise ValueError("cannot pool distinct expert counts, seeds, or token budgets")
    points, missing = {}, []
    for run in runs:
        result = read_result(Path(checkpoints_root) / run.name)
        if result is None:
            missing.append(run.name)
            continue
        validate_pretrain_result(run, result)
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


def check_transfer(points, transferred_lr, sweep_lr, from_tokens=SWEEP_TOKENS, to_tokens=FULL_TOKENS):
    """Compare the transferred LR with the 3B optimum from the transfer-check points
    [(lr, bpb or None)] (x0.5, x1, x2 the transferred LR).

    Returns the best point, the parabola vertex in log2(LR) when the best point is
    interior (with its offset from the transferred LR in octaves), and the exponent that
    the vertex implies for LR* ∝ T^beta between the sweep and the full run.
    """
    points = sorted(points)
    finite = [(lr, bpb) for lr, bpb in points if bpb is not None and math.isfinite(bpb)]
    if not finite:
        raise ValueError("every transfer-check run diverged")
    best_lr, best_bpb = min(finite, key=lambda p: p[1])
    out = {"points": [{"lr": lr, "final_val_bpb": bpb} for lr, bpb in points],
           "transferred_lr": transferred_lr, "best_lr": best_lr, "best_bpb": best_bpb,
           "assumed_exponent": TRANSFER_EXPONENT}
    i = [lr for lr, _ in points].index(best_lr)
    if 0 < i < len(points) - 1 and all(points[j][1] is not None and math.isfinite(points[j][1])
                                       for j in (i - 1, i + 1)):
        xs = [math.log2(points[j][0]) for j in (i - 1, i, i + 1)]
        ys = [points[j][1] for j in (i - 1, i, i + 1)]
        vertex = 2 ** parabola_vertex(xs, ys)
        out.update(status="interior", vertex_lr=vertex,
                   vertex_octaves_from_transferred=math.log2(vertex / transferred_lr),
                   implied_exponent=math.log(vertex / sweep_lr) / math.log(to_tokens / from_tokens))
    elif i == 0 or i == len(points) - 1:
        out["status"] = "edge"  # the 3B optimum lies beyond the checked range
        next_lr = best_lr / 2 if i == 0 else best_lr * 2
        out["next_factor"] = next_lr / transferred_lr
    else:
        # There are tried LRs on each side, but a divergent neighbor prevents
        # a local fit. This is not an edge; do not auto-extend into divergence.
        out["status"] = "unfittable"
        out["reason"] = "interior grid best has a diverged/nonfinite neighbor"
    return out


TRANSFER_RERUN_OCTAVES = 0.25  # protocol 2026-11-25: rerun the base models beyond this offset


def selection_after_transfer(selection, report, threshold=TRANSFER_RERUN_OCTAVES):
    """The LR selection for the 3B base models after the transfer check: per optimizer, the
    transferred LR if the 3B optimum is within `threshold` octaves of it, else the 3B optimum
    (the transfer-check vertex). An optimizer whose check ended at an edge is not selected."""
    out = {}
    for optimizer, choice in selection.items():
        r, c = report[optimizer], dict(choice)
        c["transferred_lr"] = choice["full_lr"]
        if r["status"] != "interior":
            c.update(status=f"transfer-check-{r['status']}", full_lr_source=None)
        elif abs(r["vertex_octaves_from_transferred"]) > threshold:
            c.update(full_lr=r["vertex_lr"], full_lr_source="transfer-check vertex",
                     octaves_from_transferred=r["vertex_octaves_from_transferred"])
        else:
            c.update(full_lr_source="transferred", octaves_from_transferred=r["vertex_octaves_from_transferred"])
        out[optimizer] = c
    return out


# ---------------------------------------------------------------------------
# Slurm job scripts

@dataclass(frozen=True)
class SlurmSettings:
    partition: str = "mit_normal_gpu"
    gres: str = "gpu:b200:1"
    cpus: int = 16
    mem: str = "128G"
    # Nodes to avoid: they mount /orcd/scratch/orcd/014 but not our directory in it
    # (node5800-c1: job 30416147, 2026-11-24; node5702-c1: job 30427014, 2026-11-25).
    exclude: str = "node5800-c1,node5702-c1"


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
    ]
    if slurm.exclude:
        lines.append(f"#SBATCH --exclude={slurm.exclude}")
    lines += [
        "set -uo pipefail",
        f"cd {shlex.quote(str(repo_dir))}",
    ]
    if scratch_base:
        lines.append(f"export NANOCHAT_SCRATCH_BASE={shlex.quote(str(scratch_base))}")
    lines += [
        'SCRATCH="${NANOCHAT_SCRATCH_BASE:-$HOME/orcd/scratch/public/nanochat-optimizer-mismatch}"',
        'POOL="${NANOCHAT_POOL_BASE:-$HOME/orcd/pool/nanochat-optimizer-mismatch}"',
        # Some nodes do not see every scratch directory; fail at once and name the node.
        'if [[ ! -d "$SCRATCH" ]]; then',
        '    echo "Scratch directory $SCRATCH is not visible on $(hostname); resubmit with --exclude=$(hostname)" >&2',
        "    exit 1",
        "fi",
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
    checked_run_names(runs)
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
        # The launcher derives canonical E/optimizer/seed/stage W&B metadata
        # from RUN_NAME, with xfer mapped to the established transfer group.
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
FINETUNE_BASE_CHECKPOINTS_DIR = Path(
    "/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/base_checkpoints"
)
FINETUNE_BASE_STEP = 11445
FINETUNE_BASE_CHECKPOINTS = {
    "e8-main-adamw-s42": ("adamw", "de13193a8de30cb460a891b97bf67d195b97d611fc53d26b37b523507fcbd8d2"),
    "e8-main-muon-s42": ("muon", "2eff018763ccde5fd70b3a60df18a3e0a812f4b126ec0860fac8cf0be286c926"),
    "e8-main-adamw-s43": ("adamw", "78f88110ad8efad1243d4e9f265959d51774f2ef38248b7bc115e0c27f3c382a"),
    "e8-main-muon-s43": ("muon", "5ea0be1bfdd93e1fc8ec9115f27db6a7aff467e5ad0686866e1681402a4785bd"),
    "e8-main-adamw-s44": ("adamw", "f893d26a7261ef6c2faf5ab3a2aefaa52f9de15d7e22a821c17e91f358af9035"),
    "e8-main-muon-s44": ("muon", "13e89b545f96d2b8a34d6bab008d97496cfd142b98d505044c9b56a815b40c23"),
}


def base_run_identity(base_run):
    """(E, optimizer, seed) from a logical main base name."""
    match = re.fullmatch(r"e(1|4|8|16)-main-(adamw|muon)-s(\d+)", base_run)
    if match is None:
        raise ValueError(f"not a main base run name: {base_run!r}")
    return int(match.group(1)), match.group(2), int(match.group(3))


def base_run_seed(base_run):
    return base_run_identity(base_run)[2]


def base_run_experts(base_run):
    return base_run_identity(base_run)[0]


@dataclass(frozen=True)
class BasePin:
    """Logical main identity plus actual source checkpoint directory and digest."""
    checkpoint_dir: Path
    step: int
    optimizer: str
    n_experts: int
    seed: int
    matrix_lr: float | None
    model_sha256: str


def _manifest_pins(document):
    if not isinstance(document, dict) or document.get("version") != 1 or not isinstance(document.get("bases"), dict):
        raise ValueError("invalid base manifest: expected version 1 and bases mapping")
    pins = {}
    required = {"checkpoint_dir", "step", "optimizer", "n_experts", "seed", "matrix_lr", "model_sha256"}
    for name, record in document["bases"].items():
        if not isinstance(record, dict) or not required <= set(record):
            raise ValueError(f"{name}: incomplete base manifest entry")
        expert_count, optimizer, seed = base_run_identity(name)
        if (record["optimizer"], record["n_experts"], record["seed"], record["step"]) != (
            optimizer, expert_count, seed, FINETUNE_BASE_STEP
        ):
            raise ValueError(f"{name}: wrong base pin identity or checkpoint step")
        if not isinstance(record["matrix_lr"], (int, float)) or not math.isfinite(record["matrix_lr"]) or record["matrix_lr"] <= 0:
            raise ValueError(f"{name}: invalid base pin matrix LR")
        lr = float(record["matrix_lr"])
        directory = Path(record["checkpoint_dir"])
        allowed_source = {name, RunSpec("xfer", optimizer, lr, seed, expert_count).name}
        if not directory.is_absolute() or directory.name not in allowed_source or ".." in directory.parts:
            raise ValueError(f"{name}: base pin directory must be an absolute actual main or selected xfer source path")
        sha256 = record["model_sha256"]
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"{name}: invalid model SHA-256")
        if name in FINETUNE_BASE_CHECKPOINTS and (directory != FINETUNE_BASE_CHECKPOINTS_DIR / name or
                                                 sha256 != FINETUNE_BASE_CHECKPOINTS[name][1]):
            raise ValueError(f"{name}: manifest cannot replace a historical E8 base pin")
        pins[name] = BasePin(directory, FINETUNE_BASE_STEP, optimizer, expert_count, seed, lr, sha256)
    if not pins:
        raise ValueError("empty base manifest")
    return pins


def load_base_manifest(path):
    return _manifest_pins(json.loads(Path(path).read_text()))


def resolve_base_pin(base_run, manifest=None):
    """Resolve a validated explicit pin; never infer a new-E checkpoint from its name."""
    expert_count, optimizer, seed = base_run_identity(base_run)
    if manifest is not None:
        if isinstance(manifest, (str, Path)):
            pins = load_base_manifest(manifest)
        elif isinstance(manifest, dict) and "version" in manifest:
            pins = _manifest_pins(manifest)
        else:
            pins = manifest
        if base_run in pins:
            pin = pins[base_run]
            if not isinstance(pin, BasePin) or (pin.n_experts, pin.optimizer, pin.seed) != (expert_count, optimizer, seed):
                raise ValueError(f"{base_run}: invalid decoded base pin")
            return pin
    if base_run in FINETUNE_BASE_CHECKPOINTS:
        return BasePin(FINETUNE_BASE_CHECKPOINTS_DIR / base_run, FINETUNE_BASE_STEP, optimizer,
                       expert_count, seed, None, FINETUNE_BASE_CHECKPOINTS[base_run][1])
    raise ValueError(f"{base_run}: new base run requires a pinned base manifest entry")


def completed_base_manifest(selection, checkpoints_root, n_experts, seeds=(DEFAULT_SEED,)):
    """Inspect actual completed full-budget bases and hash actual final model weights.

    A transfer center may serve as the logical seed-42 main base only when its
    exact LR is retained by the transfer check. Never promote a side trial.
    """
    entries = {}
    root = Path(checkpoints_root).resolve()
    for run in main_runs({o: selection[o]["full_lr"] for o in OPTIMIZERS}, seeds, n_experts):
        choice = selection[run.optimizer]
        if (choice.get("status") != "selected" or choice.get("full_lr_source") not in
                ("transferred", "transfer-check vertex") or choice.get("n_experts") != n_experts):
            raise ValueError(f"{run.name}: manifest requires final validated LR selection for this E")
        source = run
        if (n_experts != HEADLINE_EXPERTS and run.seed == choice.get("seed") and
                choice["full_lr_source"] == "transferred" and
                choice.get("transferred_lr") == run.lr):
            source = RunSpec("xfer", run.optimizer, run.lr, run.seed, n_experts, retain_checkpoints=True)
        directory = root / source.name
        result = read_result(directory)
        if result is None:
            raise FileNotFoundError(f"{run.name}: no completed result at {directory}")
        validate_pretrain_result(source, result)
        if result["status"] != "completed":
            raise ValueError(f"{run.name}: diverged pretraining cannot be used as a base")
        model_file = directory / f"model_{FINETUNE_BASE_STEP:06d}.pt"
        meta_file = directory / f"meta_{FINETUNE_BASE_STEP:06d}.json"
        if not model_file.is_file() or not meta_file.is_file():
            raise FileNotFoundError(f"{run.name}: missing final model weights or checkpoint metadata")
        meta = json.loads(meta_file.read_text())
        user = meta.get("user_config", {})
        shape = meta.get("model_config", {})
        if (meta.get("step") != FINETUNE_BASE_STEP or user.get("seed") != run.seed or
                user.get("optimizer_name") != run.optimizer or user.get(f"{run.optimizer}_matrix_lr") != run.lr or
                shape.get("n_experts") != n_experts or shape.get("ffn_type") != "moe" or
                meta.get("train_tokens") != FULL_TOKENS or meta.get("total_batch_size") != TOTAL_BATCH_SIZE):
            raise ValueError(f"{run.name}: final checkpoint metadata disagrees with selected base")
        if n_experts != HEADLINE_EXPERTS:
            routing_policy = {"disable_routing_loss": n_experts == 1,
                              "balancer": "aux",
                              "balance_coef": 0. if n_experts == 1 else .01,
                              "z_coef": 0. if n_experts == 1 else .001}
            if any(record.get(field) != value for record in (user, shape)
                   for field, value in routing_policy.items()):
                raise ValueError(f"{run.name}: base checkpoint routing-loss policy disagrees with the approved recipe")
        digest = hashlib.sha256()
        with model_file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        entries[run.name] = {
            "checkpoint_dir": str(directory), "step": FINETUNE_BASE_STEP, "optimizer": run.optimizer,
            "n_experts": n_experts, "seed": run.seed, "matrix_lr": run.lr,
            "model_sha256": digest.hexdigest(),
        }
    document = {"version": 1, "bases": entries}
    _manifest_pins(document)
    return document
# Fine-tuning LR grid per fine-tuning optimizer: fractions of that optimizer's selected
# 3B-token pretraining LR, spaced x2 (EXPERIMENT_SPEC.md §6: 5-point grid per cell).
FINETUNE_LR_FRACTIONS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)


def finetune_lr_grid(selection, fractions=FINETUNE_LR_FRACTIONS):
    return {o: [selection[o]["full_lr"] * f for f in fractions] for o in OPTIMIZERS}


def render_finetune_job(job_name, base_run, ft_name, ft_data_dir, d0_file, optimizer, lrs, time_limit, repo_dir,
                        seeds=FINETUNE_SEEDS, scratch_base=None, slurm=SlurmSettings(), extra_args=(), base_manifest=None):
    """One process runs every (LR, seed) cell of one base model, dataset and fine-tuning optimizer."""
    pin = resolve_base_pin(base_run, base_manifest)
    args = ["--base-run", base_run, "--base-checkpoints-dir", str(pin.checkpoint_dir.parent),
            "--base-step", str(pin.step),
            *(["--base-manifest", str(Path(base_manifest).resolve())] if base_manifest is not None else []),
            "--ft-data-dir", str(ft_data_dir), "--ft-name", ft_name, "--d0-file", str(d0_file),
            "--lrs", f"{optimizer}=" + ",".join(repr(lr) for lr in lrs), "--seeds", *map(str, seeds),
            *map(str, extra_args)]
    extra = list(map(str, extra_args))
    out_root = extra[extra.index("--out-root") + 1].rstrip("/") if "--out-root" in extra else None
    root = shlex.quote(out_root) if out_root else '"$SCRATCH/finetune"'
    stage = Path(out_root).name if out_root else "finetune"
    sync_args = " ".join(shlex.quote(a) for a in ["--stage", stage, "--dataset", ft_name,
                                                    "--base-run", base_run, "--ft-optimizer", optimizer,
                                                    *(["--base-manifest", str(Path(base_manifest).resolve())]
                                                      if base_manifest is not None else [])])
    lines = slurm_header(job_name, time_limit, repo_dir, scratch_base, slurm) + [
        'PYTHON="${NANOCHAT_PYTHON:-.venv/bin/python}"',
        'export NANOCHAT_BASE_DIR="$SCRATCH"',
        'export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HOME/.cache/nanochat/tiktoken_cache}"',
        'export PYTHONUNBUFFERED=1  # progress lines reach the Slurm log as they happen',
        '"$PYTHON" -m scripts.finetune ' + " ".join(shlex.quote(a) for a in args) + " || failed=1",
        # Log finalized branch histories, rather than doing network I/O in the GPU training loop.
        # Keep local metrics authoritative and retain offline recordings if online upload fails.
        f"FT_ROOT={root}",
        'export WANDB_DIR="${WANDB_DIR:-$SCRATCH}"',
        'netrc_file="${NETRC:-$HOME/.netrc}"',
        'wb_mode=offline',
        'if [[ -n "${WANDB_API_KEY:-}" ]] || { [[ -r "$netrc_file" ]] && grep -q "machine api.wandb.ai" "$netrc_file"; }; then',
        '    wb_mode=online',
        'fi',
        'if [[ "${WANDB_MODE:-}" == offline ]]; then wb_mode=offline; fi',
        'wb_sync() { "$PYTHON" -m scripts.log_finetune_wandb --root "$FT_ROOT" ' + sync_args + ' "$@"; }',
        'if [[ "$wb_mode" == online ]]; then',
        '    wb_sync --mode online --query-existing || { echo "WARNING: W&B upload failed; preparing offline histories" >&2; wb_mode=offline; }',
        'fi',
        'if [[ "$wb_mode" == offline ]]; then',
        '    echo "WARNING: fine-tuning W&B histories are offline; sync and verify them later" >&2',
        '    wb_sync --mode offline || { echo "ERROR: W&B logging failed; results remain in $FT_ROOT" >&2; failed=1; }',
        'fi',
    ]
    lines += PERSIST_LINES
    # Pool retains lightweight metrics, metadata and evaluation arrays/sets.
    # Actual model/optimizer state stays on scratch under the agreed retention
    # policy; never accidentally duplicate multi-TB pilots into the 1-TB pool.
    # -H preserves shared-prefix evaluation-array hard links.
    lines += ["rsync -aH --prune-empty-dirs --include='*/' --include='*.json' --include='*.jsonl' --include='*.log' "
              "--include='*.parquet' --include='*.npz' --exclude='model_*.pt' --exclude='optim_*.pt' "
              "--include='eval_set_*.pt' --exclude='*' \"$FT_ROOT\" \"$POOL/results/\" "
              '|| echo "WARNING: could not copy $FT_ROOT results to $POOL/results" >&2']
    lines.append('exit "$failed"')
    return "\n".join(lines) + "\n"
