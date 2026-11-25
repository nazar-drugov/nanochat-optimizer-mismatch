"""
E = 8 learning-rate sweep and base models (see nanochat/pretrain_sweep.py and
PRETRAINING_PROTOCOL.md). Run on the ORCD login node from the repository:

  1. Write the two sweep jobs (3 runs each, one B200 each) and submit them:
       python -m scripts.pretrain_sweep jobs --stage sweep --out jobs/
       sbatch jobs/e8-sweep-adamw.sbatch; sbatch jobs/e8-sweep-muon.sbatch
  2. When they finish, choose the LRs (writes jobs/lr_selection.json):
       python -m scripts.pretrain_sweep select --out jobs/
     If the best point of an optimizer is at the edge of its grid, this prints the
     extra point to run (pass it as --extra-lr to `jobs --stage sweep` and `select`).
  3. Write the 3B-token base-model jobs (one per run) and submit them:
       python -m scripts.pretrain_sweep jobs --stage main --selection jobs/lr_selection.json --out jobs/
     Add --seeds 43 44 later for the extra seeds.
  4. Transfer check: 3B-token runs at x0.5 and x2 the transferred LR (the main run is x1),
     then compare (writes jobs/transfer_check.json):
       python -m scripts.pretrain_sweep jobs --stage transfer --selection jobs/lr_selection.json --out jobs/
       python -m scripts.pretrain_sweep transfer-check --selection jobs/lr_selection.json --out jobs/
"""

import argparse
import json
import os
import sys
from pathlib import Path

from nanochat.pretrain_sweep import (
    OPTIMIZERS, SWEEP_LRS, RunSpec, SlurmSettings, check_transfer, transfer_check_runs, main_runs, render_job, select_lr, sweep_points, sweep_runs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_TIME_LIMIT = "03:00:00"   # 3 runs of ~0.3 h (pessimistic ~0.5 h)
MAIN_TIME_LIMIT = "04:30:00"    # one 3B run, ~1.5 h (pessimistic ~2.5 h)


def default_scratch_base():
    return os.environ.get("NANOCHAT_SCRATCH_BASE",
                          str(Path.home() / "orcd" / "scratch" / "public" / "nanochat-optimizer-mismatch"))


def parse_extra_lrs(values):
    """--extra-lr muon=1.6e-2 adamw=5e-4 -> {"muon": [0.016], "adamw": [0.0005]}"""
    extra = {optimizer: [] for optimizer in OPTIMIZERS}
    for value in values or []:
        optimizer, _, lr = value.partition("=")
        if optimizer not in OPTIMIZERS or not lr:
            raise SystemExit(f"--extra-lr expects optimizer=lr with optimizer in {OPTIMIZERS}, got {value!r}")
        extra[optimizer].append(float(lr))
    return {optimizer: tuple(sorted(set(SWEEP_LRS[optimizer]) | set(extra[optimizer]))) for optimizer in OPTIMIZERS}


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")


def cmd_jobs(args):
    out = Path(args.out)
    slurm = SlurmSettings(exclude=args.exclude)
    if args.stage == "sweep":
        runs = sweep_runs(seed=args.seed, lrs=parse_extra_lrs(args.extra_lr))
        for optimizer in OPTIMIZERS:
            job = f"e8-sweep-{optimizer}"
            write(out / f"{job}.sbatch", render_job(job, [r for r in runs if r.optimizer == optimizer],
                                                    SWEEP_TIME_LIMIT, args.repo_dir, args.scratch_base, slurm))
    else:
        selection = load_selection(args.selection, args.stage)
        full_lrs = {o: selection[o]["full_lr"] for o in OPTIMIZERS}
        if args.stage == "main":
            runs = main_runs(full_lrs, seeds=args.seeds)
        else:
            runs = transfer_check_runs(full_lrs, seed=args.seed)
        for run in runs:
            write(out / f"{run.name}.sbatch", render_job(run.name, [run], MAIN_TIME_LIMIT, args.repo_dir, args.scratch_base, slurm))


def load_selection(path, stage):
    if not path:
        raise SystemExit(f"--stage {stage} needs --selection lr_selection.json")
    selection = json.loads(Path(path).read_text())
    not_selected = [o for o in OPTIMIZERS if selection.get(o, {}).get("status") != "selected"]
    if not_selected:
        raise SystemExit(f"no selected LR for {', '.join(not_selected)}; finish the sweep first")
    return selection


def cmd_transfer_check(args):
    selection = load_selection(args.selection, "transfer")
    full_lrs = {o: selection[o]["full_lr"] for o in OPTIMIZERS}
    runs = transfer_check_runs(full_lrs, seed=args.seed) + [RunSpec("main", o, full_lrs[o], args.seed) for o in OPTIMIZERS]
    root = Path(args.checkpoints_root or Path(default_scratch_base()) / "base_checkpoints")
    points = sweep_points(runs, root)
    report = {o: check_transfer(points[o], full_lrs[o], selection[o]["sweep_lr"]) for o in OPTIMIZERS}
    for optimizer, r in report.items():
        table = ", ".join(f"{p['lr']:.3g}: {'diverged' if p['final_val_bpb'] is None else format(p['final_val_bpb'], '.4f')}"
                          for p in r["points"])
        print(f"{optimizer}: {table}")
        if r["status"] == "interior":
            print(f"  -> 3B optimum {r['vertex_lr']:.4g}, {r['vertex_octaves_from_transferred']:+.2f} octaves from the "
                  f"transferred {r['transferred_lr']:.4g}; implied exponent {r['implied_exponent']:+.2f} (assumed {r['assumed_exponent']})")
        else:
            print(f"  -> best point {r['best_lr']:.4g} is at the edge: the 3B optimum lies outside x0.5..x2")
    write(Path(args.out) / "transfer_check.json", json.dumps(report, indent=2))
    return 0


def cmd_select(args):
    runs = sweep_runs(seed=args.seed, lrs=parse_extra_lrs(args.extra_lr))
    root = Path(args.checkpoints_root or Path(default_scratch_base()) / "base_checkpoints")
    points = sweep_points(runs, root)
    selection = {optimizer: select_lr(points[optimizer]) for optimizer in OPTIMIZERS}
    for optimizer, choice in selection.items():
        table = ", ".join(f"{p['lr']:.3g}: {'diverged' if p['final_val_bpb'] is None else format(p['final_val_bpb'], '.4f')}"
                          for p in choice["points"])
        print(f"{optimizer}: {table}")
        if choice["status"] == "selected":
            print(f"  -> {choice['sweep_lr']:.4g} at 0.5B tokens, {choice['full_lr']:.4g} at 3B ({choice['method']})")
        else:
            print(f"  -> best point is at the edge of the grid; run --extra-lr {optimizer}={choice['next_lr']:.3g}")
    write(Path(args.out) / "lr_selection.json", json.dumps(selection, indent=2))
    return 0 if all(c["status"] == "selected" for c in selection.values()) else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    jobs = sub.add_parser("jobs", help="write Slurm job scripts")
    jobs.add_argument("--stage", choices=("sweep", "main", "transfer"), required=True)
    jobs.add_argument("--selection", help="lr_selection.json from `select` (stages main and transfer)")
    jobs.add_argument("--seeds", type=int, nargs="+", default=[42], help="seeds for stage main")
    select = sub.add_parser("select", help="choose each optimizer's LR from the sweep results")
    select.add_argument("--checkpoints-root", help="default: $NANOCHAT_SCRATCH_BASE/base_checkpoints")
    transfer = sub.add_parser("transfer-check", help="compare the transferred LR with the 3B transfer-check runs")
    transfer.add_argument("--selection", required=True, help="lr_selection.json from `select`")
    transfer.add_argument("--checkpoints-root", help="default: $NANOCHAT_SCRATCH_BASE/base_checkpoints")
    transfer.add_argument("--out", default="jobs", help="directory for transfer_check.json")
    transfer.add_argument("--seed", type=int, default=42, help="seed of the transfer-check and main runs")
    for p in (jobs, select):
        p.add_argument("--out", default="jobs", help="directory for job scripts and lr_selection.json")
        p.add_argument("--seed", type=int, default=42, help="seed of the sweep runs")
        p.add_argument("--extra-lr", nargs="*", default=[], help="extra sweep points, e.g. muon=1.6e-2")
    jobs.add_argument("--repo-dir", default=str(REPO_ROOT), help="repository path on the cluster")
    jobs.add_argument("--scratch-base", default=None, help="export NANOCHAT_SCRATCH_BASE in the jobs")
    jobs.add_argument("--exclude", default=SlurmSettings.exclude, help="Slurm nodes to avoid ('' for none)")
    args = parser.parse_args(argv)
    return {"jobs": cmd_jobs, "select": cmd_select, "transfer-check": cmd_transfer_check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
