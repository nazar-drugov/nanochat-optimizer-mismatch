"""
Per-E learning-rate sweeps and base models (see nanochat/pretrain_sweep.py and
PRETRAINING_PROTOCOL.md). The legacy E=8 workflow remains the default.

For each E=1,4,16 use --experts E on every command. Separate output directories
prevent cross-E selection collisions (default jobs/e<E> for new E). Run in order:
  jobs --stage sweep
  select [--extra-lr OPTIMIZER=LR ...]
  jobs --stage transfer --selection <out>/lr_selection.json
  transfer-check --selection <out>/lr_selection.json [--extra-factor OPTIMIZER=FACTOR ...]
  jobs --stage main --selection <out>/lr_selection_3b.json
  base-manifest --selection <out>/lr_selection_3b.json

New-E transfer jobs are separately named x0.5, x1, x2 trials. The x1 trial saves
the regular full checkpoint cadence; a retained center requires no redundant
main job. The base manifest maps the logical main name to its actual source
checkpoint directory after verifying completed training and hashing real model
weights. If the checked optimum moves, generate a distinct main job. To keep
grid selections and results aligned, use the same --extra-lr/--extra-factor
extensions for jobs and selection, and submit only the intended stage scripts.

Legacy E=8 commands on the ORCD login node from the repository:

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
     If an optimizer's best point is at the edge, add --extra-factor (e.g. adamw=4 adamw=8)
     to both commands. transfer-check also writes jobs/lr_selection_3b.json: the transferred
     LR, or the 3B optimum where it is more than 0.25 octave away.
  5. If the 3B optimum replaced an LR: move each superseded base model to its x1
     transfer-check name (e8-xfer-<optimizer>-lr<transferred>-s42, so transfer-check still
     finds it), then write and submit the base models from the new selection:
       python -m scripts.pretrain_sweep jobs --stage main --selection jobs/lr_selection_3b.json --seeds 42 43 44 --out jobs/
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from nanochat.pretrain_sweep import (
    EXPERT_COUNTS, HEADLINE_EXPERTS, OPTIMIZERS, TRANSFER_CHECK_FACTORS, RunSpec, SlurmSettings,
    check_transfer, completed_base_manifest, selection_after_transfer, transfer_check_runs,
    main_runs, render_job, select_lr, sweep_lr_grid, sweep_points, sweep_runs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_TIME_LIMIT = "03:00:00"   # 3 runs of ~0.3 h (pessimistic ~0.5 h)
MAIN_TIME_LIMIT = "04:30:00"    # one 3B run, ~1.5 h (pessimistic ~2.5 h)


def default_scratch_base():
    return os.environ.get("NANOCHAT_SCRATCH_BASE",
                          str(Path.home() / "orcd" / "scratch" / "public" / "nanochat-optimizer-mismatch"))


def parse_extra_lrs(values, experts=HEADLINE_EXPERTS):
    """--extra-lr muon=1.6e-2 adamw=5e-4 -> {"muon": [0.016], "adamw": [0.0005]}"""
    extra = {optimizer: [] for optimizer in OPTIMIZERS}
    for value in values or []:
        optimizer, _, lr = value.partition("=")
        if optimizer not in OPTIMIZERS or not lr:
            raise SystemExit(f"--extra-lr expects optimizer=lr with optimizer in {OPTIMIZERS}, got {value!r}")
        try:
            parsed = float(lr)
        except ValueError:
            raise SystemExit(f"invalid --extra-lr {value!r}") from None
        if not math.isfinite(parsed) or parsed <= 0:
            raise SystemExit(f"invalid --extra-lr {value!r}")
        extra[optimizer].append(parsed)
    grid = sweep_lr_grid(experts)
    return {optimizer: tuple(sorted(set(grid[optimizer]) | set(extra[optimizer]))) for optimizer in OPTIMIZERS}


def parse_extra_factors(values):
    """--extra-factor adamw=4 adamw=8 -> {"adamw": (0.5, 2.0, 4.0, 8.0), "muon": (0.5, 2.0)}"""
    extra = {optimizer: [] for optimizer in OPTIMIZERS}
    for value in values or []:
        optimizer, _, factor = value.partition("=")
        if optimizer not in OPTIMIZERS or not factor:
            raise SystemExit(f"--extra-factor expects optimizer=factor with optimizer in {OPTIMIZERS}, got {value!r}")
        try:
            parsed = float(factor)
        except ValueError:
            raise SystemExit(f"invalid --extra-factor {value!r}") from None
        if not math.isfinite(parsed) or parsed <= 0:
            raise SystemExit(f"invalid --extra-factor {value!r}")
        extra[optimizer].append(parsed)
    return {o: tuple(sorted(set(TRANSFER_CHECK_FACTORS) | set(extra[o]))) for o in OPTIMIZERS}


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")


def cmd_jobs(args):
    out = Path(args.out)
    slurm = SlurmSettings(exclude=args.exclude)
    optimizers = OPTIMIZERS if args.optimizer is None else (args.optimizer,)
    if args.stage == "sweep":
        runs = sweep_runs(seed=args.seed, lrs=parse_extra_lrs(args.extra_lr, args.experts), n_experts=args.experts)
        for optimizer in optimizers:
            job = f"e{args.experts}-sweep-{optimizer}"
            write(out / f"{job}.sbatch", render_job(job, [r for r in runs if r.optimizer == optimizer],
                                                    SWEEP_TIME_LIMIT if args.experts == HEADLINE_EXPERTS else MAIN_TIME_LIMIT,
                                                    args.repo_dir, args.scratch_base, slurm))
    else:
        selection = load_selection(args.selection, args.stage, args.experts)
        full_lrs = {o: selection[o]["full_lr"] for o in OPTIMIZERS}
        if args.stage == "main":
            runs = main_runs(full_lrs, seeds=args.seeds, n_experts=args.experts)
            if args.experts != HEADLINE_EXPERTS:
                # A separately named center run is a completed base if retained
                # at its exact checked LR and initialization.
                skipped = [r for r in runs if selection[r.optimizer]["full_lr_source"] == "transferred"
                           and selection[r.optimizer]["seed"] == r.seed
                           and selection[r.optimizer]["transferred_lr"] == r.lr]
                runs = [r for r in runs if r not in skipped]
                for r in skipped:
                    print(f"{r.name}: reuse selected x1 transfer checkpoint; no redundant main job")
        else:
            if args.experts != HEADLINE_EXPERTS:
                check_selection_seed(selection, args.seed)
            runs = transfer_check_runs(full_lrs, seed=args.seed, factors=parse_extra_factors(args.extra_factor),
                                       n_experts=args.experts)
        for run in runs:
            if run.optimizer not in optimizers:
                continue
            write(out / f"{run.name}.sbatch", render_job(run.name, [run], MAIN_TIME_LIMIT, args.repo_dir, args.scratch_base, slurm))


def load_selection(path, stage, experts=HEADLINE_EXPERTS):
    if not path:
        raise SystemExit(f"--stage {stage} needs --selection lr_selection.json")
    selection = json.loads(Path(path).read_text())
    not_selected = [o for o in OPTIMIZERS if selection.get(o, {}).get("status") != "selected"]
    if not_selected:
        raise SystemExit(f"no selected LR for {', '.join(not_selected)}; finish the sweep first")
    for o, choice in selection.items():
        if o not in OPTIMIZERS:
            raise SystemExit(f"unknown optimizer in selection: {o}")
        if choice.get("n_experts", HEADLINE_EXPERTS) != experts:
            raise SystemExit(f"selection expert count mismatch for {o}: expected E={experts}")
        if not isinstance(choice.get("full_lr"), (float, int)) or not math.isfinite(choice["full_lr"]) or choice["full_lr"] <= 0:
            raise SystemExit(f"invalid selected full LR for {o}")
        if experts != HEADLINE_EXPERTS and stage in ("main", "base-manifest") and (
            choice.get("full_lr_source") not in ("transferred", "transfer-check vertex") or
            "transferred_lr" not in choice or "seed" not in choice
        ):
            raise SystemExit("new-E main bases must use final lr_selection_3b.json after transfer checks")
        if experts != HEADLINE_EXPERTS and stage == "transfer" and choice.get("full_lr_source") is not None:
            raise SystemExit("transfer checks need the original 0.5B lr_selection.json, not a final 3B selection")
    return selection


def check_selection_seed(selection, seed):
    if any(choice.get("seed") != seed for choice in selection.values()):
        raise SystemExit(f"selection pretraining seed differs from transfer-check seed {seed}")


def cmd_transfer_check(args):
    selection = load_selection(args.selection, "transfer", args.experts)
    full_lrs = {o: selection[o]["full_lr"] for o in OPTIMIZERS}
    root = Path(args.checkpoints_root or Path(default_scratch_base()) / "base_checkpoints")
    # The x1 point is the base model, unless that was rerun at the 3B optimum: then the
    # original base model was moved to the x1 transfer-check name.
    runs = transfer_check_runs(full_lrs, seed=args.seed, factors=parse_extra_factors(args.extra_factor),
                               n_experts=args.experts)
    if args.experts == HEADLINE_EXPERTS:
        x1 = [RunSpec("xfer", o, full_lrs[o], args.seed) for o in OPTIMIZERS]
        x1 = [r if (root / r.name).is_dir() else RunSpec("main", r.optimizer, r.lr, r.seed) for r in x1]
        runs += x1
    else:
        check_selection_seed(selection, args.seed)
    points = sweep_points(runs, root)
    report = {o: check_transfer(points[o], full_lrs[o], selection[o]["sweep_lr"]) for o in OPTIMIZERS}
    if args.experts != HEADLINE_EXPERTS:
        for item in report.values():
            item.update(n_experts=args.experts, seed=args.seed)
    for optimizer, r in report.items():
        table = ", ".join(f"{p['lr']:.3g}: {'diverged' if p['final_val_bpb'] is None else format(p['final_val_bpb'], '.4f')}"
                          for p in r["points"])
        print(f"{optimizer}: {table}")
        if r["status"] == "interior":
            print(f"  -> 3B optimum {r['vertex_lr']:.4g}, {r['vertex_octaves_from_transferred']:+.2f} octaves from the "
                  f"transferred {r['transferred_lr']:.4g}; implied exponent {r['implied_exponent']:+.2f} (assumed {r['assumed_exponent']})")
        elif r["status"] == "edge":
            print(f"  -> best point {r['best_lr']:.4g} is at the edge: the 3B optimum lies beyond it; "
                  f"extend with --extra-factor {optimizer}={r['next_factor']:.12g}")
        else:
            print(f"  -> {r['reason']}; cannot fit a finite local vertex. Review before further transfer checks.")
    write(Path(args.out) / "transfer_check.json", json.dumps(report, indent=2))
    final = selection_after_transfer(selection, report)
    for optimizer, c in final.items():
        if c["full_lr_source"] == "transfer-check vertex":
            print(f"{optimizer}: base models at the 3B optimum {c['full_lr']:.4g} (rerun those at {c['transferred_lr']:.4g})")
    write(Path(args.out) / "lr_selection_3b.json", json.dumps(final, indent=2))
    return 0 if args.experts == HEADLINE_EXPERTS or all(c["status"] == "selected" for c in final.values()) else 3


def cmd_select(args):
    runs = sweep_runs(seed=args.seed, lrs=parse_extra_lrs(args.extra_lr, args.experts), n_experts=args.experts)
    root = Path(args.checkpoints_root or Path(default_scratch_base()) / "base_checkpoints")
    points = sweep_points(runs, root)
    selection = {optimizer: select_lr(points[optimizer]) for optimizer in OPTIMIZERS}
    if args.experts != HEADLINE_EXPERTS:
        for item in selection.values():
            item.update(n_experts=args.experts, seed=args.seed)
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


def cmd_base_manifest(args):
    selection = load_selection(args.selection, "base-manifest", args.experts)
    root = Path(args.checkpoints_root or Path(default_scratch_base()) / "base_checkpoints")
    document = completed_base_manifest(selection, root, args.experts, args.seeds)
    write(Path(args.out) / "base_manifest.json", json.dumps(document, indent=2) + "\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    jobs = sub.add_parser("jobs", help="write Slurm job scripts")
    jobs.add_argument("--stage", choices=("sweep", "main", "transfer"), required=True)
    jobs.add_argument("--optimizer", choices=OPTIMIZERS, help="emit jobs only for this optimizer")
    jobs.add_argument("--selection", help="lr_selection.json from `select` (stages main and transfer)")
    jobs.add_argument("--seeds", type=int, nargs="+", default=[42], help="seeds for stage main")
    select = sub.add_parser("select", help="choose each optimizer's LR from the sweep results")
    select.add_argument("--checkpoints-root", help="default: $NANOCHAT_SCRATCH_BASE/base_checkpoints")
    transfer = sub.add_parser("transfer-check", help="compare the transferred LR with the 3B transfer-check runs")
    transfer.add_argument("--selection", required=True, help="lr_selection.json from `select`")
    transfer.add_argument("--checkpoints-root", help="default: $NANOCHAT_SCRATCH_BASE/base_checkpoints")
    transfer.add_argument("--seed", type=int, default=42, help="seed of the transfer-check and main runs")
    manifest = sub.add_parser("base-manifest", help="pin completed new-E bases at final checked LRs")
    manifest.add_argument("--selection", required=True, help="final lr_selection_3b.json")
    manifest.add_argument("--checkpoints-root", help="default: $NANOCHAT_SCRATCH_BASE/base_checkpoints")
    manifest.add_argument("--seeds", type=int, nargs="+", default=[42])
    for p in (jobs, select, transfer, manifest):
        p.add_argument("--experts", type=int, choices=EXPERT_COUNTS, default=HEADLINE_EXPERTS,
                       help="expert count; defaults preserve the historic E=8 protocol")
        p.add_argument("--out", help="output directory (default jobs for E8, jobs/e<E> otherwise)")
    for p in (jobs, transfer):
        p.add_argument("--extra-factor", nargs="*", default=[],
                       help="extra transfer-check points as multiples of the transferred LR, e.g. adamw=4 adamw=8")
    for p in (jobs, select):
        p.add_argument("--seed", type=int, default=42, help="seed of the sweep runs")
        p.add_argument("--extra-lr", nargs="*", default=[], help="extra sweep points, e.g. muon=1.6e-2")
    jobs.add_argument("--repo-dir", default=str(REPO_ROOT), help="repository path on the cluster")
    jobs.add_argument("--scratch-base", default=None, help="export NANOCHAT_SCRATCH_BASE in the jobs")
    jobs.add_argument("--exclude", default=SlurmSettings.exclude, help="Slurm nodes to avoid ('' for none)")
    args = parser.parse_args(argv)
    if not args.out:
        args.out = "jobs" if args.experts == HEADLINE_EXPERTS else f"jobs/e{args.experts}"
    return {"jobs": cmd_jobs, "select": cmd_select, "transfer-check": cmd_transfer_check,
            "base-manifest": cmd_base_manifest}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
