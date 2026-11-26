"""
Write the fine-tuning Slurm jobs: one per (base model, dataset, fine-tuning optimizer),
each running a 5-point LR grid x 5 seeds through scripts/finetune.py on one B200.

  python -m scripts.finetune_jobs --selection results/e8/lr_selection_3b.json \
      --base-runs e8-main-adamw-s42 e8-main-muon-s42 \
      --dataset wikitext103=<scratch>/finetune_data/wikitext103 --dataset starcoderdata=<scratch>/finetune_data/starcoderdata \
      --d0-file <scratch>/heldout/fineweb-edu-100b/d0.parquet --out jobs/

Length pilot (decay branches; see scripts/finetune.py): add e.g.
  --branch-tokens 3932160 62914560 --seeds 1 2 3 --out-root <scratch>/finetune_pilot --job-prefix ftpilot

Main grid at the LRs selected on seed 42: give each (pretraining, fine-tuning) optimizer pair its own
fractions, e.g. --pair-lr-fractions adamw:adamw=1,0.5 muon:adamw=0.25,0.5 (pairs not listed use
--lr-fractions).
"""

import argparse
import json
import sys
from pathlib import Path

from nanochat.pretrain_sweep import (
    FINETUNE_LR_FRACTIONS, FINETUNE_SEEDS, OPTIMIZERS, SlurmSettings, base_run_seed, finetune_lr_grid, render_finetune_job,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def base_run_optimizer(base_run):
    base_run_seed(base_run)  # validates the name
    return base_run.split("-")[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", required=True, help="3B pretraining LR selection, results/e8/lr_selection_3b.json")
    parser.add_argument("--base-runs", nargs="+", required=True)
    parser.add_argument("--dataset", action="append", required=True, help="name=directory (repeatable)")
    parser.add_argument("--d0-file", required=True)
    parser.add_argument("--lr-fractions", type=float, nargs="+", default=list(FINETUNE_LR_FRACTIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FINETUNE_SEEDS))
    parser.add_argument("--time-limit", default="05:00:00")
    parser.add_argument("--out", default="jobs")
    parser.add_argument("--repo-dir", default=str(REPO_ROOT))
    parser.add_argument("--exclude", default=SlurmSettings.exclude, help="Slurm nodes to avoid ('' for none)")
    parser.add_argument("--branch-tokens", type=int, nargs="*", default=[],
                        help="further fine-tuning lengths run as decay branches (scripts/finetune.py)")
    parser.add_argument("--out-root", default=None, help="finetune --out-root (default: $SCRATCH/finetune)")
    parser.add_argument("--job-prefix", default="ft")
    parser.add_argument("--pair-lr-fractions", nargs="+", default=[], metavar="PRE:FT=F1,F2",
                        help="LR fractions for one (pretraining, fine-tuning) optimizer pair")
    args = parser.parse_args(argv)
    selection = json.loads(Path(args.selection).read_text())

    def checked(values, ft, what):
        """Positive finite fractions whose LRs differ in their first 3 significant digits (cell names)."""
        if not values or not all(0 < v < float("inf") for v in values) \
                or len({f"{v * selection[ft]['full_lr']:.3g}" for v in values}) != len(values):
            parser.error(f"non-positive, non-finite or duplicate LR fractions: {what}")
        return values

    for ft in OPTIMIZERS:
        checked(args.lr_fractions, ft, f"--lr-fractions {args.lr_fractions}")
    grid = finetune_lr_grid(selection, args.lr_fractions)
    pair_grid = {}
    for spec in args.pair_lr_fractions:
        pair, _, fractions = spec.partition("=")
        pre, _, ft = pair.partition(":")
        if pre not in OPTIMIZERS or ft not in OPTIMIZERS or not fractions:
            parser.error(f"bad --pair-lr-fractions entry: {spec!r}")
        if (pre, ft) in pair_grid:
            parser.error(f"pair given twice in --pair-lr-fractions: {spec!r}")
        try:
            values = [float(f) for f in fractions.split(",")]
        except ValueError:
            parser.error(f"bad --pair-lr-fractions entry: {spec!r}")
        pair_grid[pre, ft] = finetune_lr_grid(selection, checked(values, ft, repr(spec)))[ft]
    if pair_grid and len(pair_grid) < len(OPTIMIZERS) ** 2:
        missing = sorted({(p, f) for p in OPTIMIZERS for f in OPTIMIZERS} - set(pair_grid))
        print(f"WARNING: pairs {missing} not in --pair-lr-fractions use --lr-fractions {args.lr_fractions}",
              file=sys.stderr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = (["--branch-tokens", *map(str, args.branch_tokens)] if args.branch_tokens else []) \
        + (["--out-root", args.out_root] if args.out_root else [])
    for base_run in args.base_runs:
        for spec in args.dataset:
            name, _, directory = spec.partition("=")
            for optimizer in OPTIMIZERS:
                lrs = pair_grid.get((base_run_optimizer(base_run), optimizer), grid[optimizer])
                job = f"{args.job_prefix}-{base_run}-{name}-{optimizer}"
                (out / f"{job}.sbatch").write_text(render_finetune_job(
                    job, base_run, name, directory, args.d0_file, optimizer, lrs, args.time_limit,
                    args.repo_dir, seeds=args.seeds, slurm=SlurmSettings(exclude=args.exclude), extra_args=extra))
                print(f"wrote {out / job}.sbatch  ({optimizer}: " + ", ".join(f"{lr:.3g}" for lr in lrs) + ")")


if __name__ == "__main__":
    main()
