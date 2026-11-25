"""
Write the fine-tuning Slurm jobs: one per (base model, dataset, fine-tuning optimizer),
each running a 5-point LR grid x 5 seeds through scripts/finetune.py on one B200.

  python -m scripts.finetune_jobs --selection jobs/lr_selection.json \
      --base-runs e8-main-adamw-s42 e8-main-muon-s42 \
      --dataset wikitext103=<scratch>/finetune_data/wikitext103 --dataset starcoderdata=<scratch>/finetune_data/starcoderdata \
      --d0-file <scratch>/heldout/fineweb-edu-100b/d0.parquet --out jobs/

Length pilot (decay branches; see scripts/finetune.py): add e.g.
  --branch-tokens 3932160 62914560 --seeds 1 2 3 --out-root <scratch>/finetune_pilot --job-prefix ftpilot
"""

import argparse
import json
from pathlib import Path

from nanochat.pretrain_sweep import (
    FINETUNE_LR_FRACTIONS, FINETUNE_SEEDS, OPTIMIZERS, SlurmSettings, finetune_lr_grid, render_finetune_job,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", required=True, help="pretraining lr_selection.json (scripts/pretrain_sweep.py)")
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
    args = parser.parse_args(argv)
    grid = finetune_lr_grid(json.loads(Path(args.selection).read_text()), args.lr_fractions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = (["--branch-tokens", *map(str, args.branch_tokens)] if args.branch_tokens else []) \
        + (["--out-root", args.out_root] if args.out_root else [])
    for base_run in args.base_runs:
        for spec in args.dataset:
            name, _, directory = spec.partition("=")
            for optimizer in OPTIMIZERS:
                job = f"{args.job_prefix}-{base_run}-{name}-{optimizer}"
                (out / f"{job}.sbatch").write_text(render_finetune_job(
                    job, base_run, name, directory, args.d0_file, optimizer, grid[optimizer], args.time_limit,
                    args.repo_dir, seeds=args.seeds, slurm=SlurmSettings(exclude=args.exclude), extra_args=extra))
                print(f"wrote {out / job}.sbatch  ({optimizer}: " + ", ".join(f"{lr:.3g}" for lr in grid[optimizer]) + ")")


if __name__ == "__main__":
    main()
