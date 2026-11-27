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
    FINETUNE_LR_FRACTIONS, FINETUNE_SEEDS, OPTIMIZERS, SlurmSettings, base_run_seed, base_run_experts,
    finetune_lr_grid, render_finetune_job, resolve_base_pin,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def base_run_optimizer(base_run):
    base_run_seed(base_run)  # validates the name
    return base_run.split("-")[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", required=True, help="3B pretraining LR selection, results/e8/lr_selection_3b.json")
    parser.add_argument("--base-manifest", type=Path, help="required for non-E8: explicit verified base checkpoint pins")
    parser.add_argument("--base-runs", nargs="+", required=True)
    parser.add_argument("--dataset", action="append", required=True, help="name=directory (repeatable)")
    parser.add_argument("--d0-file", required=True)
    parser.add_argument("--lr-fractions", type=float, nargs="+", default=list(FINETUNE_LR_FRACTIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
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
    parser.add_argument("--pairs", nargs="+", default=None,
                        choices=[f"{p}:{f}" for p in OPTIMIZERS for f in OPTIMIZERS],
                        help="generate only these PRE:FT pairs (for edge extensions)")
    args = parser.parse_args(argv)
    selection = json.loads(Path(args.selection).read_text())
    experts = {base_run_experts(name) for name in args.base_runs}
    if len(experts) != 1 or len(set(args.base_runs)) != len(args.base_runs):
        parser.error("generate jobs for distinct base runs of one E at a time (one per-E LR selection)")
    E = experts.pop()
    pairs = None if args.pairs is None else {tuple(spec.split(":")) for spec in args.pairs}
    if pairs is not None and (len(pairs) != len(args.pairs) or
                              not {p for p, _ in pairs} <= {base_run_optimizer(r) for r in args.base_runs}):
        parser.error("--pairs must be unique and refer to requested pretraining optimizer bases")
    # Resolve every requested pin before writing any scripts. New-E selections must
    # be the exact finalized per-E pretraining LRs, not an E8 or provisional LR file.
    for base_run in args.base_runs:
        resolve_base_pin(base_run, args.base_manifest)
    if E != 8:
        if args.base_manifest is None or args.out_root is None:
            parser.error("non-E8 pilots require --base-manifest and an explicit --out-root")
        if any(base_run_seed(r) != 42 for r in args.base_runs):
            parser.error("the approved non-E8 pilot uses pretraining seed 42")
        for opt in OPTIMIZERS:
            pin = resolve_base_pin(f"e{E}-main-{opt}-s42", args.base_manifest)
            choice = selection.get(opt, {})
            if (choice.get("status") != "selected" or choice.get("n_experts") != E or
                    choice.get("full_lr_source") not in ("transferred", "transfer-check vertex") or
                    choice.get("seed") != 42 or choice.get("full_lr") != pin.matrix_lr):
                parser.error(f"{opt}: selection is not the finalized E={E} seed-42 pinned pretraining LR")
    args.seeds = args.seeds if args.seeds is not None else ([1, 2, 3] if E != 8 else list(FINETUNE_SEEDS))
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s <= 0 for s in args.seeds):
        parser.error("fine-tuning seeds must be distinct positive integers")
    # The 240-step trunk reference fixes the existing eight-step warmup. Only
    # the additional horizons differ by dataset, as in the clean E8 pilots.
    branch_by_dataset = {"starcoderdata": [3_932_160, 62_914_560], "wikitext103": [3_932_160, 47_185_920]}
    for spec in args.dataset:
        name, sep, directory = spec.partition("=")
        if not sep or not name or not directory:
            parser.error(f"bad dataset entry {spec!r}; expected name=directory")
        if E != 8 and (name not in branch_by_dataset or
                       (args.branch_tokens and set(args.branch_tokens) != set(branch_by_dataset[name]))):
            parser.error(f"E={E} pilot lengths must be the approved three for {name}; omit --branch-tokens for defaults")

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
    for base_run in args.base_runs:
        for spec in args.dataset:
            name, _, directory = spec.partition("=")
            tokens = args.branch_tokens if E == 8 else branch_by_dataset[name]
            extra = (["--branch-tokens", *map(str, tokens)] if tokens else []) \
                + (["--out-root", args.out_root] if args.out_root else [])
            if E != 8:
                extra += ["--train-tokens", "15728640", "--total-batch-size", "65536",
                          "--device-batch-size", "32", "--warmup-ratio", ".03", "--warmdown-ratio", ".2",
                          "--weight-decay", "0", "--grad-clip", "1", "--non-matrix-lr-scale", ".1",
                          "--eval-packing", "bestfit", "--eval-tokens", "1048576",
                          "--checkpoint-spacing", "log"]
            for optimizer in OPTIMIZERS:
                if pairs is not None and (base_run_optimizer(base_run), optimizer) not in pairs:
                    continue
                lrs = pair_grid.get((base_run_optimizer(base_run), optimizer), grid[optimizer])
                job = f"{args.job_prefix}-{base_run}-{name}-{optimizer}"
                (out / f"{job}.sbatch").write_text(render_finetune_job(
                    job, base_run, name, directory, args.d0_file, optimizer, lrs, args.time_limit,
                    args.repo_dir, seeds=args.seeds, slurm=SlurmSettings(exclude=args.exclude), extra_args=extra,
                    **({"base_manifest": args.base_manifest.resolve()} if args.base_manifest else {})))
                print(f"wrote {out / job}.sbatch  ({optimizer}: " + ", ".join(f"{lr:.3g}" for lr in lrs) + ")")


if __name__ == "__main__":
    main()
