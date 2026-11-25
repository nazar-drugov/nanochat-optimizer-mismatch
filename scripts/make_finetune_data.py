"""
Download and prepare a fine-tuning dataset (EXPERIMENT_SPEC.md §6) in nanochat's parquet
layout: train_*.parquet + val_00000.parquet (held-out D1) + manifest.json.

  python -m scripts.make_finetune_data --dataset wikitext103 --out <scratch>/finetune_data/wikitext103
  python -m scripts.make_finetune_data --dataset codeparrot --out <scratch>/finetune_data/codeparrot

Needs network access (run on a login node). Raw downloads go to --raw-dir (default <out>/raw).
"""

import argparse
import json
from pathlib import Path

from nanochat.finetune_data import prepare_codeparrot, prepare_wikitext103


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("wikitext103", "codeparrot"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--codeparrot-train-files", type=int, default=2,
                        help="number of codeparrot-clean parquet files to use for training (~0.1-0.2B tokens each)")
    args = parser.parse_args()
    raw = args.raw_dir or str(Path(args.out) / "raw")
    if args.dataset == "wikitext103":
        info = prepare_wikitext103(args.out, raw)
    else:
        info = prepare_codeparrot(args.out, raw, max_train_files=args.codeparrot_train_files)
    print(json.dumps({k: v for k, v in info.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
