"""
Download and prepare a fine-tuning dataset (EXPERIMENT_SPEC.md §6) in nanochat's parquet
layout: train_*.parquet + val_00000.parquet (held-out D1) + manifest.json.

  python -m scripts.make_finetune_data --dataset wikitext103 --out <scratch>/finetune_data/wikitext103
  python -m scripts.make_finetune_data --dataset starcoderdata --out <scratch>/finetune_data/starcoderdata

Needs network access (run on a login node). StarCoderData is gated: it reads the Hugging Face
token from $HF_TOKEN or ~/.cache/huggingface/token. Raw downloads go to --raw-dir (default <out>/raw).
"""

import argparse
import json
from pathlib import Path

from nanochat.finetune_data import prepare_starcoderdata, prepare_wikitext103


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("wikitext103", "starcoderdata"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--starcoder-language", default="python", help="StarCoderData language directory")
    parser.add_argument("--starcoder-train-files", type=int, default=1,
                        help="StarCoderData parquet files used for training (python: ~0.4 GB each)")
    parser.add_argument("--starcoder-val-docs", type=int, default=20_000,
                        help="documents sampled from the language's last file as held-out D1")
    args = parser.parse_args()
    raw = args.raw_dir or str(Path(args.out) / "raw")
    if args.dataset == "wikitext103":
        info = prepare_wikitext103(args.out, raw)
    else:
        info = prepare_starcoderdata(args.out, raw, language=args.starcoder_language,
                                     train_files=args.starcoder_train_files, val_docs=args.starcoder_val_docs)
    print(json.dumps({k: v for k, v in info.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
