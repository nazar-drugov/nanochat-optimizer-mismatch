"""Calculate parameter-based token budgets for a base-model architecture.

This is a planning tool only. It does not start training or choose a training
budget automatically. Copy the selected rounded token count into
``scripts.base_train --train-tokens`` so every matched run uses the same budget.

Example:

python -m scripts.count_tokens_chinchilla \
    --ffn-type=moe --n-experts=4 --depth=20 --total-batch-size=524288

If the nanoChat tokenizer has not been trained yet, pass its planned vocabulary
size explicitly with ``--vocab-size``.
"""

import argparse
import math

import torch

from nanochat.gpt import GPT, make_gpt_config
from nanochat.tokenizer import get_tokenizer


def calculate_token_budget(num_params, multiplier, total_batch_size):
    """Round a parameter-based token target up to a whole training step."""
    if num_params <= 0:
        raise ValueError("num_params must be greater than zero")
    if multiplier <= 0:
        raise ValueError("multiplier must be greater than zero")
    if total_batch_size <= 0:
        raise ValueError("total_batch_size must be greater than zero")

    target_tokens = multiplier * num_params
    num_iterations = math.ceil(target_tokens / total_batch_size)
    rounded_tokens = num_iterations * total_batch_size
    exact_ratio = rounded_tokens / num_params
    return {
        "multiplier": multiplier,
        "target_tokens": target_tokens,
        "num_iterations": num_iterations,
        "rounded_tokens": rounded_tokens,
        "exact_ratio": exact_ratio,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count model parameters and print Chinchilla-style token budgets"
    )
    parser.add_argument("--ffn-type", choices=("dense", "moe"), default="moe")
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="model vocabulary size (default: load it from the trained tokenizer)",
    )
    parser.add_argument(
        "--total-batch-size",
        type=int,
        required=True,
        help="total batch size in tokens across all ranks",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_experts <= 0:
        raise SystemExit("--n-experts must be greater than zero")
    if args.total_batch_size <= 0:
        raise SystemExit("--total-batch-size must be greater than zero")
    if args.vocab_size is not None and args.vocab_size <= 0:
        raise SystemExit("--vocab-size must be greater than zero")

    vocab_size = (
        args.vocab_size
        if args.vocab_size is not None
        else get_tokenizer().get_vocab_size()
    )
    config = make_gpt_config(
        depth=args.depth,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        vocab_size=vocab_size,
        ffn_type=args.ffn_type,
        n_experts=args.n_experts,
    )
    with torch.device("meta"):
        model = GPT(config)
    num_params = model.num_scaling_params()["total"]

    print(f"Model: {args.ffn_type}, {args.n_experts} experts, depth {args.depth}")
    print(f"Total parameters: {num_params:,}")
    print(f"Total batch size: {args.total_batch_size:,} tokens")
    for multiplier in (20, 40):
        budget = calculate_token_budget(
            num_params=num_params,
            multiplier=multiplier,
            total_batch_size=args.total_batch_size,
        )
        print()
        print(f"{multiplier}x token target: {budget['target_tokens']:,}")
        print(f"Rounded training tokens: {budget['rounded_tokens']:,}")
        print(f"Required iterations: {budget['num_iterations']:,}")
        print(f"Exact token-to-parameter ratio: {budget['exact_ratio']:.6f}")


if __name__ == "__main__":
    main()
