"""Tests for the Chinchilla-style token-budget planning script."""

import torch

from nanochat.gpt import GPT, make_gpt_config
from scripts.count_tokens_chinchilla import calculate_token_budget


def make_meta_model(n_experts=4):
    config = make_gpt_config(
        depth=2,
        aspect_ratio=8,
        head_dim=8,
        max_seq_len=16,
        vocab_size=32,
        ffn_type="moe",
        n_experts=n_experts,
    )
    with torch.device("meta"):
        return GPT(config, pad_vocab_size_to=1)


def test_planner_uses_total_scaling_parameter_count():
    model = make_meta_model()

    counts = model.num_scaling_params()

    assert counts["total"] == sum(parameter.numel() for parameter in model.parameters())


def test_more_experts_increase_total_parameter_count():
    four_experts = make_meta_model(n_experts=4)
    eight_experts = make_meta_model(n_experts=8)

    assert (
        eight_experts.num_scaling_params()["total"]
        > four_experts.num_scaling_params()["total"]
    )


def test_token_budget_rounds_up_to_a_whole_training_step():
    budget = calculate_token_budget(
        num_params=101,
        multiplier=20,
        total_batch_size=128,
    )

    assert budget["target_tokens"] == 2020
    assert budget["rounded_tokens"] >= budget["target_tokens"]
    assert budget["rounded_tokens"] % 128 == 0
    assert budget["num_iterations"] == budget["rounded_tokens"] // 128
    assert budget["exact_ratio"] == budget["rounded_tokens"] / 101


def test_shared_config_builder_matches_training_dimension_rule():
    config = make_gpt_config(
        depth=7,
        aspect_ratio=64,
        head_dim=128,
        max_seq_len=1024,
        vocab_size=32768,
        ffn_type="moe",
        n_experts=4,
    )

    assert config.n_embd == 512
    assert config.n_head == 4
    assert config.n_kv_head == 4
    assert config.n_layer == 7
    assert config.sequence_len == 1024
    assert config.vocab_size == 32768
    assert config.ffn_type == "moe"
    assert config.n_experts == 4
