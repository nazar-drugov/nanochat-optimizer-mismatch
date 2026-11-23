"""Tests for checkpoint metadata validation."""

from types import SimpleNamespace

import pytest

from nanochat.checkpoint_manager import make_base_experiment_name, validate_checkpoint_seed, validate_experiment_name


def make_experiment_args(**overrides):
    values = {
        "depth": 12,
        "ffn_type": "moe",
        "n_experts": 4,
        "optimizer_name": "adamw",
        "seed": 42,
        "train_tokens": 125_829_120,
        "total_batch_size": 524_288,
        "embedding_lr": 0.3,
        "unembedding_lr": 0.008,
        "router_lr": 0.004,
        "adamw_matrix_lr": 6e-4,
        "muon_matrix_lr": 0.02,
        "weight_decay": 0.28,
        "muon_ns_steps": 5,
        "warmup_steps": 40,
        "warmdown_ratio": 0.65,
        "router_optimizer": "same",
        "muon_variant": "plain",
        "moe_layers": "all",
        "balancer": "aux",
        "grad_clip": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_base_experiment_name_includes_training_identity():
    name = make_base_experiment_name(make_experiment_args())

    assert name == (
        "d12-moe-e4-balaux-adamw-s42-tok125829120-bs524288-"
        "mlr0.0006-elr0.3-ulr0.008-rlr0.004-wd0.28-clip1-wu40-wdr0.65"
    )


@pytest.mark.parametrize("overrides, fragment", [
    ({"optimizer_name": "muon"}, "-muon-plain-"),
    ({"optimizer_name": "muon", "muon_variant": "keller"}, "-muon-keller-"),
    ({"router_optimizer": "adamw", "optimizer_name": "muon"}, "-radamw-"),
    ({"balancer": "bias"}, "-balbias-"),
    ({"moe_layers": "every:2"}, "-mlevery2-"),
    ({"grad_clip": 0.0}, "-noclip-"),
])
def test_base_experiment_name_distinguishes_cell_factors(overrides, fragment):
    assert fragment in make_base_experiment_name(make_experiment_args(**overrides))
    assert make_base_experiment_name(make_experiment_args(**overrides)) != make_base_experiment_name(make_experiment_args())


def test_base_experiment_name_changes_with_training_budget():
    first = make_base_experiment_name(make_experiment_args(train_tokens=1_048_576))
    second = make_base_experiment_name(make_experiment_args(train_tokens=2_097_152))

    assert first != second


def test_checkpoint_seed_validation_accepts_matching_seed():
    validate_checkpoint_seed({"user_config": {"seed": 42}}, seed=42)


def test_checkpoint_seed_validation_rejects_mismatched_seed():
    with pytest.raises(ValueError, match=r"Checkpoint seed 42 does not match requested --seed=7"):
        validate_checkpoint_seed({"user_config": {"seed": 42}}, seed=7)


def test_checkpoint_seed_validation_warns_when_seed_is_missing():
    with pytest.warns(RuntimeWarning, match=r"does not contain user_config.seed"):
        validate_checkpoint_seed({"user_config": {}}, seed=42)


@pytest.mark.parametrize("name", ["experiment-1", "d12.moe_adamw-s42"])
def test_experiment_name_validation_accepts_safe_names(name):
    assert validate_experiment_name(name) == name


@pytest.mark.parametrize("name", ["", "   ", ".", "..", "group/run", r"group\run"])
def test_experiment_name_validation_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        validate_experiment_name(name)
