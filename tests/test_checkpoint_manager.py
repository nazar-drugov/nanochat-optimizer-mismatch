"""Tests for checkpoint metadata validation."""

from types import SimpleNamespace

import pytest

from nanochat.checkpoint_manager import (
    make_base_experiment_name, train_shard_inventory, validate_checkpoint_seed,
    validate_checkpoint_tokenizer, validate_checkpoint_training_data, validate_experiment_name,
)


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


SHARDS = [{"name": "shard_00000.parquet", "size": 100}, {"name": "shard_00001.parquet", "size": 200}]


def test_train_shard_inventory_records_names_and_sizes_in_order(tmp_path):
    paths = []
    for name, size in (("shard_00000.parquet", 3), ("shard_00001.parquet", 5)):
        path = tmp_path / name
        path.write_bytes(b"x" * size)
        paths.append(str(path))
    assert train_shard_inventory(paths) == [{"name": "shard_00000.parquet", "size": 3},
                                            {"name": "shard_00001.parquet", "size": 5}]


def test_training_data_validation_accepts_identical_shards():
    validate_checkpoint_training_data({"train_shards": SHARDS}, [dict(shard) for shard in SHARDS])


def test_training_data_validation_accepts_the_same_data_in_another_directory():
    # Only names, order and sizes matter, so moving the data (pool to scratch) is fine.
    validate_checkpoint_training_data({"train_shards": SHARDS, "data_dir": "/pool/data"}, SHARDS)


def test_training_data_validation_rejects_a_missing_shard():
    with pytest.raises(ValueError, match=r"missing shards \(1\): shard_00001.parquet"):
        validate_checkpoint_training_data({"train_shards": SHARDS}, SHARDS[:1])


def test_training_data_validation_rejects_an_extra_shard():
    extra = SHARDS + [{"name": "shard_00002.parquet", "size": 300}]
    with pytest.raises(ValueError, match=r"extra shards \(1\): shard_00002.parquet"):
        validate_checkpoint_training_data({"train_shards": SHARDS}, extra)


def test_training_data_validation_rejects_a_shard_with_a_different_size():
    changed = [SHARDS[0], {"name": "shard_00001.parquet", "size": 201}]
    with pytest.raises(ValueError, match=r"different size \(1\): shard_00001.parquet"):
        validate_checkpoint_training_data({"train_shards": SHARDS}, changed)


def test_training_data_validation_rejects_reordered_shards():
    with pytest.raises(ValueError, match="different order"):
        validate_checkpoint_training_data({"train_shards": SHARDS}, SHARDS[::-1])


def test_training_data_validation_shortens_long_lists_of_problems():
    saved = [{"name": f"shard_{i:05d}.parquet", "size": 1} for i in range(8)]
    with pytest.raises(ValueError, match=r"missing shards \(8\): .* and 3 more"):
        validate_checkpoint_training_data({"train_shards": saved}, [])


def test_training_data_validation_warns_for_checkpoints_without_an_inventory():
    with pytest.warns(RuntimeWarning, match="does not contain train_shards"):
        validate_checkpoint_training_data({}, SHARDS)


def test_tokenizer_validation_accepts_the_same_tokenizer():
    identity = {"files": {"a": "1", "b": "2"}}
    validate_checkpoint_tokenizer({"tokenizer": "gpt2", "tokenizer_identity": identity}, "gpt2", identity)


def test_tokenizer_validation_rejects_a_different_tokenizer_name():
    with pytest.raises(ValueError, match=r"trained with --tokenizer=gpt2 but --tokenizer=nanochat"):
        validate_checkpoint_tokenizer({"tokenizer": "gpt2"}, "nanochat")


def test_tokenizer_validation_treats_old_checkpoints_without_a_name_as_nanochat():
    validate_checkpoint_tokenizer({}, "nanochat")
    with pytest.raises(ValueError, match=r"--tokenizer=nanochat but --tokenizer=gpt2"):
        validate_checkpoint_tokenizer({}, "gpt2")


def test_tokenizer_validation_rejects_a_changed_tokenizer_file():
    saved = {"files": {"a": "1", "b": "2"}}
    with pytest.raises(ValueError, match=r"Tokenizer files differ.*: b"):
        validate_checkpoint_tokenizer({"tokenizer": "gpt2", "tokenizer_identity": saved}, "gpt2",
                                      {"files": {"a": "1", "b": "changed"}})


def test_tokenizer_validation_rejects_a_missing_tokenizer_file():
    saved = {"files": {"a": "1", "b": "2"}}
    with pytest.raises(ValueError, match=r"Tokenizer files differ.*: b"):
        validate_checkpoint_tokenizer({"tokenizer": "gpt2", "tokenizer_identity": saved}, "gpt2", {"files": {"a": "1"}})


def test_tokenizer_validation_allows_files_added_to_the_tokenizer_directory():
    saved = {"files": {"a": "1"}}
    validate_checkpoint_tokenizer({"tokenizer": "gpt2", "tokenizer_identity": saved}, "gpt2",
                                  {"files": {"a": "1", "other_encoding": "9"}})


def test_tokenizer_validation_checks_only_the_name_for_checkpoints_without_an_identity():
    validate_checkpoint_tokenizer({"tokenizer": "gpt2"}, "gpt2", {"files": {"a": "1"}})


@pytest.mark.parametrize("name", ["experiment-1", "d12.moe_adamw-s42"])
def test_experiment_name_validation_accepts_safe_names(name):
    assert validate_experiment_name(name) == name


@pytest.mark.parametrize("name", ["", "   ", ".", "..", "group/run", r"group\run"])
def test_experiment_name_validation_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        validate_experiment_name(name)
