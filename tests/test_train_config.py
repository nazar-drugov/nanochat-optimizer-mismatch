"""Shared training options: default resolution, schedules and run records."""

import argparse
import json

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.train_config import (
    ADAMW_MATRIX_LR_DEFAULT, MUON_MATRIX_LR_DEFAULTS, add_moe_args, add_optimizer_args,
    muon_momentum_at, resolve_moe_args, resolve_optimizer_args, wsd_lr_multiplier, write_run_record,
)


def parse(argv, weight_decay_default=0.1):
    parser = argparse.ArgumentParser()
    add_moe_args(parser)
    add_optimizer_args(parser, weight_decay_default=weight_decay_default)
    return parser.parse_args(argv)


def test_defaults_follow_the_agreed_settings():
    args = parse([])
    kwargs = resolve_optimizer_args(args)
    moe = resolve_moe_args(args)
    assert kwargs["optimizer_name"] == "adamw" and kwargs["router_optimizer"] == "same"
    assert kwargs["muon_variant"] == "plain" and args.muon_matrix_lr == MUON_MATRIX_LR_DEFAULTS["plain"] == 4e-3
    assert kwargs["adamw_matrix_lr"] == ADAMW_MATRIX_LR_DEFAULT == 2e-3
    assert kwargs["weight_decay"] == 0.1 and kwargs["router_weight_decay"] == 0.0
    assert kwargs["muon_momentum"] == 0.95 and args.muon_momentum_schedule == "constant"
    assert kwargs["max_grad_norm"] == 1.0 and kwargs["track_grad_norm"] is True
    assert kwargs["router_lr"] is None  # follows the matrix LR of the router's optimizer
    assert moe == dict(ffn_type="dense", n_experts=4, moe_layers="all", balancer="aux",
                       balance_coef=0.01, z_coef=0.001, bias_update_rate=0.001)


@pytest.mark.parametrize("argv, expected_lr", [
    (["--muon-variant", "keller"], 0.02),
    (["--muon-variant", "plain"], 4e-3),
    (["--muon-variant", "keller", "--muon-matrix-lr", "0.01"], 0.01),
])
def test_muon_lr_default_depends_on_variant_unless_given(argv, expected_lr):
    args = parse(argv)
    assert resolve_optimizer_args(args)["muon_matrix_lr"] == expected_lr


def test_nanochat_muon_variant_is_not_offered():
    with pytest.raises(SystemExit):
        parse(["--muon-variant", "nanochat"])


def test_grad_clip_disable_and_z_coef_per_balancer():
    assert resolve_optimizer_args(parse(["--grad-clip", "0"]))["max_grad_norm"] is None
    assert resolve_moe_args(parse(["--balancer", "bias"]))["z_coef"] == 0.0
    assert resolve_moe_args(parse(["--balancer", "bias", "--z-coef", "0.001"]))["z_coef"] == 0.001
    assert resolve_moe_args(parse(["--balancer", "aux"]))["z_coef"] == 0.001


def test_resolved_kwargs_build_an_optimizer():
    args = parse(["--optimizer-name", "muon", "--ffn-type", "moe"])
    model = GPT(GPTConfig(sequence_len=16, vocab_size=32, n_layer=2, n_head=2, n_kv_head=2, n_embd=16,
                          **resolve_moe_args(args)), pad_vocab_size_to=1)
    model.init_weights()
    optimizer = model.setup_optimizer(**resolve_optimizer_args(args))
    assert optimizer.max_grad_norm == 1.0
    assert {g["kind"] for g in optimizer.param_groups if g["role"] in ("matrix", "router")} == {"muon"}


def test_muon_momentum_schedules():
    args = parse([])
    assert all(muon_momentum_at(step, 1000, 0.2, args) == 0.95 for step in (0, 500, 999))
    args = parse(["--muon-momentum-schedule", "nanochat"])
    assert muon_momentum_at(0, 1000, 0.2, args) == pytest.approx(0.85)
    assert muon_momentum_at(400, 1000, 0.2, args) == pytest.approx(0.97)
    assert muon_momentum_at(800, 1000, 0.2, args) == pytest.approx(0.97)
    assert muon_momentum_at(900, 1000, 0.2, args) == pytest.approx(0.935)


def test_wsd_schedule_shape():
    lrm = lambda step: wsd_lr_multiplier(step, num_iterations=100, warmup_iters=10, decay_iters=20, final_lr_frac=0.0)
    assert lrm(0) == pytest.approx(0.1) and lrm(9) == pytest.approx(1.0)
    assert lrm(10) == 1.0 and lrm(80) == 1.0
    assert lrm(90) == pytest.approx(0.5) and lrm(100) == pytest.approx(0.0)
    values = [lrm(step) for step in range(80, 101)]
    assert all(a >= b for a, b in zip(values, values[1:]))  # monotone decay
    assert wsd_lr_multiplier(5, 100, 0, 0, 0.0) == 1.0
    assert wsd_lr_multiplier(100, 100, 0, 20, 0.1) == pytest.approx(0.1)


def test_warmup_tokens_match_the_paper_at_256k():
    # base_train: warmup_steps = ceil(warmup_tokens / total_batch_size)
    assert -(-130_000_000 // 262_144) == 496


def test_run_record_contains_everything_needed_to_reproduce(tmp_path):
    args = parse(["--ffn-type", "moe", "--optimizer-name", "muon"])
    model = GPT(GPTConfig(sequence_len=16, vocab_size=32, n_layer=2, n_head=2, n_kv_head=2, n_embd=16,
                          **resolve_moe_args(args)), pad_vocab_size_to=1)
    model.init_weights()
    optimizer = model.setup_optimizer(**resolve_optimizer_args(args))
    path = tmp_path / "run" / "config.json"
    write_run_record(path, args=args, model=model, optimizer=optimizer, device_type="cpu", world_size=1,
                     compute_dtype=torch.float32, data={"train_data_dir": "/data"}, extra={"stage": "pretrain"})
    record = json.loads(path.read_text())
    assert record["args"]["optimizer_name"] == "muon" and record["args"]["muon_matrix_lr"] == 4e-3
    assert record["model_config"]["balancer"] == "aux" and record["moe_layer_indices"] == [0, 1]
    assert set(record["parameter_counts"]) >= {"embedding", "active_non_embedding", "active_with_embeddings", "total"}
    roles = {g["role"]: g["kind"] for g in record["optimizer"]["groups"]}
    assert roles == {"unembedding": "adamw", "embedding": "adamw", "router": "muon", "matrix": "muon"}
    assert record["optimizer"]["max_grad_norm"] == 1.0
    assert sum(g["num_params"] for g in record["optimizer"]["groups"]) == record["parameter_counts"]["total"]
    assert set(record["git"]) == {"commit", "branch", "dirty", "dirty_files"} and record["git"]["commit"]
    assert record["environment"]["world_size"] == 1 and record["data"]["train_data_dir"] == "/data"
    assert record["stage"] == "pretrain"
