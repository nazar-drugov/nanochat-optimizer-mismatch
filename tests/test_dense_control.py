"""The explicitly routing-loss-free E=1 control must optimize as dense SwiGLU."""

import argparse
import json
from dataclasses import asdict

import pytest
import torch

from nanochat.checkpoint_manager import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_routing_loss,
)
from nanochat.finetune_metrics import flip_cause_counts, route_comparison, separation, top2_margin
from nanochat.gpt import GPT, GPTConfig, make_gpt_config
from nanochat.train_config import add_moe_args, add_optimizer_args, resolve_moe_args, resolve_optimizer_args, write_run_record


def parse(argv):
    parser = argparse.ArgumentParser()
    add_moe_args(parser)
    add_optimizer_args(parser, weight_decay_default=0.1)
    return parser.parse_args(argv)


def build(config):
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights(seed=19)
    return model


def small_config(**overrides):
    return GPTConfig(**(dict(sequence_len=16, vocab_size=32, n_layer=2, n_head=2,
                             n_kv_head=2, n_embd=16, ffn_type="moe", n_experts=1,
                             moe_dispatch="loop") | overrides))


def test_explicit_flag_resolves_both_coefficients_and_persists(tmp_path):
    args = parse(["--ffn-type", "moe", "--n-experts", "1", "--disable-routing-loss",
                  "--balance-coef", "0.4", "--z-coef", "0.2"])
    resolved = resolve_moe_args(args)
    assert resolved["disable_routing_loss"] is True and args.disable_routing_loss
    assert args.balance_coef == args.z_coef == 0.0
    config = make_gpt_config(depth=2, aspect_ratio=8, head_dim=8, max_seq_len=16, vocab_size=32,
                             **resolved)
    assert config.disable_routing_loss and config.balance_coef == config.z_coef == 0.0
    model = build(config)
    optimizer = model.setup_optimizer(**resolve_optimizer_args(args))
    write_run_record(tmp_path / "config.json", args=args, model=model, optimizer=optimizer,
                     device_type="cpu", world_size=1, compute_dtype=torch.float32, data={})
    record = json.loads((tmp_path / "config.json").read_text())
    for saved in (record["args"], record["model_config"]):
        assert saved["disable_routing_loss"] is True and saved["balance_coef"] == saved["z_coef"] == 0.0
    save_checkpoint(str(tmp_path), 0, model.state_dict(), optimizer.state_dict(), {"model_config": asdict(config)})
    _, _, metadata = load_checkpoint(str(tmp_path), 0, "cpu", load_optimizer=True)
    restored = GPTConfig(**metadata["model_config"])
    assert restored.disable_routing_loss and restored.balance_coef == restored.z_coef == 0.0
    validate_checkpoint_routing_loss(metadata, config)


@pytest.mark.parametrize("ffn,experts", [("moe", 4), ("moe", 8), ("swiglu", 1), ("dense", 1)])
def test_disable_routing_loss_is_only_valid_for_one_expert_moe(ffn, experts):
    args = parse(["--ffn-type", ffn, "--n-experts", str(experts), "--disable-routing-loss"])
    with pytest.raises(ValueError, match="only for the E=1 MoE"):
        resolve_moe_args(args)
    with pytest.raises(ValueError, match="only for the E=1 MoE"):
        small_config(ffn_type=ffn, n_experts=experts, disable_routing_loss=True)


def test_resume_cannot_toggle_policy_and_old_unflagged_configs_work():
    legacy = small_config(disable_routing_loss=False)
    flagged = small_config(disable_routing_loss=True)
    old_metadata = {"model_config": {k: v for k, v in asdict(legacy).items() if k != "disable_routing_loss"}}
    validate_checkpoint_routing_loss(old_metadata, legacy)
    assert not GPTConfig(**old_metadata["model_config"]).disable_routing_loss
    with pytest.raises(ValueError, match="policy differs"):
        validate_checkpoint_routing_loss(old_metadata, flagged)
    with pytest.raises(ValueError, match="policy differs"):
        validate_checkpoint_routing_loss({"model_config": asdict(flagged)}, legacy)
    with pytest.raises(ValueError, match="zero balance/z"):
        validate_checkpoint_routing_loss({"model_config": asdict(flagged) | {"z_coef": .001}}, flagged)


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_e1_control_matches_dense_swiglu_over_multiple_clipped_steps(optimizer_name):
    # Identical expert/shared weights and batch stream. The single router has
    # zero gradient, so it cannot contribute to clipping or the matrix update.
    e1 = build(small_config(disable_routing_loss=True))
    dense = build(small_config(ffn_type="swiglu"))
    kwargs = dict(optimizer_name=optimizer_name, muon_variant="moonlight",
                  adamw_matrix_lr=1e-3, muon_matrix_lr=8e-4, embedding_lr=.03,
                  unembedding_lr=.008, weight_decay=.1, router_weight_decay=0.,
                  max_grad_norm=.03, track_grad_norm=True)
    e1_optimizer, dense_optimizer = e1.setup_optimizer(**kwargs), dense.setup_optimizer(**kwargs)
    e1_params, dense_params = dict(e1.named_parameters()), dict(dense.named_parameters())
    shared_params = [(p, dense_params[name.replace(".experts.0.", ".")])
                     for name, p in e1_params.items() if ".router." not in name]
    assert len(shared_params) == len(dense_params)
    router_initial = {name: p.detach().clone() for name, p in e1_params.items() if ".router." in name}
    assert router_initial
    for p, q in shared_params:
        torch.testing.assert_close(p, q, rtol=0., atol=0.)
    for step in range(3):
        generator = torch.Generator().manual_seed(181 + step)
        ids = torch.randint(0, 32, (2, 8), generator=generator)
        targets = torch.roll(ids, -1, dims=1)
        replay = {layer: torch.zeros_like(ids) for layer in e1.moe_layer_indices}
        torch.testing.assert_close(e1(ids, targets, loss_reduction="none", forced_routes=replay),
                                   e1(ids, targets, loss_reduction="none"), rtol=0., atol=0.)
        torch.testing.assert_close(e1(ids), dense(ids), rtol=2e-5, atol=2e-6)
        e1_loss, metrics, router = e1(ids, targets, return_metrics=True)
        dense_loss = dense(ids, targets)
        torch.testing.assert_close(e1_loss, dense_loss, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(e1_loss.detach(), metrics["ce"], rtol=0., atol=0.)
        assert metrics["balance"].item() == len(e1.moe_layer_indices)
        assert torch.isfinite(metrics["z"])
        for stats in router:
            assert stats["router_entropy"].item() == stats["max_vio"].item() == 0.0
            assert stats["expert_shares"].tolist() == stats["mean_expert_probs"].tolist() == [1.0]
        e1_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)
        e1_loss.backward()
        dense_loss.backward()
        for p, q in shared_params:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)
        for name in router_initial:
            grad = e1_params[name].grad
            assert grad is not None and torch.count_nonzero(grad).item() == 0
        e1_optimizer.step()
        dense_optimizer.step()
        assert e1_optimizer.last_clip_coef.item() < 1.0 and dense_optimizer.last_clip_coef.item() < 1.0
        torch.testing.assert_close(e1_optimizer.last_grad_norm, dense_optimizer.last_grad_norm, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(e1_optimizer.last_clip_coef, dense_optimizer.last_clip_coef, rtol=2e-6, atol=2e-6)
        for p, q in shared_params:
            torch.testing.assert_close(p, q, rtol=2e-5, atol=2e-6)
        for name, initial in router_initial.items():
            torch.testing.assert_close(e1_params[name], initial, rtol=0., atol=0.)


def test_single_expert_route_metrics_zero_or_explicitly_undefined():
    routes = torch.zeros(11, dtype=torch.long)
    stats = route_comparison(routes, routes, n_experts=1)
    for name in ("flip_rate", "chance_flip_rate", "tv", "churn", "maxvio", "base_maxvio"):
        assert stats[name] == 0.0
    assert stats["load"] == [1.0]
    assert stats["flip_rate_normalized"] is None
    assert top2_margin(torch.randn(11, 1)) is None
    assert separation(torch.ones(1), torch.ones(1)) == 1.0
    inputs, weight = torch.randn(11, 4), torch.randn(1, 4)
    causes = flip_cause_counts(inputs, inputs + 1, weight, weight + 1, routes, routes)
    assert set(causes.values()) == {0}
    assert json.loads(json.dumps(stats, allow_nan=False))["flip_rate_normalized"] is None
