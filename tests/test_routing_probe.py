"""Tests for fixed-batch top-1 MoE routing snapshots."""

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.routing_probe import (
    load_routing_snapshot,
    per_layer_change_rates,
    routing_probe,
    save_routing_snapshot,
    snapshot_change_rates,
)


def make_model(ffn_type="moe"):
    model = GPT(GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        ffn_type=ffn_type,
        n_experts=4,
    ), pad_vocab_size_to=1)
    model.init_weights(seed=7)
    return model


def test_probe_captures_one_choice_per_position_per_layer_and_restores_model():
    model = make_model()
    tokens = torch.arange(16).reshape(2, 8)

    model.train()
    choices = routing_probe(model, tokens)
    assert choices.shape == (2, 2, 8)
    assert choices.device.type == "cpu"
    assert torch.all((0 <= choices) & (choices < 4))
    assert model.training
    assert all(not block.mlp.router._forward_hooks for block in model.transformer.h)

    model.eval()
    torch.testing.assert_close(routing_probe(model, tokens), choices)
    assert not model.training


def test_probe_removes_hooks_and_restores_mode_after_forward_error(monkeypatch):
    model = make_model()
    router = model.transformer.h[1].mlp.router

    def fail(_inputs):
        raise RuntimeError("probe forward failed")

    monkeypatch.setattr(router, "forward", fail)
    with pytest.raises(RuntimeError, match="probe forward failed"):
        routing_probe(model, torch.arange(8).reshape(1, 8))
    assert model.training
    assert all(not block.mlp.router._forward_hooks for block in model.transformer.h)


def test_per_layer_rates_count_changed_positions_independently():
    previous = torch.zeros((2, 2, 4), dtype=torch.long)
    current = previous.clone()
    current[0, 0, 0] = 1
    current[0, 1, 3] = 2
    current[1, 0, 2] = 3

    torch.testing.assert_close(
        per_layer_change_rates(previous, current), torch.tensor([0.25, 0.125])
    )
    with pytest.raises(ValueError, match="same shape"):
        per_layer_change_rates(previous, current[:, :, :-1])


def test_snapshot_round_trip_and_rejects_changed_probe(tmp_path):
    tokens = torch.arange(8).reshape(1, 8)
    choices = torch.zeros((2, 1, 8), dtype=torch.long)
    path = tmp_path / "routing_000010.pt"
    save_routing_snapshot(path, 10, tokens, choices)
    previous = load_routing_snapshot(path)
    assert previous["step"] == 10
    torch.testing.assert_close(previous["probe_tokens"], tokens)
    torch.testing.assert_close(previous["choices"], choices)

    changed = choices.clone()
    changed[1, 0, 0] = 1
    current = {"step": 20, "probe_tokens": tokens.clone(), "choices": changed}
    torch.testing.assert_close(
        snapshot_change_rates(previous, current), torch.tensor([0.0, 0.125])
    )

    current["probe_tokens"] = tokens + 1
    with pytest.raises(ValueError, match="different probe tokens"):
        snapshot_change_rates(previous, current)
    with pytest.raises(FileExistsError):
        save_routing_snapshot(path, 10, tokens, choices)


def test_probe_rejects_dense_model():
    with pytest.raises(ValueError, match="MoE model"):
        routing_probe(make_model("dense"), torch.arange(8).reshape(1, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA precision modes")
def test_cuda_probe_uses_training_matmul_precision_and_restores_caller():
    model = make_model().cuda()
    observed = []
    hook = model.transformer.h[0].mlp.router.register_forward_hook(
        lambda _module, _inputs, _output: observed.append(torch.get_float32_matmul_precision())
    )
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        routing_probe(model, torch.arange(8, device="cuda").reshape(1, 8))
        assert observed == ["high"]
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        hook.remove()
        torch.set_float32_matmul_precision(previous)


def test_missing_snapshot_in_disposable_directory_fails_with_path(tmp_path):
    missing = tmp_path / "fresh-resume" / "routing_000001.pt"
    with pytest.raises(FileNotFoundError, match="routing_000001.pt"):
        load_routing_snapshot(missing)


def test_malformed_snapshot_rejected_before_change_rate(tmp_path):
    path = tmp_path / "routing_000003.pt"
    good = {
        "step": 3,
        "probe_tokens": torch.arange(8).reshape(1, 8),
        "choices": torch.zeros((2, 1, 8), dtype=torch.long),
    }
    for altered, message in [
        ({"choices": torch.zeros((2, 8), dtype=torch.long)}, "shape"),
        ({"choices": torch.zeros((2, 1, 7), dtype=torch.long)}, "shape"),
        ({"choices": torch.full((2, 1, 8), 4)}, "out-of-range"),
        ({"step": 2}, "wrong step"),
    ]:
        torch.save({**good, **altered}, path)
        with pytest.raises(ValueError, match=message):
            load_routing_snapshot(path, expected_step=3, n_experts=4)

    with pytest.raises(ValueError, match="different probe tokens"):
        snapshot_change_rates(good, {**good, "probe_tokens": good["probe_tokens"] + 1})
