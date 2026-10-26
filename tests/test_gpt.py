"""Regression tests for the simplified dense GPT baseline."""

import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig, flash_attn


def make_model(seed=42):
    torch.manual_seed(seed)
    config = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
    )
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()
    return model


def test_dense_forward_backward():
    model = make_model()
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, (2, 8))

    logits = model(idx)
    assert logits.shape == (2, 8, model.config.vocab_size)
    assert logits.dtype == torch.float32

    loss = model(idx, targets)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_dense_optimizer_covers_every_parameter_once():
    model = make_model()
    optimizer = model.setup_optimizer()

    model_parameter_ids = [id(parameter) for parameter in model.parameters()]
    optimizer_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]

    assert len(optimizer_parameter_ids) == len(set(optimizer_parameter_ids))
    assert set(optimizer_parameter_ids) == set(model_parameter_ids)
    assert {group["kind"] for group in optimizer.param_groups} == {"adamw", "muon"}


def test_dense_state_dict_round_trip():
    model = make_model(seed=1)
    restored = make_model(seed=2)

    incompatible_keys = restored.load_state_dict(model.state_dict(), strict=True)
    assert incompatible_keys.missing_keys == []
    assert incompatible_keys.unexpected_keys == []

    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    torch.testing.assert_close(restored(idx), model(idx))


def test_gpt_uses_full_attention_for_training_and_kv_cache(monkeypatch):
    model = make_model()
    idx = torch.randint(0, model.config.vocab_size, (1, 4))
    training_calls = []
    cache_calls = []

    original_training_attention = flash_attn.flash_attn_func
    original_cache_attention = flash_attn.flash_attn_with_kvcache

    def record_training_attention(*args, **kwargs):
        training_calls.append((kwargs["causal"], kwargs["window_size"]))
        return original_training_attention(*args, **kwargs)

    def record_cache_attention(*args, **kwargs):
        cache_calls.append((kwargs["causal"], kwargs["window_size"]))
        return original_cache_attention(*args, **kwargs)

    monkeypatch.setattr(flash_attn, "flash_attn_func", record_training_attention)
    monkeypatch.setattr(flash_attn, "flash_attn_with_kvcache", record_cache_attention)

    model(idx)
    assert training_calls == [(True, (-1, 0))] * model.config.n_layer

    head_dim = model.config.n_embd // model.config.n_head
    kv_cache = KVCache(
        batch_size=1,
        num_heads=model.config.n_kv_head,
        seq_len=model.config.sequence_len,
        head_dim=head_dim,
        num_layers=model.config.n_layer,
        device=model.get_device(),
        dtype=COMPUTE_DTYPE,
    )
    model(idx, kv_cache=kv_cache)

    assert cache_calls == [(True, (-1, 0))] * model.config.n_layer
    assert kv_cache.get_pos() == idx.size(1)


def test_full_attention_cost_estimates():
    model = make_model()
    config = model.config
    head_dim = config.n_embd // config.n_head
    matmul_params = model.num_matmul_params()

    expected_training = (
        6 * matmul_params
        + config.n_layer * 12 * config.n_head * head_dim * config.sequence_len
    )
    assert model.estimate_flops() == expected_training

    context_len = 7
    expected_decode = (
        2 * matmul_params
        + config.n_layer * 4 * config.n_head * head_dim * context_len
    )
    assert model.estimate_decode_flops(context_len) == expected_decode

    num_tokens = 5
    attended_tokens = num_tokens * (num_tokens + 1) // 2
    expected_prefill = (
        2 * matmul_params * num_tokens
        + config.n_layer * 4 * config.n_head * head_dim * attended_tokens
    )
    assert model.estimate_prefill_flops(num_tokens) == expected_prefill

    expected_kv_bytes_per_token = (
        config.n_layer
        * 2
        * config.n_kv_head
        * head_dim
        * COMPUTE_DTYPE.itemsize
    )
    assert model.kv_bytes_per_token() == expected_kv_bytes_per_token
    assert model.kv_read_bytes(context_len) == expected_kv_bytes_per_token * context_len
