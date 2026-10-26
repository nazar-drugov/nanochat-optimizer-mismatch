"""Regression tests for the simplified dense and MoE GPT variants."""

import pytest
import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import Block, DenseMLP, GPT, GPTConfig, MoEMLP, flash_attn


def make_model(seed=42, ffn_type="dense"):
    torch.manual_seed(seed)
    config = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        ffn_type=ffn_type,
        n_experts=4,
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


@pytest.mark.parametrize("ffn_type", ["dense", "moe"])
def test_gpt_uses_full_attention_for_training_and_kv_cache(monkeypatch, ffn_type):
    model = make_model(ffn_type=ffn_type)
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


def test_block_selects_requested_feed_forward_implementation():
    dense_config = GPTConfig(n_layer=1, n_head=2, n_kv_head=2, n_embd=16, ffn_type="dense")
    moe_config = GPTConfig(n_layer=1, n_head=2, n_kv_head=2, n_embd=16, ffn_type="moe")

    assert isinstance(Block(dense_config, layer_idx=0).mlp, DenseMLP)
    assert isinstance(Block(moe_config, layer_idx=0).mlp, MoEMLP)


def test_moe_forward_backward_routes_every_token_once():
    config = GPTConfig(n_layer=1, n_head=2, n_kv_head=2, n_embd=8, ffn_type="moe", n_experts=4)
    mlp = MoEMLP(config)

    # Make routing deterministic and ensure every expert receives tokens.
    with torch.no_grad():
        mlp.router.weight.zero_()
        mlp.router.weight[:, :config.n_experts].copy_(torch.eye(config.n_experts))
    token = torch.eye(config.n_experts, config.n_embd)
    x = token.unsqueeze(0).repeat(2, 1, 1).requires_grad_()

    token_counts = [None] * config.n_experts
    hooks = []
    for expert_idx, expert in enumerate(mlp.experts):
        def record_input_count(_module, args, expert_idx=expert_idx):
            token_counts[expert_idx] = args[0].shape[0]

        hooks.append(expert.register_forward_pre_hook(record_input_count))

    output = mlp(x)
    for hook in hooks:
        hook.remove()

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert token_counts == [2] * config.n_experts
    assert sum(token_counts) == x.shape[0] * x.shape[1]

    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert mlp.router.weight.grad is not None
    assert torch.isfinite(mlp.router.weight.grad).all()
    for expert in mlp.experts:
        assert all(parameter.grad is not None for parameter in expert.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in expert.parameters())


def test_moe_scales_selected_expert_output_by_router_probability():
    class OnesExpert(nn.Module):
        def forward(self, x):
            return torch.ones_like(x)

    config = GPTConfig(n_layer=1, n_head=2, n_kv_head=2, n_embd=8, ffn_type="moe", n_experts=4)
    mlp = MoEMLP(config)
    mlp.experts = nn.ModuleList(OnesExpert() for _ in range(config.n_experts))
    with torch.no_grad():
        mlp.router.weight.zero_()

    x = torch.randn(2, 3, config.n_embd)
    output = mlp(x)

    # Equal router logits give every expert probability 1 / n_experts, and
    # argmax deterministically selects expert zero.
    torch.testing.assert_close(output, torch.full_like(output, 1 / config.n_experts))


def test_moe_router_computation_is_float32_under_autocast():
    config = GPTConfig(n_layer=1, n_head=2, n_kv_head=2, n_embd=8, ffn_type="moe", n_experts=4)
    mlp = MoEMLP(config)
    router_output_dtypes = []
    hook = mlp.router.register_forward_hook(
        lambda _module, _args, output: router_output_dtypes.append(output.dtype)
    )

    x = torch.randn(2, 3, config.n_embd, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = mlp(x)
    hook.remove()

    assert router_output_dtypes == [torch.float32]
    assert output.dtype == x.dtype


def test_dense_and_moe_initialization_matches_for_shared_weights():
    dense = make_model(seed=1234, ffn_type="dense")
    moe = make_model(seed=1234, ffn_type="moe")

    assert all(isinstance(block.mlp, DenseMLP) for block in dense.transformer.h)
    assert all(isinstance(block.mlp, MoEMLP) for block in moe.transformer.h)

    torch.testing.assert_close(dense.transformer.wte.weight, moe.transformer.wte.weight, rtol=0, atol=0)
    torch.testing.assert_close(dense.lm_head.weight, moe.lm_head.weight, rtol=0, atol=0)
    for dense_block, moe_block in zip(dense.transformer.h, moe.transformer.h):
        for projection in ("c_q", "c_k", "c_v", "c_proj"):
            torch.testing.assert_close(
                getattr(dense_block.attn, projection).weight,
                getattr(moe_block.attn, projection).weight,
                rtol=0,
                atol=0,
            )


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
