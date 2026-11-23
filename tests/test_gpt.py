"""Regression tests for the simplified dense and MoE GPT variants."""

import pytest
import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import Block, DenseMLP, GPT, GPTConfig, MoEMLP, flash_attn


def make_model(seed=42, ffn_type="dense"):
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
    model.init_weights(seed=seed)
    return model


@pytest.mark.parametrize("ffn_type", ["dense", "moe"])
def test_same_seed_initializes_identical_parameters(ffn_type):
    first = make_model(seed=1234, ffn_type=ffn_type)
    second = make_model(seed=1234, ffn_type=ffn_type)

    first_parameters = dict(first.named_parameters())
    second_parameters = dict(second.named_parameters())
    assert first_parameters.keys() == second_parameters.keys()
    for name in first_parameters:
        torch.testing.assert_close(
            first_parameters[name],
            second_parameters[name],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("ffn_type", ["dense", "moe"])
def test_different_seeds_initialize_different_parameters(ffn_type):
    first = make_model(seed=1234, ffn_type=ffn_type)
    second = make_model(seed=5678, ffn_type=ffn_type)

    first_parameters = dict(first.named_parameters())
    second_parameters = dict(second.named_parameters())
    assert first_parameters.keys() == second_parameters.keys()
    assert any(
        not torch.equal(first_parameters[name], second_parameters[name])
        for name in first_parameters
    )


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


def test_moe_auxiliary_objective_and_router_gradients():
    model = make_model(ffn_type="moe")
    idx = torch.arange(16).reshape(2, 8) % model.config.vocab_size
    targets = (idx + 1) % model.config.vocab_size

    ce_loss = model(idx, targets, loss_reduction="none").mean()
    ce_loss.backward()
    ce_router_grads = [block.mlp.router.weight.grad.clone() for block in model.transformer.h]
    model.zero_grad(set_to_none=True)

    full_loss, metrics, router_stats = model(idx, targets, return_metrics=True)
    assert set(metrics) == {"ce", "balance", "z", "total"}
    assert all(value.ndim == 0 and torch.isfinite(value) and not value.requires_grad
               for value in metrics.values())
    torch.testing.assert_close(metrics["ce"], ce_loss.detach())
    torch.testing.assert_close(
        full_loss.detach(),
        metrics["ce"] + 0.01 * metrics["balance"] + 0.001 * metrics["z"],
    )
    torch.testing.assert_close(metrics["total"], full_loss.detach())
    torch.testing.assert_close(model(idx, targets).detach(), full_loss.detach())
    assert model(idx).shape == (2, 8, model.config.vocab_size)
    assert len(router_stats) == model.config.n_layer
    for layer in router_stats:
        assert set(layer) == {"expert_shares", "expert_counts", "router_entropy",
                              "mean_expert_probs", "max_vio", "z_loss"}
        assert all(not value.requires_grad for value in layer.values())
        torch.testing.assert_close(layer["expert_shares"].sum(), torch.tensor(1.0))
        assert layer["expert_counts"].sum().item() == idx.numel()

    full_loss.backward()
    for ce_grad, block in zip(ce_router_grads, model.transformer.h):
        aux_grad = block.mlp.router.weight.grad
        assert aux_grad is not None
        assert torch.isfinite(aux_grad).all()
        assert torch.any(aux_grad != 0)
        assert not torch.allclose(aux_grad, ce_grad)


def test_dense_model_reports_only_cross_entropy():
    model = make_model(ffn_type="dense")
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    loss, metrics, router_stats = model(idx, idx, return_metrics=True)
    assert router_stats == []
    assert set(metrics) == {"ce", "total"}
    torch.testing.assert_close(loss.detach(), metrics["ce"])


@pytest.mark.parametrize("ffn_type", ["dense", "swiglu", "moe"])
@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
@pytest.mark.parametrize("router_optimizer", ["same", "adamw", "muon"])
def test_optimizer_groups_cover_every_parameter_once(ffn_type, optimizer_name, router_optimizer):
    model = make_model(ffn_type=ffn_type)
    optimizer = model.setup_optimizer(
        optimizer_name=optimizer_name,
        router_optimizer=router_optimizer,
        adamw_beta1=0.81,
        adamw_beta2=0.93,
        adamw_eps=2e-8,
        adamw_matrix_lr=3e-3,
        muon_matrix_lr=0.03,
        muon_ns_steps=3,
        muon_variant="moonlight",
        weight_decay=0.1,
    )

    model_parameter_ids = [id(parameter) for parameter in model.parameters()]
    optimizer_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(optimizer_parameter_ids) == len(set(optimizer_parameter_ids))
    assert set(optimizer_parameter_ids) == set(model_parameter_ids)

    for group in optimizer.param_groups:
        if group["kind"] == "adamw":
            assert group["betas"] == (0.81, 0.93)
            assert group["eps"] == 2e-8
        else:
            assert group["ns_steps"] == 3
            assert group["variant"] == "moonlight"

    # Embeddings and lm_head are AdamW in every cell; matrices use the run's optimizer.
    roles = {}
    for group in optimizer.param_groups:
        roles.setdefault(group["role"], set()).add(group["kind"])
    assert roles["embedding"] == {"adamw"} and roles["unembedding"] == {"adamw"}
    assert roles["matrix"] == {optimizer_name}
    matrix_groups = [g for g in optimizer.param_groups if g["role"] == "matrix"]
    assert all(g["lr"] == (3e-3 if optimizer_name == "adamw" else 0.03) for g in matrix_groups)
    assert all(g["weight_decay"] == 0.1 for g in matrix_groups)

    if ffn_type == "moe":
        expected_router_kind = optimizer_name if router_optimizer == "same" else router_optimizer
        assert roles["router"] == {expected_router_kind}
        router_groups = [g for g in optimizer.param_groups if g["role"] == "router"]
        router_parameter_ids = {id(p) for block in model.transformer.h for p in block.mlp.router.parameters()}
        assert {id(p) for g in router_groups for p in g["params"]} == router_parameter_ids
        # Router LR defaults to the matrix LR of the optimizer the router uses; no weight decay.
        expected_lr = 3e-3 if expected_router_kind == "adamw" else 0.03
        assert all(g["lr"] == expected_lr and g["weight_decay"] == 0.0 for g in router_groups)
    else:
        assert "router" not in roles


def test_router_optimizer_same_puts_routers_in_muon_and_steps():
    """The spec's decision: routers follow the run's matrix optimizer."""
    model = make_model(ffn_type="moe")
    optimizer = model.setup_optimizer(optimizer_name="muon", router_optimizer="same", muon_variant="moonlight")
    router_groups = [g for g in optimizer.param_groups if g["role"] == "router"]
    assert router_groups and all(g["kind"] == "muon" for g in router_groups)
    before = [block.mlp.router.weight.detach().clone() for block in model.transformer.h]
    idx = torch.arange(16).reshape(2, 8) % model.config.vocab_size
    model(idx, (idx + 1) % model.config.vocab_size).backward()
    optimizer.step()
    for old, block in zip(before, model.transformer.h):
        assert not torch.equal(old, block.mlp.router.weight)


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

    output, aux_losses, router_stats = mlp(x)
    for hook in hooks:
        hook.remove()

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert token_counts == [2] * config.n_experts
    assert sum(token_counts) == x.shape[0] * x.shape[1]
    assert set(aux_losses) == {"balance", "z"}
    assert router_stats["expert_shares"].shape == (config.n_experts,)

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
    output, _, router_stats = mlp(x)

    # Equal router logits give every expert probability 1 / n_experts, and
    # argmax deterministically selects expert zero.
    torch.testing.assert_close(output, torch.full_like(output, 1 / config.n_experts))
    torch.testing.assert_close(router_stats["router_entropy"], torch.tensor(1.3862944))
    torch.testing.assert_close(router_stats["mean_expert_probs"], torch.full((4,), 0.25))
    torch.testing.assert_close(router_stats["expert_shares"], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_moe_nonuniform_router_statistics_against_independent_logits():
    import math

    config = GPTConfig(n_layer=1, n_head=2, n_kv_head=2, n_embd=8, ffn_type="moe", n_experts=4)
    mlp = MoEMLP(config)
    with torch.no_grad():
        mlp.router.weight.zero_()
        mlp.router.weight[:, :4].copy_(torch.eye(4))
    rows = [(3.0, 0.0, 0.0, 0.0), (0.0, 2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 4.0, 0.0)]
    x = torch.zeros((1, 4, 8))
    x[0, :, :4] = torch.tensor(rows)
    _, _, stats = mlp(x)

    # Compute references from the specified logits using scalar Python math.
    probabilities = []
    entropies = []
    squared_log_partitions = []
    for row in rows:
        weights = [math.exp(value) for value in row]
        partition = sum(weights)
        probs = [weight / partition for weight in weights]
        probabilities.append(probs)
        entropies.append(-sum(prob * math.log(prob) for prob in probs))
        squared_log_partitions.append(math.log(partition) ** 2)
    mean_probs = [sum(row[idx] for row in probabilities) / 4 for idx in range(4)]

    torch.testing.assert_close(stats["expert_shares"], torch.tensor([0.25, 0.5, 0.25, 0.0]))
    torch.testing.assert_close(stats["mean_expert_probs"], torch.tensor(mean_probs), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(stats["router_entropy"], torch.tensor(sum(entropies) / 4), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(stats["z_loss"], torch.tensor(sum(squared_log_partitions) / 4), rtol=1e-6, atol=1e-7)
    assert stats["mean_expert_probs"][3] > 0


def test_moe_router_computation_is_float32_under_autocast():
    config = GPTConfig(n_layer=1, n_head=2, n_kv_head=2, n_embd=8, ffn_type="moe", n_experts=4)
    mlp = MoEMLP(config)
    router_output_dtypes = []
    hook = mlp.router.register_forward_hook(
        lambda _module, _args, output: router_output_dtypes.append(output.dtype)
    )

    x = torch.randn(2, 3, config.n_embd, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output, _, _ = mlp(x)
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


@pytest.mark.parametrize("ffn_type", ["dense", "moe"])
def test_full_attention_cost_estimates(ffn_type):
    model = make_model(ffn_type=ffn_type)
    config = model.config
    head_dim = config.n_embd // config.n_head
    # Count the projections used by one token directly from the architecture.
    # MoE executes the router and one expert, regardless of how many are stored.
    matmul_params = model.lm_head.weight.numel()
    for block in model.transformer.h:
        matmul_params += sum(
            getattr(block.attn, name).weight.numel()
            for name in ("c_q", "c_k", "c_v", "c_proj")
        )
        if ffn_type == "moe":
            matmul_params += block.mlp.router.weight.numel()
            matmul_params += sum(
                getattr(block.mlp.experts[0], name).weight.numel()
                for name in ("up", "gate", "down")
            )
        else:
            matmul_params += block.mlp.c_fc.weight.numel()
            matmul_params += block.mlp.c_proj.weight.numel()
    assert model.num_matmul_params() == matmul_params

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
