"""
Correctness checks required before the pretraining runs (EXPERIMENT_SPEC.md §8),
plus the MoE features the runs depend on: balancers, MoE-layer selection, route
capture/replay hooks, parameter-count conventions and checkpoint compatibility.

python -m pytest tests/test_pretraining_spec.py -v
"""

import pytest
import torch

from nanochat.checkpoint_manager import checkpoint_tokenizer_name, patch_missing_expert_bias
from nanochat.gpt import GPT, GPTConfig, MoEMLP, SwiGLUMLP, moe_layer_indices
from nanochat.routing_probe import routing_probe
from nanochat.training_stats import accumulate_router_stats, add_router_log_fields, update_batch_fingerprint


def make_model(seed=42, **overrides):
    values = dict(sequence_len=16, vocab_size=32, n_layer=2, n_head=2, n_kv_head=2, n_embd=16,
                  ffn_type="moe", n_experts=4)
    values.update(overrides)
    model = GPT(GPTConfig(**values), pad_vocab_size_to=1)
    model.init_weights(seed=seed)
    return model


def batch(seed=0, B=2, T=8, vocab=32):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, vocab, (B, T), generator=gen)
    return idx, torch.roll(idx, -1, dims=1)


def skew_router(model, expert=0, strength=5.0):
    """Make one expert win almost every token (a badly imbalanced router)."""
    with torch.no_grad():
        for block in model.transformer.h:
            if block.is_moe:
                block.mlp.router.weight.zero_()
                block.mlp.router.weight[expert].fill_(strength)


# -----------------------------------------------------------------------------
# §8.1 Dense equivalence: E=1 MoE == dense SwiGLU model (the E=1 control)

@pytest.mark.parametrize("seed", [0, 7])
def test_e1_moe_reproduces_dense_swiglu_model(seed):
    moe = make_model(seed=seed, ffn_type="moe", n_experts=1, balancer="none", z_coef=0.0)
    dense = make_model(seed=seed, ffn_type="swiglu")
    # Same seed -> identical shared weights and expert 0 == the dense MLP
    for block_moe, block_dense in zip(moe.transformer.h, dense.transformer.h):
        assert isinstance(block_moe.mlp, MoEMLP) and isinstance(block_dense.mlp, SwiGLUMLP)
        for name in ("up", "gate", "down"):
            torch.testing.assert_close(getattr(block_moe.mlp.experts[0], name).weight,
                                       getattr(block_dense.mlp, name).weight, rtol=0, atol=0)
    idx, targets = batch(seed)
    # softmax over one expert is exactly 1.0, so the gate is the identity
    torch.testing.assert_close(moe(idx), dense(idx), rtol=0, atol=0)
    torch.testing.assert_close(moe(idx, targets), dense(idx, targets), rtol=0, atol=0)


def test_e1_with_default_balancer_only_adds_constant_and_z_terms():
    moe = make_model(ffn_type="moe", n_experts=1)  # aux 0.01, z 0.001 defaults
    idx, targets = batch(1)
    loss, metrics, _ = moe(idx, targets, return_metrics=True)
    torch.testing.assert_close(metrics["balance"], torch.tensor(1.0))  # E * f * P = 1 * 1 * 1
    torch.testing.assert_close(loss.detach(), metrics["ce"] + 0.01 * 1.0 + 0.001 * metrics["z"])


# -----------------------------------------------------------------------------
# §8.2 Self-replay identity: forcing the model's own routes changes nothing

@pytest.mark.parametrize("balancer", ["aux", "bias", "none"])
def test_self_replay_reproduces_loss_exactly(balancer):
    model = make_model(balancer=balancer)
    if balancer == "bias":  # make the bias actually change some routes
        for block in model.transformer.h:
            block.mlp.expert_bias.copy_(torch.tensor([0.3, -0.2, 0.1, 0.0]))
    idx, targets = batch(2)
    captured = []
    logits = model(idx, routing_capture=captured)
    assert [entry["layer"] for entry in captured] == model.moe_layer_indices
    routes = {entry["layer"]: entry["choices"] for entry in captured}
    stacked = torch.stack([routes[i] for i in range(model.config.n_layer)])
    torch.testing.assert_close(model(idx, forced_routes=routes), logits, rtol=0, atol=0)
    torch.testing.assert_close(model(idx, forced_routes=stacked), logits, rtol=0, atol=0)
    per_token = model(idx, targets, loss_reduction="none")
    torch.testing.assert_close(model(idx, targets, loss_reduction="none", forced_routes=routes), per_token, rtol=0, atol=0)


def test_bias_changes_selection_but_not_gate_value():
    model = make_model(balancer="bias")
    idx, _ = batch(3)
    captured = []
    model(idx, routing_capture=captured)
    routes = {entry["layer"]: entry["choices"] for entry in captured}
    reference = model(idx, forced_routes=routes)
    for block in model.transformer.h:
        block.mlp.expert_bias.copy_(torch.tensor([5.0, -5.0, 2.0, 0.0]))
    # With routes forced, the bias has no effect: it never enters the gate value
    torch.testing.assert_close(model(idx, forced_routes=routes), reference, rtol=0, atol=0)
    # Without forcing, a large bias does move tokens to expert 0
    biased = []
    model(idx, routing_capture=biased)
    assert all((entry["choices"] == 0).all() for entry in biased)


def test_forced_routes_validate_shape():
    model = make_model()
    idx, _ = batch(0)
    with pytest.raises(ValueError, match="forced_idx has shape"):
        model(idx, forced_routes={0: torch.zeros(1, 3, dtype=torch.long)})


# -----------------------------------------------------------------------------
# §8.3 Router gradient from the language-model loss alone

def test_router_gets_lm_gradient_with_balancer_off():
    model = make_model(balancer="none", z_coef=0.0)
    idx, targets = batch(4)
    loss, metrics, _ = model(idx, targets, return_metrics=True)
    torch.testing.assert_close(loss.detach(), metrics["ce"])  # nothing but cross-entropy
    loss.backward()
    for block in model.transformer.h:
        grad = block.mlp.router.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_e1_router_gets_no_lm_gradient():
    """Documented property of the E=1 control: its gate is identically 1."""
    model = make_model(n_experts=1, balancer="none", z_coef=0.0)
    idx, targets = batch(4)
    model(idx, targets).backward()
    for block in model.transformer.h:
        assert block.mlp.router.weight.grad.abs().sum() == 0


# -----------------------------------------------------------------------------
# §8.4 Counter sanity

def test_expert_counts_sum_to_tokens_and_match_routes():
    model = make_model(n_layer=3)
    idx, targets = batch(5, B=3, T=8)
    captured = []
    _, _, stats = model(idx, targets, return_metrics=True, routing_capture=captured)
    for layer_stats, entry in zip(stats, captured):
        counts = layer_stats["expert_counts"]
        assert counts.dtype == torch.long and counts.sum().item() == idx.numel()
        torch.testing.assert_close(counts, torch.bincount(entry["choices"].flatten(), minlength=4))
        torch.testing.assert_close(layer_stats["expert_shares"], counts.float() / idx.numel())
        torch.testing.assert_close(layer_stats["max_vio"], 4 * layer_stats["expert_shares"].max() - 1)


def test_accumulated_counts_are_summed_not_averaged():
    model = make_model()
    sums = None
    total = 0
    for seed in range(3):
        idx, targets = batch(seed)
        _, _, stats = model(idx, targets, return_metrics=True)
        sums = accumulate_router_stats(sums, stats, micro_steps=3)
        total += idx.numel()
    assert sums["expert_counts"].shape == (2, 4)
    assert (sums["expert_counts"].sum(dim=1) == total).all()
    torch.testing.assert_close(sums["expert_shares"].sum(dim=1), torch.ones(2))


# -----------------------------------------------------------------------------
# §8.5 Positive control (mechanics only; the real check needs a trained model)

def test_scrambled_routes_change_loss_and_replay_restores_it():
    model = make_model()
    idx, targets = batch(6)
    captured = []
    clean = model(idx, targets, loss_reduction="none", routing_capture=captured)
    routes = {entry["layer"]: entry["choices"] for entry in captured}
    scrambled = {layer: (choices + 1) % 4 for layer, choices in routes.items()}
    assert not torch.equal(model(idx, targets, loss_reduction="none", forced_routes=scrambled), clean)
    torch.testing.assert_close(model(idx, targets, loss_reduction="none", forced_routes=routes), clean, rtol=0, atol=0)


# -----------------------------------------------------------------------------
# §8.6 Pairing: the data fingerprint is deterministic and order-sensitive

def test_batch_fingerprint_is_deterministic_and_order_sensitive():
    batches = [batch(seed, B=2, T=8)[0] for seed in range(3)]
    def run(sequence):
        fp, weights = torch.zeros((), dtype=torch.long), None
        for b in sequence:
            fp, weights = update_batch_fingerprint(fp, b, weights)
        return fp.item()
    assert run(batches) == run([b.clone() for b in batches])
    assert run(batches) != run(batches[::-1])
    swapped = batches[0].clone()
    swapped[0, 0], swapped[0, 1] = batches[0][0, 1], batches[0][0, 0]
    if not torch.equal(swapped, batches[0]):
        assert run([swapped] + batches[1:]) != run(batches)


# -----------------------------------------------------------------------------
# §8.7 Balancer behaviour

def skewed_layer_inputs(model, n_tokens=512, seed=0):
    """Diverse inputs sharing a common positive direction, and a router whose
    expert 0 is aligned with it: expert 0 wins almost every token. (Inputs need
    realistic diversity: with near-identical tokens, any balancer moves them all
    at once and oscillates.)"""
    gen = torch.Generator().manual_seed(seed)
    d = model.config.n_embd
    x = 1.0 + torch.randn(1, n_tokens, d, generator=gen)
    with torch.no_grad():
        for block in model.transformer.h:
            if block.is_moe:
                block.mlp.router.weight.zero_()
                block.mlp.router.weight[0].fill_(0.05)
                block.mlp.router.weight[1:].normal_(0.0, 0.05, generator=gen)
    return x


def test_bias_balancer_reduces_max_vio_on_fixed_inputs():
    model = make_model(balancer="bias", n_experts=4)
    x = skewed_layer_inputs(model)
    mlps = [block.mlp for block in model.transformer.h]

    def step_stats():
        stats = [mlp(x)[2] for mlp in mlps]
        return torch.stack([s["max_vio"] for s in stats]), torch.stack([s["expert_counts"] for s in stats])

    start, _ = step_stats()
    assert (start > 2.5).all()  # expert 0 takes (nearly) everything: MaxVio close to E - 1 = 3
    history = []
    for _ in range(400):
        max_vio, counts = step_stats()
        history.append(max_vio)
        model.update_expert_bias(counts, rate=0.002)
    # The sign rule settles into a small oscillation around balance; judge the tail
    # (measured 2026-11-23: ~3.0 -> ~0.05)
    tail = torch.stack(history[-50:]).mean(dim=0)
    assert (tail < 0.2).all(), f"MaxVio did not fall enough: {start.tolist()} -> {tail.tolist()}"
    bias = model.expert_bias_state()
    assert bias.shape == (2, 4) and (bias[:, 0] < bias[:, 1:].min(dim=1).values).all()


def test_update_expert_bias_sign_rule_shape_check_and_noop():
    model = make_model(balancer="bias")
    counts = torch.tensor([[10, 0, 5, 5], [5, 5, 5, 5]])
    model.update_expert_bias(counts, rate=0.1)
    torch.testing.assert_close(model.transformer.h[0].mlp.expert_bias, torch.tensor([-0.1, 0.1, 0.0, 0.0]))
    torch.testing.assert_close(model.transformer.h[1].mlp.expert_bias, torch.zeros(4))
    with pytest.raises(ValueError):
        model.update_expert_bias(torch.zeros(3, 4))
    aux = make_model(balancer="aux")
    aux.update_expert_bias(counts, rate=0.1)
    assert all((block.mlp.expert_bias == 0).all() for block in aux.transformer.h)


def test_aux_balance_gradient_moves_probability_off_the_overloaded_expert():
    """One small gradient step on the Switch loss E * sum_i f_i P_i lowers the mean
    probability of the overloaded expert and lowers the loss at the old loads f.
    (Repeated large steps overshoot: every token flips to another expert at once.)"""
    model = make_model(balancer="aux", z_coef=0.0)
    x = skewed_layer_inputs(model)
    mlp = model.transformer.h[0].mlp
    _, aux, stats = mlp(x)
    old_shares = stats["expert_shares"].clone()
    assert old_shares[0] > 0.9
    aux["balance"].backward()
    assert mlp.router.weight.grad.abs().sum() > 0
    with torch.no_grad():
        mlp.router.weight -= 0.05 * mlp.router.weight.grad
    _, aux_after, stats_after = mlp(x)
    assert stats_after["mean_expert_probs"][0] < stats["mean_expert_probs"][0]
    loss_at_old_loads = lambda probs: 4 * (old_shares * probs).sum()
    assert loss_at_old_loads(stats_after["mean_expert_probs"]) < loss_at_old_loads(stats["mean_expert_probs"])


def test_balancer_coefficients_enter_the_loss_as_configured():
    idx, targets = batch(9)
    for balancer, coef, z in [("aux", 0.05, 0.0), ("aux", 0.0, 0.002), ("bias", 0.05, 0.0), ("none", 0.05, 0.0)]:
        model = make_model(balancer=balancer, balance_coef=coef, z_coef=z)
        loss, metrics, _ = model(idx, targets, return_metrics=True)
        expected = metrics["ce"] + (coef * metrics["balance"] if balancer == "aux" else 0) + z * metrics["z"]
        torch.testing.assert_close(loss.detach(), expected)


# -----------------------------------------------------------------------------
# MoE-layer selection and mixed models

def test_moe_layer_spec_parsing():
    config = GPTConfig(n_layer=6, ffn_type="moe")
    assert moe_layer_indices(config) == list(range(6))
    config.moe_layers = "every:2"
    assert moe_layer_indices(config) == [1, 3, 5]
    config.moe_layers = "0, 4"
    assert moe_layer_indices(config) == [0, 4]
    for bad in ("6", "-1", "every:0", ""):
        config.moe_layers = bad
        with pytest.raises(ValueError):
            moe_layer_indices(config)
    assert moe_layer_indices(GPTConfig(n_layer=6, ffn_type="swiglu", moe_layers="0")) == []


def test_mixed_model_forward_probe_and_optimizer():
    model = make_model(n_layer=4, moe_layers="every:2", balancer="bias")
    assert model.moe_layer_indices == [1, 3]
    assert [type(b.mlp).__name__ for b in model.transformer.h] == ["SwiGLUMLP", "MoEMLP", "SwiGLUMLP", "MoEMLP"]
    for block in model.transformer.h:
        if block.is_moe:
            block.mlp.expert_bias.copy_(torch.tensor([0.0, 0.4, 0.0, 0.0]))
    idx, targets = batch(10)
    loss, _, stats = model(idx, targets, return_metrics=True)
    assert len(stats) == 2 and torch.isfinite(loss)
    captured = []
    model(idx, routing_capture=captured)
    probe = routing_probe(model, idx)
    assert probe.shape == (2,) + tuple(idx.shape)
    # The probe records the routes actually used (with the bias), not raw argmax
    torch.testing.assert_close(probe, torch.stack([entry["choices"] for entry in captured]))
    optimizer = model.setup_optimizer(optimizer_name="muon", muon_variant="moonlight")
    loss.backward()
    optimizer.step()
    record = add_router_log_fields({}, accumulate_router_stats(None, stats, 1), None,
                                   layer_indices=model.moe_layer_indices, expert_bias=model.expert_bias_state())
    assert "train/moe/layer_3/bias_spread" in record and "train/moe/layer_1/max_vio" in record
    assert not any(key.startswith("train/moe/layer_0/") for key in record)


# -----------------------------------------------------------------------------
# Parameter-count conventions (spec decision 7)

def test_parameter_counts_follow_the_stated_conventions():
    d, L, E, V = 64, 3, 4, 50304  # GPT-2 vocab 50257 padded to a multiple of 64
    model = GPT(GPTConfig(sequence_len=16, vocab_size=50257, n_layer=L, n_head=1, n_kv_head=1,
                          n_embd=d, ffn_type="moe", n_experts=E))
    counts = model.parameter_counts()
    assert counts["embedding"] == V * d and counts["unembedding"] == V * d
    assert counts["router"] == L * E * d
    assert counts["active_non_embedding"] == L * (4 * d * d + 9 * d * d + E * d)
    assert counts["active_with_unembedding"] == counts["active_non_embedding"] + V * d == model.num_matmul_params()
    assert counts["active_with_embeddings"] == counts["active_non_embedding"] + 2 * V * d
    assert counts["total"] == 2 * V * d + L * (4 * d * d + E * 9 * d * d + E * d)


# -----------------------------------------------------------------------------
# Checkpoint compatibility

def test_old_moe_checkpoints_load_with_zero_bias():
    model = make_model()
    state = {k: v for k, v in model.state_dict().items() if not k.endswith("expert_bias")}
    restored = make_model(seed=1)
    for block in restored.transformer.h:
        block.mlp.expert_bias.fill_(0.5)
    patch_missing_expert_bias(restored, state)
    restored.load_state_dict(state, strict=True)
    assert all((block.mlp.expert_bias == 0).all() for block in restored.transformer.h)
    idx, _ = batch(11)
    torch.testing.assert_close(restored(idx), model(idx), rtol=0, atol=0)


def test_checkpoint_tokenizer_name_fallbacks():
    assert checkpoint_tokenizer_name({"tokenizer": "gpt2"}) == "gpt2"
    assert checkpoint_tokenizer_name({"user_config": {"tokenizer": "gpt2"}}) == "gpt2"
    assert checkpoint_tokenizer_name({"user_config": {}}) == "nanochat"
