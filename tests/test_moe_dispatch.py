"""
MoE dispatch: the grouped-GEMM path must compute exactly what the reference
per-expert loop computes (same routes, gates, outputs and gradients up to bf16
GEMM rounding), and must not break torch.compile graphs.

python -m pytest tests/test_moe_dispatch.py -v
"""

import pytest
import torch

from nanochat.gpt import GPT, GROUPED_MM_ALIGN, GPTConfig, MoEMLP, grouped_segment_ends, resolve_moe_dispatch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
needs_grouped_mm = pytest.mark.skipif(not hasattr(torch, "_grouped_mm"), reason="torch._grouped_mm unavailable")


def make_mlp(n_experts, balancer="aux", dispatch="loop", n_embd=64, device="cpu", seed=0):
    config = GPTConfig(n_embd=n_embd, n_head=1, n_kv_head=1, n_layer=1, ffn_type="moe", n_experts=n_experts,
                       balancer=balancer, moe_dispatch=dispatch)
    mlp = MoEMLP(config).to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in mlp.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
    return mlp


def run(mlp, x, forced=None):
    x = x.detach().clone().requires_grad_(True)
    mlp.zero_grad(set_to_none=True)
    out, aux, stats = mlp(x, forced_idx=forced)
    # A loss that depends on every output element and on the aux losses
    weight = torch.linspace(-1, 1, out.numel(), device=out.device).view_as(out)
    ((out.float() * weight).sum() + aux["balance"] + aux["z"]).backward()
    grads = {n: p.grad.detach().clone() for n, p in mlp.named_parameters() if p.grad is not None}
    return out.detach(), x.grad.detach(), grads, stats


def rel(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-12)).item()


def assert_equivalent(mlp, x, forced=None, tol=2e-3):
    mlp.config.moe_dispatch = "loop"
    ref = run(mlp, x, forced)
    mlp.config.moe_dispatch = "grouped"
    new = run(mlp, x, forced)
    assert torch.equal(ref[3]["expert_counts"], new[3]["expert_counts"])  # identical routing
    assert rel(new[0], ref[0]) < tol, "outputs differ"
    assert rel(new[1], ref[1]) < tol, "input gradients differ"
    assert ref[2].keys() == new[2].keys()
    for name in ref[2]:
        assert rel(new[2][name], ref[2][name]) < tol, f"gradient of {name} differs"
    return ref, new


# ---------------------------------------------------------------------------
# Which dispatch runs

def test_dispatch_resolution():
    cfg = GPTConfig(n_embd=64, ffn_type="moe", moe_dispatch="auto")
    assert resolve_moe_dispatch(cfg, "cpu", torch.float32) == "loop"
    assert resolve_moe_dispatch(cfg, "cpu", torch.bfloat16) == "loop"  # auto only picks grouped on CUDA
    assert resolve_moe_dispatch(cfg, "cuda", torch.float32) == "loop"
    if hasattr(torch, "_grouped_mm"):
        assert resolve_moe_dispatch(cfg, "cuda", torch.bfloat16) == "grouped"
    cfg.moe_dispatch = "loop"
    assert resolve_moe_dispatch(cfg, "cuda", torch.bfloat16) == "loop"
    cfg.moe_dispatch = "grouped"
    with pytest.raises(RuntimeError, match="bf16"):
        resolve_moe_dispatch(cfg, "cuda", torch.float32)
    with pytest.raises(ValueError, match="moe_dispatch"):
        GPT(GPTConfig(n_embd=64, n_head=1, n_kv_head=1, n_layer=1, ffn_type="moe", moe_dispatch="fast"))


# ---------------------------------------------------------------------------
# Grouped == loop

def test_grouped_segments_are_aligned_never_empty_and_fit_the_buffer():
    """The native B200 grouped GEMM leaves the weight gradient of an empty group
    uninitialized, so every expert's segment must have at least one aligned block."""
    A = GROUPED_MM_ALIGN
    for counts in ([17, 0, 33, 1, 15, 0, 33, 1], [100, 0, 0, 0], [0, 0, 0, 1], [16, 32, 48, 0], [64]):
        counts = torch.tensor(counts)
        ends, padded = grouped_segment_ends(counts, A)
        starts = ends - padded
        assert (padded >= A).all() and (padded % A == 0).all() and (starts % A == 0).all()
        assert (padded >= counts).all() and ((padded - counts < A) | (counts == 0)).all()
        assert ends[-1] <= counts.sum() + len(counts) * A  # fits the N + E*A buffer


@needs_grouped_mm
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_experts", [1, 4, 8])
def test_grouped_matches_loop_on_natural_routes(device, n_experts):
    mlp = make_mlp(n_experts, device=device)
    x = torch.randn(3, 37, 64, generator=torch.Generator().manual_seed(1)).to(device, torch.bfloat16)  # 111 tokens
    ref, _ = assert_equivalent(mlp, x)
    if n_experts > 1:
        assert (ref[3]["expert_counts"] > 0).sum() > 1  # the test really spreads tokens


@needs_grouped_mm
@pytest.mark.parametrize("device", DEVICES)
def test_grouped_matches_loop_with_empty_and_unaligned_experts(device):
    """Forced routes: experts 1 and 5 get no tokens, the others counts that are not
    multiples of the 16-row alignment, and expert 7 a single token."""
    mlp = make_mlp(8, device=device)
    B, T = 2, 50
    routes = torch.tensor([0] * 17 + [2] * 33 + [3] * 1 + [4] * 15 + [6] * 33 + [7] * 1)
    assert routes.numel() == B * T
    forced = routes[torch.randperm(B * T, generator=torch.Generator().manual_seed(2))].view(B, T).to(device)
    x = torch.randn(B, T, 64, generator=torch.Generator().manual_seed(3)).to(device, torch.bfloat16)
    ref, new = assert_equivalent(mlp, x, forced=forced)
    counts = new[3]["expert_counts"].cpu()
    assert counts.tolist() == [17, 0, 33, 1, 15, 0, 33, 1]
    for e in (1, 5):  # experts without tokens get zero gradient on both paths
        for proj in ("up", "gate", "down"):
            assert new[2][f"experts.{e}.{proj}.weight"].abs().sum() == 0


@needs_grouped_mm
@pytest.mark.parametrize("device", DEVICES)
def test_grouped_matches_loop_with_bias_balancer_and_single_token(device):
    mlp = make_mlp(4, balancer="bias", device=device)
    with torch.no_grad():
        mlp.expert_bias.copy_(torch.tensor([0.0, 0.3, -0.3, 0.1]))
    assert_equivalent(mlp, torch.randn(2, 29, 64).to(device, torch.bfloat16))
    assert_equivalent(mlp, torch.randn(1, 1, 64).to(device, torch.bfloat16))  # one decoding step


@needs_grouped_mm
def test_grouped_dispatch_has_no_graph_breaks_and_loop_does():
    mlp = make_mlp(4, dispatch="grouped")
    x = torch.randn(2, 32, 64).to(torch.bfloat16)
    explain = torch._dynamo.explain(lambda t: mlp(t)[0])
    assert explain(x).graph_break_count == 0
    mlp.config.moe_dispatch = "loop"
    torch._dynamo.reset()
    assert explain(x).graph_break_count >= 1  # why the loop is slow under torch.compile


# ---------------------------------------------------------------------------
# Full model on CUDA (bf16 compute)

needs_cuda_grouped = pytest.mark.skipif(not (torch.cuda.is_available() and hasattr(torch, "_grouped_mm")),
                                        reason="needs CUDA and torch._grouped_mm")


def cuda_model(dispatch, n_experts=4):
    config = GPTConfig(sequence_len=128, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=128,
                       ffn_type="moe", n_experts=n_experts, moe_dispatch=dispatch)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cuda")
    model.init_weights(seed=3)
    return model


@needs_cuda_grouped
def test_full_model_grouped_matches_loop_on_cuda():
    loop, grouped = cuda_model("loop"), cuda_model("grouped")
    grouped.load_state_dict(loop.state_dict())
    gen = torch.Generator(device="cuda").manual_seed(0)
    idx = torch.randint(0, 512, (4, 128), device="cuda", generator=gen)
    tgt = torch.randint(0, 512, (4, 128), device="cuda", generator=gen)
    l_loop, l_grouped = loop(idx, tgt), grouped(idx, tgt)
    l_loop.backward(); l_grouped.backward()
    assert abs(l_loop.item() - l_grouped.item()) < 1e-3
    for (name, p), (_, q) in zip(loop.named_parameters(), grouped.named_parameters()):
        if p.grad is not None:
            assert rel(q.grad, p.grad) < 2e-3, name
    # Bitwise deterministic across calls (exact resume relies on it)
    grads = [p.grad.clone() for p in grouped.parameters() if p.grad is not None]
    grouped.zero_grad(set_to_none=True)
    grouped(idx, tgt).backward()
    for g, p in zip(grads, [p for p in grouped.parameters() if p.grad is not None]):
        assert torch.equal(g, p.grad)


@needs_cuda_grouped
def test_compiled_training_step_is_one_graph_on_cuda():
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()
    model = cuda_model("auto", n_experts=8)
    compiled = torch.compile(model, dynamic=False)
    gen = torch.Generator(device="cuda").manual_seed(0)
    tgt = torch.randint(0, 512, (4, 128), device="cuda", generator=gen)
    for _ in range(4):  # different routes every step must not recompile
        compiled(torch.randint(0, 512, (4, 128), device="cuda", generator=gen), tgt).backward()
    assert sum(counters["graph_break"].values()) == 0
    assert counters["stats"]["unique_graphs"] == 1
