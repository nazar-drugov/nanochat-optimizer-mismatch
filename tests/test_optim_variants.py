"""
Optimizer behaviour needed for the pretraining runs, on CPU (or CUDA if present):

- Muon "moonlight" (Moonlight scaling + decoupled weight decay) and "keller" (original
  scaling) match torch.optim.Muon with adjust_lr_fn="match_rms_adamw" / "original".
- Global gradient clipping matches torch.nn.utils.clip_grad_norm_ and reports the
  pre-clip norm.
- The two-rank path (reduce-scatter / all-gather, sharded state, clipping across
  ranks) matches a single rank that sees the averaged gradient (gloo on CPU).

python -m pytest tests/test_optim_variants.py -v
"""

import math
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanochat.optim import DistributedOptimizer, MUON_VARIANTS, muon_lr_scale

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIDE, TALL = (32, 64), (64, 32)


def assert_updates_close(after, reference_after, before, rel=2e-2):
    """Compare parameter updates by relative Frobenius error. Muon orthogonalizes in
    bf16, so last-bit differences in its input (e.g. a different reduction order)
    move individual entries by ~1e-4; the update as a whole must still agree."""
    for a, r, b in zip(after, reference_after, before):
        update, reference_update = (a - b).float(), (r - b).float()
        error = (update - reference_update).norm() / reference_update.norm().clamp_min(1e-12)
        assert error.item() < rel, f"relative update error {error.item():.2e} >= {rel}"


def make_params(shape, n, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(shape, generator=gen, device=DEVICE)) for _ in range(n)]


def test_muon_lr_scale_values():
    assert muon_lr_scale("moonlight", (3 * 640, 640)) == pytest.approx(0.2 * math.sqrt(1920))
    assert muon_lr_scale("moonlight", (640, 1920)) == pytest.approx(0.2 * math.sqrt(1920))
    assert muon_lr_scale("keller", (1920, 640)) == pytest.approx(math.sqrt(3))
    assert muon_lr_scale("keller", (640, 1920)) == 1.0
    assert MUON_VARIANTS == ("moonlight", "keller")
    with pytest.raises(ValueError):
        muon_lr_scale("nanochat", (4, 4))


def test_unknown_muon_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown Muon variant"):
        DistributedOptimizer([dict(kind="muon", params=make_params(WIDE, 1, 0), lr=0.01,
                                   momentum=0.95, ns_steps=5, variant="nanochat")])


@pytest.mark.parametrize("shape", [WIDE, TALL])
@pytest.mark.parametrize("variant, adjust_lr_fn", [("moonlight", "match_rms_adamw"), ("keller", "original")])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_muon_variant_matches_torch_reference(shape, variant, adjust_lr_fn, weight_decay):
    lr = 4e-3 if variant == "moonlight" else 0.02
    ours = make_params(shape, 3, seed=2026)
    ref = [torch.nn.Parameter(p.detach().clone()) for p in ours]
    opt_ours = DistributedOptimizer([dict(kind="muon", params=ours, lr=lr, momentum=0.95, ns_steps=5,
                                          variant=variant, weight_decay=weight_decay)])
    opt_ref = torch.optim.Muon(ref, lr=lr, weight_decay=weight_decay, momentum=0.95, nesterov=True,
                               ns_steps=5, adjust_lr_fn=adjust_lr_fn)
    for step in range(5):
        gen = torch.Generator(device=DEVICE).manual_seed(3000 + step)
        for p_ours, p_ref in zip(ours, ref):
            grad = torch.randn(shape, generator=gen, device=DEVICE)
            p_ours.grad, p_ref.grad = grad.clone(), grad.clone()
        opt_ours.step()
        opt_ref.step()
    # Tolerance: our kernel is Keller Jordan's reference Newton-Schulz
    # (B = b*A + c*A@A; X = a*X + B@X as separate bf16 ops); torch fuses each step
    # into addmm. Measured 2026-11-23: one NS pass differs by 2.6-4.3% between the
    # two orderings (torch-vs-torch with a 1e-6 input perturbation: ~0.3%), and five
    # optimizer steps differ by 1.3-2.1%. The test still discriminates: a wrong
    # weight decay changes these updates by 35-50%, a wrong shape scaling by >2x
    # (and the exact moonlight/keller ratio is tested below).
    assert_updates_close([p.detach() for p in ours], [p.detach() for p in ref],
                         [p.detach() for p in make_params(shape, 3, seed=2026)], rel=5e-2)


def test_moonlight_and_keller_differ_only_by_per_shape_scale():
    """Same direction, different step size: one step from zero momentum."""
    results = {}
    for variant in ("moonlight", "keller"):
        params = make_params(TALL, 1, seed=7)
        p0 = params[0].detach().clone()
        params[0].grad = torch.randn(TALL, generator=torch.Generator(device=DEVICE).manual_seed(1), device=DEVICE)
        DistributedOptimizer([dict(kind="muon", params=params, lr=1.0, momentum=0.0, ns_steps=5,
                                   variant=variant, weight_decay=0.0)]).step()
        results[variant] = params[0].detach() - p0
    ratio = muon_lr_scale("moonlight", TALL) / muon_lr_scale("keller", TALL)
    torch.testing.assert_close(results["moonlight"], ratio * results["keller"], rtol=1e-5, atol=1e-6)


def make_mixed_groups(seed, max_grad_norm=None):
    """AdamW small (<1024) and large tensors plus two Muon shape groups."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    def rand(shape):
        return torch.nn.Parameter(torch.randn(shape, generator=gen, device=DEVICE) * 0.05)
    params = [rand((48,)), rand((1024, 4))] + [rand(WIDE) for _ in range(3)] + [rand(TALL) for _ in range(2)]
    groups = [
        dict(kind="adamw", params=params[0:1], lr=0.02, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        dict(kind="adamw", params=params[1:2], lr=0.02, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1),
        dict(kind="muon", params=params[2:5], lr=4e-3, momentum=0.95, ns_steps=5, variant="moonlight", weight_decay=0.1),
        dict(kind="muon", params=params[5:7], lr=0.02, momentum=0.95, ns_steps=5, variant="keller", weight_decay=0.0),
    ]
    return params, DistributedOptimizer(groups, max_grad_norm=max_grad_norm, track_grad_norm=True)


def set_grads(params, seed, scale):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=gen, device=DEVICE) * scale


@pytest.mark.parametrize("scale, should_clip", [(10.0, True), (1e-3, False)])
def test_global_clipping_matches_torch_clip_grad_norm(scale, should_clip):
    initial = [p.detach().clone() for p in make_mixed_groups(seed=11)[0]]
    clipped_params, clipped_opt = make_mixed_groups(seed=11, max_grad_norm=1.0)
    ref_params, ref_opt = make_mixed_groups(seed=11, max_grad_norm=None)
    for step in range(3):
        set_grads(clipped_params, seed=100 + step, scale=scale)
        set_grads(ref_params, seed=100 + step, scale=scale)
        expected_norm = torch.nn.utils.clip_grad_norm_(ref_params, max_norm=1.0)
        clipped_opt.step()
        ref_opt.step()
        torch.testing.assert_close(clipped_opt.last_grad_norm, expected_norm.to(clipped_opt.last_grad_norm.dtype), rtol=1e-5, atol=1e-6)
        assert (clipped_opt.last_clip_coef.item() < 1.0) == should_clip
        # Unclipped optimizer still reports the norm of what it received
        assert ref_opt.last_clip_coef.item() == 1.0
    # AdamW tensors agree to float precision; Muon updates up to bf16 noise
    for p_clip, p_ref in zip(clipped_params[:2], ref_params[:2]):
        torch.testing.assert_close(p_clip, p_ref, rtol=1e-5, atol=1e-6)
    # torch sums the norm in a different order (last-bit coef difference) and bf16
    # Newton-Schulz amplifies it, so Muon tensors get the relative-update check
    assert_updates_close([p.detach() for p in clipped_params[2:]], [p.detach() for p in ref_params[2:]], initial[2:], rel=1e-2)


def test_no_norm_computed_unless_requested():
    params, _ = make_mixed_groups(seed=3)
    opt = DistributedOptimizer([dict(kind="adamw", params=params[:2], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)])
    set_grads(params[:2], seed=1, scale=1.0)
    opt.step()
    assert opt.last_grad_norm is None
    with pytest.raises(ValueError):
        DistributedOptimizer([dict(kind="adamw", params=params[:1], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)], max_grad_norm=0.0)


# -----------------------------------------------------------------------------
# Two ranks on gloo: each rank gets a different local gradient; the result must
# equal a single-rank optimizer that receives their average.

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _two_rank_worker(rank, port, result_path, max_grad_norm):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        params, opt = make_mixed_groups(seed=5, max_grad_norm=max_grad_norm)
        norms = []
        for step in range(3):
            set_grads(params, seed=1000 * (rank + 1) + step, scale=5.0)
            opt.step()
            norms.append(opt.last_grad_norm.item())
        if rank == 0:
            torch.save({"params": [p.detach().cpu() for p in params], "norms": norms}, result_path)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(DEVICE != "cpu" or not dist.is_gloo_available(), reason="gloo CPU test")
@pytest.mark.parametrize("max_grad_norm", [None, 1.0])
def test_two_rank_step_matches_single_rank_average(tmp_path, max_grad_norm):
    result_path = tmp_path / "rank0.pt"
    mp.spawn(_two_rank_worker, args=(_free_port(), str(result_path), max_grad_norm), nprocs=2, join=True)
    result = torch.load(result_path)

    initial = [p.detach().clone().cpu() for p in make_mixed_groups(seed=5)[0]]
    params, opt = make_mixed_groups(seed=5, max_grad_norm=max_grad_norm)
    for step in range(3):
        set_grads(params, seed=1000 + step, scale=5.0)
        first = [p.grad.clone() for p in params]
        set_grads(params, seed=2000 + step, scale=5.0)
        for p, g in zip(params, first):
            p.grad = (p.grad + g) / 2
        opt.step()
        assert result["norms"][step] == pytest.approx(opt.last_grad_norm.item(), rel=1e-6)
    single = [p.detach().cpu() for p in params]
    # AdamW (small all-reduced tensor and reduce-scattered shards) must match the
    # single-rank average to float precision. This is what catches the regression
    # where gloo re-copied reduce_scatter outputs on each wait() and undid the
    # clipping on sharded tensors (fixed by marking waited work complete).
    for p_dist, p_single in zip(result["params"][:2], single[:2]):
        torch.testing.assert_close(p_dist, p_single, rtol=1e-6, atol=1e-7)
    # Muon: each rank orthogonalizes a stack of 2 or 1 matrices instead of 3, and
    # compiled bf16 batched matmuls round differently per batch shape (the first
    # step is typically bit-exact; later steps drift through momentum), and the clip
    # coefficient can differ in its last bit. Check the update as a whole.
    assert_updates_close(result["params"][2:], single[2:], initial[2:], rel=1e-2)
