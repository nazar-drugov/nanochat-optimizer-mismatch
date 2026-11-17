"""
Test the distributed optimizer on one GPU without a process group.
The AdamW half is checked against torch.optim.AdamW as a reference;
the Muon half is checked against torch.optim.Muon and behaviorally.

Requires a GPU (the fused kernels are compiled on first use, which
dominates the runtime of this file).

python -m pytest tests/test_optim.py -v
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="optimizer tests require CUDA")

if cuda_available:
    from nanochat.optim import DistributedOptimizer

DEVICE = "cuda"

# keep shapes small and reuse them across tests so the fused kernels compile once
ADAMW_SMALL_SHAPE = (48,)        # numel < 1024 path
ADAMW_LARGE_SHAPE = (1024, 4)    # numel >= 1024 path
MUON_WIDE_SHAPE = (32, 64)
MUON_TALL_SHAPE = (64, 32)


def make_params_and_groups(seed=1337, offset=0.0, targets=None):
    """The menagerie: small/large AdamW params, wide/tall Muon stacks."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    def rand(shape):
        base = torch.randn(shape, generator=gen, device=DEVICE) * 0.05
        return torch.nn.Parameter(base + offset)
    params = [rand(ADAMW_SMALL_SHAPE), rand(ADAMW_LARGE_SHAPE)]
    params += [rand(MUON_WIDE_SHAPE) for _ in range(3)]
    params += [rand(MUON_TALL_SHAPE) for _ in range(2)]
    groups = [
        dict(kind="adamw", params=params[0:1], lr=0.02, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", params=params[1:2], lr=0.02, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.0),
        dict(kind="muon", params=params[2:5], lr=0.02, momentum=0.95, ns_steps=5),
        dict(kind="muon", params=params[5:7], lr=0.02, momentum=0.95, ns_steps=5),
    ]
    return params, groups


def test_adamw_matches_torch_reference():
    """Our fused AdamW must agree with torch.optim.AdamW (both decoupled wd)."""
    hypers = dict(lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    p_ours = torch.nn.Parameter(torch.randn(64, 32, device=DEVICE))
    p_ref = torch.nn.Parameter(p_ours.detach().clone())
    opt_ours = DistributedOptimizer([dict(kind="adamw", params=[p_ours], **hypers)])
    opt_ref = torch.optim.AdamW([p_ref], **hypers)
    for step in range(10):
        grad = torch.randn(p_ours.shape, generator=torch.Generator(device=DEVICE).manual_seed(step), device=DEVICE)
        p_ours.grad = grad.clone()
        p_ref.grad = grad.clone()
        opt_ours.step()
        opt_ref.step()
    torch.testing.assert_close(p_ours, p_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [MUON_WIDE_SHAPE, MUON_TALL_SHAPE])
def test_muon_matches_torch_reference(shape):
    """Original Muon must match torch.optim.Muon across shapes and momentum changes."""
    gen = torch.Generator(device=DEVICE).manual_seed(2026)
    params_ours = [torch.nn.Parameter(torch.randn(shape, generator=gen, device=DEVICE)) for _ in range(3)]
    params_ref = [torch.nn.Parameter(p.detach().clone()) for p in params_ours]
    opt_ours = DistributedOptimizer([
        dict(kind="muon", params=params_ours, lr=0.02, momentum=0.95, ns_steps=5),
    ])
    opt_ref = torch.optim.Muon(
        params_ref,
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust_lr_fn="original",
    )

    for step, momentum in enumerate((0.85, 0.90, 0.95, 0.95)):
        opt_ours.param_groups[0]["momentum"] = momentum
        opt_ref.param_groups[0]["momentum"] = momentum
        grad_gen = torch.Generator(device=DEVICE).manual_seed(3000 + step)
        for p_ours, p_ref in zip(params_ours, params_ref):
            grad = torch.randn(shape, generator=grad_gen, device=DEVICE)
            p_ours.grad = grad.clone()
            p_ref.grad = grad.clone()
        opt_ours.step()
        opt_ref.step()

    for p_ours, p_ref in zip(params_ours, params_ref):
        torch.testing.assert_close(p_ours, p_ref, rtol=5e-3, atol=2e-3)

    state = opt_ours.state[params_ours[0]]
    assert set(state) == {"momentum_buffer"}
    for index, p_ref in enumerate(params_ref):
        torch.testing.assert_close(
            state["momentum_buffer"][index],
            opt_ref.state[p_ref]["momentum_buffer"],
            # Batched versus per-parameter float32 arithmetic differs by up to
            # two ulps near zero on L40S; parameter updates are checked above.
            rtol=2e-6,
            atol=2e-7,
        )


def test_determinism():
    """Two identical runs must produce bitwise identical parameters."""
    results = []
    for _ in range(2):
        params, groups = make_params_and_groups()
        opt = DistributedOptimizer(groups)
        for step in range(5):
            gen = torch.Generator(device=DEVICE).manual_seed(step)
            for p in params:
                p.grad = torch.randn(p.shape, generator=gen, device=DEVICE) * 0.01
            opt.step()
        results.append([p.detach().clone() for p in params])
    for pa, pb in zip(*results):
        assert torch.equal(pa, pb)


def test_convergence():
    """Optimizing distance-to-target must actually approach the target."""
    targets_params, _ = make_params_and_groups(seed=999)
    targets = [p.detach().clone() for p in targets_params]
    params, groups = make_params_and_groups(seed=999, offset=0.1) # start offset from the targets
    opt = DistributedOptimizer(groups)
    reference_params, _ = make_params_and_groups(seed=999, offset=0.1)
    reference_muon = torch.optim.Muon([
        dict(params=reference_params[2:5], lr=0.02, momentum=0.95,
             nesterov=True, ns_steps=5, adjust_lr_fn="original", weight_decay=0.0),
        dict(params=reference_params[5:7], lr=0.02, momentum=0.95,
             nesterov=True, ns_steps=5, adjust_lr_fn="original", weight_decay=0.0),
    ])
    def distances():
        return [(p.detach() - t).norm().item() for p, t in zip(params, targets)]
    initial = distances()
    for _ in range(50):
        for p, t in zip(params, targets):
            p.grad = 2 * (p.detach() - t) # gradient of ||p - t||^2
        for p, t in zip(reference_params[2:], targets[2:]):
            p.grad = 2 * (p.detach() - t)
        opt.step()
        reference_muon.step()
    final = distances()
    for i, (d0, d1) in enumerate(zip(initial, final)):
        if i < 2:
            assert d1 < 0.5 * d0, f"AdamW param {i} did not converge: {d0:.4f} -> {d1:.4f}"
        else:
            # Original Muon itself does not halve this Euclidean distance in
            # 50 steps. Require progress close to PyTorch's reference instead.
            reference_distance = (reference_params[i].detach() - targets[i]).norm().item()
            assert d1 < d0 and d1 <= 1.01 * reference_distance, (
                f"Muon param {i}: {d0:.4f} -> {d1:.4f}, reference {reference_distance:.4f}"
            )
        assert torch.isfinite(params[i]).all()


def test_muon_update_is_orthogonalized():
    """
    The very first Muon update (zero momentum state) of a full-rank gradient
    should be near semi-orthogonal after the original fixed-coefficient
    Newton-Schulz iteration.
    """
    p = torch.nn.Parameter(torch.zeros(MUON_WIDE_SHAPE, device=DEVICE))
    group = dict(kind="muon", params=[p], lr=1.0, momentum=0.0, ns_steps=5)
    opt = DistributedOptimizer([group])
    p.grad = torch.randn(p.shape, generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE)
    opt.step()
    # With lr=1, p_new = -update, so the update is just -p.
    svals = torch.linalg.svdvals(-p.detach().float())
    svals = svals / svals.mean()
    assert svals.max() / svals.min() < 4.0, f"update far from semi-orthogonal: {svals}"


def _distributed_muon_worker(rank, world_size, init_method):
    """Compare sharded two-rank Muon against PyTorch Muon on averaged gradients."""
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        shape = MUON_WIDE_SHAPE
        init_gen = torch.Generator(device=device).manual_seed(4040)
        params_ours = [torch.nn.Parameter(torch.randn(shape, generator=init_gen, device=device)) for _ in range(3)]
        params_ref = [torch.nn.Parameter(p.detach().clone()) for p in params_ours]
        opt_ours = DistributedOptimizer([
            dict(kind="muon", params=params_ours, lr=0.02, momentum=0.95, ns_steps=5),
        ])
        opt_ref = torch.optim.Muon(
            params_ref,
            lr=0.02,
            weight_decay=0.0,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adjust_lr_fn="original",
        )

        for step, momentum in enumerate((0.85, 0.90, 0.95)):
            local_gen = torch.Generator(device=device).manual_seed(5000 + step * world_size + rank)
            local_grads = [torch.randn(shape, generator=local_gen, device=device) for _ in params_ours]

            average_grads = [torch.zeros(shape, device=device) for _ in params_ref]
            for source_rank in range(world_size):
                source_gen = torch.Generator(device=device).manual_seed(5000 + step * world_size + source_rank)
                for average_grad in average_grads:
                    average_grad.add_(
                        torch.randn(shape, generator=source_gen, device=device),
                        alpha=1 / world_size,
                    )

            opt_ours.param_groups[0]["momentum"] = momentum
            opt_ref.param_groups[0]["momentum"] = momentum
            for p_ours, p_ref, local_grad, average_grad in zip(params_ours, params_ref, local_grads, average_grads):
                p_ours.grad = local_grad
                p_ref.grad = average_grad
            opt_ours.step()
            opt_ref.step()

        for p_ours, p_ref in zip(params_ours, params_ref):
            replicas = [torch.empty_like(p_ours) for _ in range(world_size)]
            dist.all_gather(replicas, p_ours)
            assert all(torch.equal(replicas[0], replica) for replica in replicas[1:])
            torch.testing.assert_close(p_ours, p_ref, rtol=5e-3, atol=2e-3)

        chunk_size = (len(params_ours) + world_size - 1) // world_size
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params_ours) - start_idx))
        momentum_buffer = opt_ours.state[params_ours[0]]["momentum_buffer"]
        for local_index in range(num_owned):
            ref_param = params_ref[start_idx + local_index]
            torch.testing.assert_close(
                momentum_buffer[local_index],
                opt_ref.state[ref_param]["momentum_buffer"],
                rtol=1e-6,
                atol=1e-6,
            )
    finally:
        dist.destroy_process_group()


def test_two_rank_muon_matches_torch_reference(tmp_path):
    """Two ranks with different local gradients must match averaged reference Muon."""
    if torch.cuda.device_count() < 2:
        pytest.skip("distributed Muon parity test requires at least two CUDA devices")
    init_method = f"file://{tmp_path / 'muon_dist_init'}"
    mp.spawn(_distributed_muon_worker, args=(2, init_method), nprocs=2, join=True)
