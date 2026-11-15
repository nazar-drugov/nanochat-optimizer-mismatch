"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
The same class handles both single GPU and distributed training: when there is no
multi-rank process group, the communication ops are simply skipped and every rank
(i.e. the only rank) owns all of the parameters.

Adapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    step_t: Tensor,         # () - 0-D CPU tensor, step count
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """
    # Some params (such as wte) are stored in bf16, so do the math in fp32 and
    # cast back at the end. MPS errors on mixed-dtype ops (CUDA promotes them), and
    # scalar arithmetic like 1 - beta2 loses all precision in bf16. compile fuses the casts.
    p32 = p.float()
    exp_avg32 = exp_avg.float()
    exp_avg_sq32 = exp_avg_sq.float()
    grad32 = grad.float()
    # Weight decay (decoupled, applied before the update)
    p32.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg32.lerp_(grad32, 1 - beta1_t)
    exp_avg_sq32.lerp_(grad32.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq32 / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p32.add_(exp_avg32 / denom, alpha=-step_size)
    # Write back (no-ops in the common case where everything is already fp32)
    p.copy_(p32)
    exp_avg.copy_(exp_avg32)
    exp_avg_sq.copy_(exp_avg_sq32)

# -----------------------------------------------------------------------------
"""
Original Muon optimizer from Keller et al.
https://kellerjordan.github.io/posts/muon/

Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.
"""

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,   # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,  # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor, # (12, 768, 3072) - momentum buffer
    momentum_t: Tensor,      # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,            # () - 0-D CPU tensor, aspect-ratio-adjusted learning rate
    ns_steps: int,           # 5 - number of Newton-Schulz iterations
) -> None:
    """
    Original Muon update: Nesterov momentum, fixed-coefficient Newton-Schulz
    orthogonalization, then the parameter update. The aspect-ratio adjustment is
    included in lr_t by the caller.
    """
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Original Muon always performs the orthogonalization in bfloat16.
    X = g.bfloat16()
    if g.size(-2) > g.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if g.size(-2) > g.size(-1):
        X = X.mT
    stacked_params.sub_(lr_t.to(stacked_params.dtype) * X.to(stacked_params.dtype))

# -----------------------------------------------------------------------------

class DistributedOptimizer(torch.optim.Optimizer):
    """
    Distributed optimizer supporting AdamW and original Muon parameter groups.

    AdamW - Fused AdamW optimizer step.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    The same class covers single GPU and distributed training. In the distributed setting
    (a multi-rank process group is initialized), gradients are synchronized here in the
    optimizer (nanochat does not use DDP) and optimizer states are sharded across ranks
    (ZeRO-2 style). On a single rank, all communication is skipped and the rank owns all
    parameters, so the sharded code paths degenerate to plain full-tensor updates.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - Batch small tensors into single comm ops where possible

    Communication Pattern (3-phase async):
    We use a 3-phase structure to maximize overlap between communication and compute:

        Phase 1: Launch all async reduce ops
            - Kick off all reduce_scatter/all_reduce operations
            - Don't wait - let them run in background while we continue

        Phase 2: Wait for reduces, compute updates, launch gathers
            - For each group: wait for its reduce, compute the update, launch gather
            - By processing groups in order, earlier gathers run while later computes happen

        Phase 3: Wait for gathers, copy back
            - Wait for all gathers to complete
            - Copy updated params back to original tensors (Muon only)

    AdamW Communication (ZeRO-2 style):
    - Small params (<1024 elements): all_reduce gradients, update full param on each rank.
      Optimizer state is replicated but these params are tiny (scalars, biases).
    - Large params: reduce_scatter gradients so each rank gets 1/N of the grad, update
      only that slice, then all_gather the updated slices. Optimizer state (exp_avg,
      exp_avg_sq) is sharded - each rank only stores state for its slice.
      Requires param.shape[0] divisible by world_size.

    Muon Communication (stacked + chunked):
    - All params in a Muon group must have the same shape (caller's responsibility).
    - Stack all K params into a single (K, *shape) tensor for efficient comm.
    - Divide K params across N ranks: each rank "owns" ceil(K/N) params.
    - reduce_scatter the stacked grads so each rank gets its chunk.
    - Each rank computes Muon update only for params it owns.
    - all_gather the updated params back to all ranks.
    - Optimizer state (momentum_buffer) is sharded by chunk.
    - Padding: if K doesn't divide evenly, we zero-pad to (ceil(K/N) * N) for comm,
      then ignore the padding when copying back.

    Buffer Reuse:
    - For Muon, we allocate stacked_grads for reduce_scatter input, then reuse the
      same buffer as the output for all_gather (stacked_params). This saves memory
      since we don't need both buffers simultaneously.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if world_size == 1:
                # Single rank: no communication, update the full param in place
                param_infos[p] = dict(future=None, grad_slice=grad, is_small=True)
            elif p.numel() < 1024:
                # Small params: all_reduce (no scatter/gather needed)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                # Large params: reduce_scatter
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for Muon group. Returns info dict."""
        params = group['params']
        if world_size == 1:
            # Single rank: this rank owns all params, the stacked grads are the "chunk"
            grad_chunk = torch.stack([p.grad for p in params])
            return dict(future=None, grad_chunk=grad_chunk, stacked_grads=None, chunk_size=len(params))
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        # Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            if pinfo['future'] is not None:
                pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1

            # Fill 0-D tensors and run fused kernel
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            # Large params need all_gather
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        if info['future'] is not None:
            info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)

        stacked_owned = None
        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, group["ns_steps"],
            )

        if info['stacked_grads'] is None:
            # Single rank: no gather needed, the updated stack maps directly onto the params
            gather_list.append(dict(future=None, stacked_params=stacked_owned, params=params))
            return

        # Build the input buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        if num_owned > 0:
            updated_params[:num_owned].copy_(stacked_owned)
        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            if info["future"] is not None:
                info["future"].wait()
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        # On a single rank (no multi-rank process group), all communication is skipped
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Phase 1: launch all async reduce ops
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 2: wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)
