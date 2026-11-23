"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- feed-forward: relu^2 MLP (ffn_type="dense"), SwiGLU MLP ("swiglu"), or a
  top-1 Switch-style MoE of SwiGLU experts ("moe")
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import DistributedOptimizer

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # dense:  nanochat ReLU^2 MLP, 4x width (legacy dense model)
    # swiglu: dense SwiGLU MLP, 3x width, identical to one MoE expert
    # moe:    top-1 MoE of SwiGLU experts in the layers named by moe_layers;
    #         the other layers use a dense SwiGLU MLP
    ffn_type: str = "dense"
    n_experts: int = 4
    moe_layers: str = "all" # "all" | "every:K" (layers K-1, 2K-1, ...) | "0,2,5"
    # Load balancing (MoE only)
    # aux:  Switch auxiliary loss balance_coef * E * sum_i f_i P_i
    # bias: auxiliary-loss-free bias on selection only (Wang et al. 2024)
    # none: no balancing
    balancer: str = "aux"
    balance_coef: float = 0.01
    z_coef: float = 0.001
    bias_update_rate: float = 0.001
    # How tokens reach their expert (same routing and maths either way):
    # grouped: sort tokens by expert and run all experts in one torch._grouped_mm per
    #          projection; static shapes, no CPU sync, so the model compiles as one graph
    # loop:    boolean-mask gather per expert (data-dependent shapes; breaks torch.compile)
    # auto:    grouped for bf16 activations on CUDA, loop otherwise
    moe_dispatch: str = "auto"


FFN_TYPES = ("dense", "swiglu", "moe")
BALANCERS = ("aux", "bias", "none")
MOE_DISPATCH = ("auto", "grouped", "loop")
GROUPED_MM_ALIGN = 16  # each expert's segment starts at a multiple of this many rows


def grouped_dispatch_supported(n_embd, dtype):
    """torch._grouped_mm needs bf16 operands and 16-byte aligned rows."""
    return hasattr(torch, "_grouped_mm") and dtype == torch.bfloat16 and n_embd % 8 == 0


def resolve_moe_dispatch(config, device_type, dtype):
    """The dispatch an MoE layer will use for activations of this device type and dtype."""
    if config.moe_dispatch not in MOE_DISPATCH:
        raise ValueError(f"Unknown moe_dispatch {config.moe_dispatch!r} (expected one of {MOE_DISPATCH})")
    supported = grouped_dispatch_supported(config.n_embd, dtype)
    if config.moe_dispatch == "grouped":
        if not supported:
            raise RuntimeError(f"moe_dispatch='grouped' needs torch._grouped_mm, bf16 activations and n_embd % 8 == 0 "
                               f"(got dtype {dtype}, n_embd {config.n_embd})")
        return "grouped"
    if config.moe_dispatch == "auto" and supported and device_type == "cuda":
        return "grouped"
    return "loop"


def moe_layer_indices(config):
    """Indices of the blocks whose feed-forward layer is an MoE layer."""
    if config.ffn_type != "moe":
        return []
    n_layer = config.n_layer
    spec = config.moe_layers.strip()
    if spec == "all":
        return list(range(n_layer))
    if spec.startswith("every:"):
        k = int(spec.split(":", 1)[1])
        if k <= 0:
            raise ValueError(f"moe_layers every:K needs K >= 1, got {spec!r}")
        return [i for i in range(n_layer) if (i + 1) % k == 0]
    indices = sorted({int(part) for part in spec.split(",") if part.strip()})
    if not indices or indices[0] < 0 or indices[-1] >= n_layer:
        raise ValueError(f"moe_layers {spec!r} must name layers in [0, {n_layer})")
    return indices


def make_gpt_config(
    *,
    depth,
    aspect_ratio,
    head_dim,
    max_seq_len,
    vocab_size,
    ffn_type,
    n_experts,
    moe_layers="all",
    balancer="aux",
    balance_coef=0.01,
    z_coef=0.001,
    bias_update_rate=0.001,
    moe_dispatch="auto",
):
    """Build the GPT configuration shared by training and planning scripts."""
    # Nudge the model dimension up to a multiple of head_dim so every attention
    # head has exactly the requested size.
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return GPTConfig(
        ffn_type=ffn_type,
        n_experts=n_experts,
        moe_layers=moe_layers,
        balancer=balancer,
        balance_coef=balance_coef,
        z_coef=z_coef,
        bias_update_rate=bias_update_rate,
        moe_dispatch=moe_dispatch,
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
    )

def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 or SDPA fallback), always using full causal attention
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=(-1, 0),
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class DenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 3 * config.n_embd
        self.up = Linear(config.n_embd, hidden, bias=False)
        self.gate = Linear(config.n_embd, hidden, bias=False)
        self.down = Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class MoEMLP(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.router = Linear(config.n_embd, config.n_experts, bias=False, dtype=torch.float32)

        # Each expert has its own expansion and projection layer
        self.experts = nn.ModuleList(
                SwiGLUMLP(config) for _ in range(config.n_experts)
            )

        # Selection-only balancing bias (balancer="bias"). It is a buffer, not a
        # parameter: no optimizer touches it; GPT.update_expert_bias() does.
        # It is persistent so checkpoints carry the balancer state.
        self.register_buffer("expert_bias", torch.zeros(config.n_experts, dtype=torch.float32))

    def forward(self, x, forced_idx=None, capture=None):
        """Top-1 Switch-style MoE.

        forced_idx: optional (batch, seq_len) expert indices that replace this
            layer's own selection (route replay). The gate value is still the
            current router's own softmax probability for the forced expert.
        capture: optional list; if given, a dict with this layer's router
            inputs, logits and the routes actually used is appended to it.
        """
        # x is (batch, seq_len, n_embd)
        total_num_tokens = x.shape[0] * x.shape[1]
        total_num_experts = self.config.n_experts

        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = self.router(x.float()) # (batch, seq_len, n_experts)
            router_probs = F.softmax(router_logits, dim=-1) # (batch, seq_len, n_experts)
        if forced_idx is not None:
            if forced_idx.shape != router_probs.shape[:2]:
                raise ValueError(f"forced_idx has shape {tuple(forced_idx.shape)}, expected {tuple(router_probs.shape[:2])}")
            router_choices = forced_idx.to(device=x.device, dtype=torch.long)
        elif self.config.balancer == "bias":
            # The bias only changes which expert is selected, never the gate value.
            router_choices = torch.argmax(router_probs + self.expert_bias, dim=-1)
        else:
            router_choices = torch.argmax(router_probs, dim=-1) # (batch, seq_len)
        if capture is not None:
            capture.append({
                "layer": self.layer_idx,
                "inputs": x.detach(),
                "logits": router_logits.detach(),
                "choices": router_choices.detach(),
            })

        # Entropy of each token's distribution over experts.
        router_entropy = -(router_probs * F.log_softmax(router_logits, dim=-1)).sum(dim=-1) # (batch, seq_len)
        mean_router_entropy = router_entropy.mean()  # one scalar for this MoE layer

        mean_expert_probs = router_probs.mean(dim=(0, 1)) # (n_experts)
        # Integer token counts per expert (the load). one_hot keeps shapes static.
        expert_counts = F.one_hot(router_choices, total_num_experts).sum(dim=(0, 1))
        expert_shares = expert_counts.float() / total_num_tokens

        if resolve_moe_dispatch(self.config, x.device.type, x.dtype) == "grouped":
            output = self._dispatch_grouped(x, router_probs, router_choices, expert_counts)
        else:
            output = self._dispatch_loop(x, router_probs, router_choices)

        # The raw z-loss also serves as a routing diagnostic.
        L_z = torch.logsumexp(router_logits, dim=-1).square().mean()
        # Switch balance loss: f_i (expert_shares) carries no gradient; P_i does.
        L_balance = total_num_experts * (expert_shares * mean_expert_probs).sum()
        router_stats = {
            "expert_shares": expert_shares.detach(),
            "expert_counts": expert_counts.detach(),
            "router_entropy": mean_router_entropy.detach(),
            "mean_expert_probs": mean_expert_probs.detach(),
            "max_vio": (total_num_experts * expert_shares.max() - 1).detach(),
            "z_loss": L_z.detach(),
        }
        return output, {"balance": L_balance, "z": L_z}, router_stats

    def _dispatch_grouped(self, x, router_probs, router_choices, expert_counts):
        """y_t = p_{t,e} * Expert_e(x_t) for each token's expert e, with all experts in one
        grouped GEMM per projection. Static shapes and no host sync, so it compiles into the
        surrounding graph. Same maths as _dispatch_loop (up to bf16 GEMM rounding)."""
        B, T, D = x.shape
        E = self.config.n_experts
        N = B * T
        A = GROUPED_MM_ALIGN
        flat_x = x.reshape(N, D)
        flat_choices = router_choices.reshape(N)
        gate = router_probs.reshape(N, E).gather(1, flat_choices[:, None]) # (N, 1) fp32, carries the router gradient
        # Sort tokens by expert. Expert e's tokens go to rows [seg_start_e, seg_start_e + count_e)
        # of a buffer with N + E*A rows, where every segment starts at a multiple of A.
        order = torch.argsort(flat_choices, stable=True)
        sorted_choices = flat_choices[order]
        padded = (expert_counts + A - 1) // A * A
        seg_end = torch.cumsum(padded, 0)
        seg_start = seg_end - padded
        first_sorted = torch.cumsum(expert_counts, 0) - expert_counts # position of each expert's first token in `order`
        rank_in_expert = torch.arange(N, device=x.device) - first_sorted[sorted_choices]
        rows = seg_start[sorted_choices] + rank_in_expert # buffer row of the i-th sorted token
        buf = x.new_zeros(N + E * A, D).index_copy(0, rows, flat_x[order])
        offs = seg_end.to(torch.int32)

        def stacked(name): # (E, in, out) view of the experts' weights, in the activation dtype
            return torch.stack([getattr(e, name).weight for e in self.experts]).to(x.dtype).transpose(1, 2)

        h = F.silu(torch._grouped_mm(buf, stacked("gate"), offs=offs)) * torch._grouped_mm(buf, stacked("up"), offs=offs)
        y = torch._grouped_mm(h, stacked("down"), offs=offs).index_select(0, rows) # (N, D), sorted order
        y = y * gate[order].to(y.dtype)
        return torch.empty_like(y).index_copy(0, order, y).view(B, T, D)

    def _dispatch_loop(self, x, router_probs, router_choices):
        """Reference dispatch: one boolean-mask gather per expert."""
        # Prepare an output tensor. We will modify distinct inputs at each iteration of
        # the loop
        output = torch.zeros_like(x) # (batch, seq_len, n_embd)
        for idx, expert in enumerate(self.experts):
            # Create a mask that's 1 for positions current expert will process, and 0
            # for all other positions
            mask = (router_choices == idx)

            # Extract token embeddings that should go to the current expert
            # It will be flattened into a (n_curr, n_embd) tensor, where n_curr is the total number of tokens
            # in the batch that were routed to the current expert
            curr_expert_input = x[mask] # (n_curr, n_embd)
            curr_expert_output = expert(curr_expert_input) # (n_curr, n_embd)

            # Get a tensor of size (n_curr, ), where element i is the probability from
            # router_probs that the current expert would process this token
            curr_expert_probs = router_probs[mask][:, idx] # (n_curr,)
            curr_expert_probs = curr_expert_probs.unsqueeze(-1)
            assert curr_expert_probs.shape[0] == curr_expert_output.shape[0]

            # Following Shazeer et al. (2017), Eq. 1, an MoE output is the
            # router-weighted sum of expert outputs: y = sum_i G(x)_i * E_i(x).
            # With top-1 routing, this reduces to: y = G(x)_selected * E_selected(x).
            # https://arxiv.org/abs/1701.06538
            curr_expert_probs = curr_expert_probs.to(curr_expert_output.dtype)
            curr_expert_output = curr_expert_output * curr_expert_probs

            # Put the current expert's outputs into relevant positions of output tensor
            output[mask] = curr_expert_output
        return output


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.attn = CausalSelfAttention(config, layer_idx)
        if config.ffn_type not in FFN_TYPES:
            raise ValueError(f"Unknown ffn_type {config.ffn_type!r} (expected one of {FFN_TYPES})")
        if config.balancer not in BALANCERS:
            raise ValueError(f"Unknown balancer {config.balancer!r} (expected one of {BALANCERS})")
        if config.moe_dispatch not in MOE_DISPATCH:
            raise ValueError(f"Unknown moe_dispatch {config.moe_dispatch!r} (expected one of {MOE_DISPATCH})")

        self.is_moe = layer_idx in moe_layer_indices(config)
        if config.ffn_type == "dense":
            self.mlp = DenseMLP(config)
        elif self.is_moe:
            self.mlp = MoEMLP(config, layer_idx)
        else:
            # ffn_type="swiglu", or a non-MoE layer of an MoE model
            self.mlp = SwiGLUMLP(config)

    def forward(self, x, cos_sin, kv_cache, forced_idx=None, capture=None):
        """Returns (x, moe) where moe is None for non-MoE layers and
        (aux_losses, router_stats) for MoE layers."""
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        mlp_input = norm(x)

        if not self.is_moe:
            return x + self.mlp(mlp_input), None
        mlp_output, aux_losses, router_stats = self.mlp(mlp_input, forced_idx=forced_idx, capture=capture)
        return x + mlp_output, (aux_losses, router_stats)


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self, seed=42):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=0.8
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            dense MLP: c_fc uniform; c_proj zero
            MoE SwiGLU: up and gate truncated normal, std=sqrt(0.1/d)
            down truncated normal, std=sqrt(0.1/(3d))
            each truncated at ±2 std
        """
        device = self.transformer.wte.weight.device

        # Create a random number generator for components that are the same
        # in both MoE and dense transformer
        shared_rng = torch.Generator(device=device)
        shared_rng.manual_seed(seed)

        # Create a separate rng for FFNs
        ffn_rng = torch.Generator(device=device)
        ffn_rng.manual_seed(seed + 1)

        # Create a separate rng for routers
        router_rng = torch.Generator(device=device)
        router_rng.manual_seed(seed + 2)

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8, generator=shared_rng)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001, generator=shared_rng)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        
        # First initialize components that have identical structure in both dense and MoE transformer
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s, generator=shared_rng) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s, generator=shared_rng)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s, generator=shared_rng)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero

        def init_swiglu(mlp):
            for weight in (mlp.up.weight, mlp.gate.weight, mlp.down.weight):
                fan_in = weight.shape[1]
                std = (0.1 / fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    weight, mean=0.0, std=std,
                    a=-2 * std, b=2 * std,
                    generator=ffn_rng,
                )

        # Next, initialize the MLPs. A dense SwiGLU layer draws exactly what
        # expert 0 of an MoE layer draws, so with the same seed an E=1 MoE and a
        # dense SwiGLU model start from identical weights.
        for block in self.transformer.h:
            if self.config.ffn_type == "dense":
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4, generator=ffn_rng)  # 0.4x init scale for c_fc
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            elif block.is_moe:
                for expert in block.mlp.experts:
                    init_swiglu(expert)
            else:
                init_swiglu(block.mlp)

        # Finally, initialize the MoE routers and the selection bias
        for block in self.transformer.h:
            if block.is_moe:
                torch.nn.init.uniform_(block.mlp.router.weight, -s * 0.1, s * 0.1, generator=router_rng) # 0.1x init scale for the router
                torch.nn.init.zeros_(block.mlp.expert_bias)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * sequence_len accounts for key @ query matmul FLOPs inside attention.
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len

        attn_flops = self.config.n_layer * 12 * h * q * t
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of active matmul parameters per token. Each contributes 2
        FLOPs/param in the forward pass. All matmuls go through Linear; embeddings
        and per-layer scalars do not contribute. Top-1 MoE uses one expert per
        layer, so exclude the other experts while retaining shared projections.
        """
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        for block in self.transformer.h:
            if block.is_moe:
                for expert in block.mlp.experts[1:]:
                    matmul_params -= sum(
                        m.weight.numel() for m in expert.modules() if isinstance(m, Linear)
                    )
        return matmul_params

    @property
    def moe_layer_indices(self):
        return [i for i, block in enumerate(self.transformer.h) if block.is_moe]

    def parameter_counts(self):
        """Parameter counts under the conventions the experiment needs to state.

        embedding:          input token embedding (a lookup, no matmul FLOPs)
        unembedding:        lm_head
        router:             all MoE routers (E x d per MoE layer)
        active_non_embedding: per-token active block parameters (attention,
                            one expert per MoE layer, routers)
        active_with_unembedding: active_non_embedding + lm_head
                            (= num_matmul_params; the FLOP-count convention)
        active_with_embeddings: + input embedding as well (the joint MoE
                            scaling-law paper's N_act, arXiv:2502.05172 App. A.1)
        total:              every parameter, all experts included
        """
        embedding = sum(p.numel() for p in self.transformer.wte.parameters())
        unembedding = sum(p.numel() for p in self.lm_head.parameters())
        router = sum(block.mlp.router.weight.numel() for block in self.transformer.h if block.is_moe)
        active_with_unembedding = self.num_matmul_params()
        total = sum(p.numel() for p in self.parameters())
        return {
            "embedding": embedding,
            "unembedding": unembedding,
            "router": router,
            "active_non_embedding": active_with_unembedding - unembedding,
            "active_with_unembedding": active_with_unembedding,
            "active_with_embeddings": active_with_unembedding + embedding,
            "total": total,
        }

    @torch.no_grad()
    def update_expert_bias(self, expert_counts, rate=None):
        """Auxiliary-loss-free balancing step (Wang et al. 2024, arXiv:2408.15664).

        expert_counts: (n_moe_layers, n_experts) token counts for the last
        optimizer step, already summed over micro-batches and ranks so every
        rank applies the identical update. b_i <- b_i + u * sign(mean - load_i).
        """
        if self.config.balancer != "bias":
            return
        rate = self.config.bias_update_rate if rate is None else rate
        moe_blocks = [block for block in self.transformer.h if block.is_moe]
        if expert_counts.shape != (len(moe_blocks), self.config.n_experts):
            raise ValueError(f"expert_counts has shape {tuple(expert_counts.shape)}, "
                             f"expected {(len(moe_blocks), self.config.n_experts)}")
        counts = expert_counts.float()
        mean_load = counts.mean(dim=-1, keepdim=True)
        updates = rate * torch.sign(mean_load - counts)
        for block, update in zip(moe_blocks, updates):
            block.mlp.expert_bias.add_(update.to(block.mlp.expert_bias.device))

    def expert_bias_state(self):
        """(n_moe_layers, n_experts) copy of the selection biases."""
        moe_blocks = [block for block in self.transformer.h if block.is_moe]
        if not moe_blocks:
            return None
        return torch.stack([block.mlp.expert_bias.detach().clone() for block in moe_blocks])

    def estimate_decode_flops(self, context_len):
        """
        Estimate forward FLOPs for decoding one token with full attention.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = self.config.n_layer * 4 * h * q * context_len
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Estimate forward FLOPs for prefilling a prompt with full causal attention."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attended_tokens = num_tokens * (num_tokens + 1) // 2
        attn_flops = self.config.n_layer * 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache read by one full-attention decode step, per row."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * context_len

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        total = wte + lm_head + transformer_matrices
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'total': total,
        }

    def setup_optimizer(
        self,
        optimizer_name="adamw",
        router_optimizer="same",
        unembedding_lr=0.004,
        embedding_lr=0.2,
        muon_matrix_lr=0.02,
        adamw_matrix_lr=2e-3,
        router_lr=None,
        adamw_beta1=0.9,
        adamw_beta2=0.95,
        adamw_eps=1e-8,
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_variant="keller",
        weight_decay=0.0,
        router_weight_decay=0.0,
        max_grad_norm=None,
        track_grad_norm=False,
    ):
        """Build the optimizer. Only the transformer-matrix (and router) optimizer
        differs between AdamW and Muon cells; the embedding and lm_head AdamW
        groups are identical in every cell.

        optimizer_name:   optimizer for attention and feed-forward matrices.
        router_optimizer: "same" (the spec's decision: routers use the run's
                          matrix optimizer), "adamw" or "muon".
        router_lr:        None = the matrix learning rate of whichever
                          optimizer the router uses.
        weight_decay:     decoupled weight decay on the matrices, for AdamW and
                          for every Muon variant.
        Learning rates of the matrix and router groups are used as given; only
        the embedding and lm_head groups keep nanochat's 1/sqrt(d/768) scaling.
        """
        if optimizer_name not in ("adamw", "muon"):
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        if router_optimizer not in ("same", "adamw", "muon"):
            raise ValueError(f"Unsupported router optimizer: {router_optimizer}")
        resolved_router_optimizer = optimizer_name if router_optimizer == "same" else router_optimizer

        model_dim = self.config.n_embd
        adamw_betas = (adamw_beta1, adamw_beta2)

        def adamw_group(role, params, lr, wd):
            return dict(kind="adamw", role=role, params=params, lr=lr, betas=adamw_betas, eps=adamw_eps, weight_decay=wd)

        def muon_groups(role, params, lr, wd):
            # Muon stacks parameters of identical shape into one group
            return [
                dict(kind="muon", role=role, params=[p for p in params if p.shape == shape],
                     lr=lr, momentum=muon_momentum, ns_steps=muon_ns_steps,
                     variant=muon_variant, weight_decay=wd)
                for shape in sorted({p.shape for p in params})
            ]

        # STEP 1: embeddings and lm_head always use AdamW, identically in every cell.
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        # Scale the LR for these AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the embedding/lm_head AdamW LR ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            adamw_group("unembedding", lm_head_params, unembedding_lr * dmodel_lr_scale, 0.01),
            adamw_group("embedding", embedding_params, embedding_lr * dmodel_lr_scale, 0.001),
        ]

        # STEP 2: routers
        router_params = [parameter
                         for block in self.transformer.h if block.is_moe
                         for parameter in block.mlp.router.parameters()]
        if router_params:
            if router_lr is None:
                router_lr = adamw_matrix_lr if resolved_router_optimizer == "adamw" else muon_matrix_lr
            if resolved_router_optimizer == "adamw":
                param_groups.append(adamw_group("router", router_params, router_lr, router_weight_decay))
            else:
                param_groups.extend(muon_groups("router", router_params, router_lr, router_weight_decay))

        # STEP 3: attention and feed-forward matrices
        router_param_ids = {id(param) for param in router_params}
        matrix_params = [
            parameter
            for parameter in self.transformer.h.parameters()
            if id(parameter) not in router_param_ids
        ]
        if optimizer_name == "adamw":
            param_groups.append(adamw_group("matrix", matrix_params, adamw_matrix_lr, weight_decay))
        else:
            param_groups.extend(muon_groups("matrix", matrix_params, muon_matrix_lr, weight_decay))

        # Verify that we have assigned every parameter to exactly one group
        model_param_ids = {id(parameter) for parameter in self.parameters()}
        grouped_param_ids = [id(parameter)
                             for group in param_groups
                             for parameter in group["params"]]
        assert len(grouped_param_ids) == len(model_param_ids)
        assert set(grouped_param_ids) == model_param_ids

        # STEP 4: Instantiate the optimizer
        optimizer = DistributedOptimizer(param_groups, max_grad_norm=max_grad_norm, track_grad_norm=track_grad_norm)

        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_metrics=False,
                forced_routes=None, routing_capture=None):
        """
        forced_routes: optional per-layer expert indices for route replay; a
            dict {layer_idx: (B, T) tensor} or a sequence/tensor indexed by layer
            (entries may be None). Non-MoE layers ignore it.
        routing_capture: optional list that MoE layers append their router
            inputs, logits and used routes to (see MoEMLP.forward). Use with the
            uncompiled model and modest batches; nothing is stored otherwise.
        """
        _, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        aux_losses_list = []
        router_stats_list = []

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Per-token losses are used as log probabilities by evaluation and RL,
        # so auxiliary losses are only added to scalar training objectives.
        for layer_idx, block in enumerate(self.transformer.h):
            forced_idx = None
            if forced_routes is not None and block.is_moe:
                if isinstance(forced_routes, dict):
                    forced_idx = forced_routes.get(layer_idx)
                else:
                    forced_idx = forced_routes[layer_idx]
            x, moe = block(x, cos_sin, kv_cache, forced_idx=forced_idx, capture=routing_capture)
            if moe is not None and targets is not None and loss_reduction != 'none':
                aux_losses, router_stats = moe
                aux_losses_list.append(aux_losses)
                if return_metrics:
                    router_stats_list.append(router_stats)

        x = norm(x)

        # Forward the lm_head (compute logits)
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for loss computation

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)

            if aux_losses_list:
                # Per-layer losses are summed over MoE layers, then weighted, so every
                # layer carries the full coefficient (Switch: "for each Switch layer, this
                # auxiliary loss is added to the total model loss"; Ludziejewski et al.'s
                # code sums over layers too). The logged "balance" and "z" are these sums.
                balance_loss = torch.stack([a["balance"] for a in aux_losses_list]).sum()
                z_loss = torch.stack([a["z"] for a in aux_losses_list]).sum()
                balance_coef = self.config.balance_coef if self.config.balancer == "aux" else 0.0
                loss = ce_loss
                if balance_coef:
                    loss = loss + balance_coef * balance_loss
                if self.config.z_coef:
                    loss = loss + self.config.z_coef * z_loss
            else:
                loss = ce_loss
            if return_metrics:
                metrics = {"ce": ce_loss.detach(), "total": loss.detach()}
                if aux_losses_list:
                    metrics.update({"balance": balance_loss.detach(), "z": z_loss.detach()})
                return loss, metrics, router_stats_list
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
