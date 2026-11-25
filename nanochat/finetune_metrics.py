"""
Routing and forgetting metrics of EXPERIMENT_SPEC.md §2 (pure tensor functions).

Conventions (recorded in every fine-tuning run's config.json):
- A route is the expert actually used: argmax(softmax(z) + b) with the selection bias b
  of the bias balancer (b = 0 for the auxiliary-loss balancer).
- Flip cause: the balancing bias goes on the weight side (it is part of the router state).
- Losses are per scored token in nats; means are token-weighted.
"""

import torch


def load_shares(routes, n_experts):
    """(E,) share of tokens routed to each expert; routes is any int tensor of expert IDs."""
    counts = torch.bincount(routes.reshape(-1).long(), minlength=n_experts).double()
    return counts / counts.sum().clamp(min=1)


def maxvio(shares):
    """E * max_i share_i - 1 (0 = perfectly balanced)."""
    return float(shares.numel() * shares.max() - 1)


def route_comparison(r0, r1, n_experts):
    """Instability of one layer's routes r1 against base routes r0 (same token positions).

    flip_rate phi = P[r1 != r0]; chance_flip_rate = 1 - sum_i l0_i l1_i (the flip rate of
    independent routes with these loads); tv = 0.5 sum_i |l1_i - l0_i| <= phi; churn = phi - tv.
    """
    r0, r1 = r0.reshape(-1).long(), r1.reshape(-1).long()
    l0, l1 = load_shares(r0, n_experts), load_shares(r1, n_experts)
    phi = float((r0 != r1).double().mean())
    chance = float(1 - (l0 * l1).sum())
    tv = float(0.5 * (l1 - l0).abs().sum())
    return {
        "flip_rate": phi, "chance_flip_rate": chance,
        "flip_rate_normalized": phi / chance if chance > 0 else float("nan"),
        "tv": tv, "churn": phi - tv,
        "load": l1.tolist(), "maxvio": maxvio(l1), "base_maxvio": maxvio(l0),
    }


def separation(shares_a, shares_b):
    """S = sum_i min(l_i(A), l_i(B)): overlap of two load distributions (1 = identical)."""
    return float(torch.minimum(shares_a, shares_b).sum())


def select_experts(logits, bias=None):
    """Routes as the MoE layer selects them: argmax(softmax(z) + b)."""
    probs = torch.softmax(logits.float(), dim=-1)
    return torch.argmax(probs if bias is None else probs + bias, dim=-1)


def top2_margin(logits):
    """Per-token p_(1) - p_(2) of the router softmax."""
    top2 = torch.topk(torch.softmax(logits.float(), dim=-1), 2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def flip_cause_counts(h0, h1, w0, w1, r0, r1, b0=None, b1=None):
    """Classify one layer's flips (r1 != r0) by cause (spec §2, "Flip cause").

    h0, h1: router inputs of the base and fine-tuned models on the same tokens (N, d);
    w0, w1: their router weights (E, d); b0, b1: selection biases (or None).
    A = new inputs alone flip the token: select(W0 h1, b0) != r0.
    B = new router alone flips it:      select(W1 h0, b1) != r0.
    """
    h0, h1 = h0.reshape(-1, h0.size(-1)).float(), h1.reshape(-1, h1.size(-1)).float()
    r0, r1 = r0.reshape(-1), r1.reshape(-1)
    a = select_experts(h1 @ w0.float().T, b0) != r0
    b = select_experts(h0 @ w1.float().T, b1) != r0
    flipped = r1 != r0
    return {
        "flips": int(flipped.sum()),
        "input_driven": int((flipped & a & ~b).sum()),
        "weight_driven": int((flipped & ~a & b).sum()),
        "either": int((flipped & a & b).sum()),
        "interaction": int((flipped & ~a & ~b).sum()),
    }


def expert_relative_change(model, base_state, n_experts):
    """rho_i = ||W_i - W_i^0||_F / ||W_i^0||_F per MoE layer and expert, with W_i the
    expert's gate, up and down matrices stacked. Returns {layer_idx: [rho_0, ...]}."""
    out = {}
    params = dict(model.named_parameters())
    for layer_idx in model.moe_layer_indices:
        rhos = []
        for e in range(n_experts):
            diff_sq = base_sq = 0.0
            for proj in ("gate", "up", "down"):
                name = f"transformer.h.{layer_idx}.mlp.experts.{e}.{proj}.weight"
                w = params[name].detach().float()
                w0 = base_state[name].float()
                diff_sq += float((w - w0).square().sum())
                base_sq += float(w0.square().sum())
            rhos.append((diff_sq / base_sq) ** 0.5 if base_sq > 0 else float("nan"))
        out[layer_idx] = rhos
    return out


def token_weighted_mean(sums, counts):
    total = float(counts.sum())
    return float(sums.sum()) / total if total > 0 else float("nan")
