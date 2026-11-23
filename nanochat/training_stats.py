"""Small, testable helpers for MoE training diagnostics."""

import json
from pathlib import Path

import torch


SUMMED_ROUTER_STATS = ("expert_counts",)


def accumulate_router_stats(sums, router_stats, micro_steps):
    """Average equal-size microbatch statistics, retaining the layer axis.

    Integer token counts (expert_counts) are summed instead, so after the last
    micro-step they hold the step's load per (layer, expert).
    """
    if not router_stats:
        return sums
    per_layer = {
        key: (torch.stack([layer[key] for layer in router_stats])
              if key in SUMMED_ROUTER_STATS
              else torch.stack([layer[key] for layer in router_stats]) / micro_steps)
        for key in router_stats[0]
    }
    if sums is None:
        return per_layer
    for key, value in per_layer.items():
        sums[key] += value
    return sums


def expert_gradient_squared_norms(model, device):
    """Sum squared gradients of each expert on this rank, before reduction.

    Returns (n_moe_layers, n_experts), in the order of model.moe_layer_indices.
    """
    moe_blocks = [block for block in model.transformer.h if block.is_moe]
    squared = torch.zeros((len(moe_blocks), model.config.n_experts), device=device)
    with torch.no_grad():
        for layer_idx, block in enumerate(moe_blocks):
            for expert_idx, expert in enumerate(block.mlp.experts):
                for param in expert.parameters():
                    if param.grad is not None:
                        squared[layer_idx, expert_idx] += param.grad.detach().float().square().sum()
    return squared


def add_router_log_fields(log_data, router_stats, expert_grad_norms, layer_indices=None, expert_bias=None):
    """Add the established W&B MoE field names to a training record.

    Rows of router_stats, expert_grad_norms and expert_bias follow the model's
    MoE layers; layer_indices maps them to block indices (identity if None).
    """
    def layer_name(row):
        return row if layer_indices is None else layer_indices[row]

    if router_stats is not None:
        for row, (shares, probs, entropy, z_loss) in enumerate(zip(
            router_stats["expert_shares"], router_stats["mean_expert_probs"],
            router_stats["router_entropy"], router_stats["z_loss"],
        )):
            layer_idx = layer_name(row)
            log_data[f"train/moe/layer_{layer_idx}/router_entropy"] = entropy.item()
            log_data[f"train/moe/layer_{layer_idx}/z_loss"] = z_loss.item()
            if "max_vio" in router_stats:
                # Global MaxVio over the step's tokens: E * max_i share_i - 1
                log_data[f"train/moe/layer_{layer_idx}/max_vio"] = (len(shares) * shares.max() - 1).item()
            for expert_idx, (share, prob) in enumerate(zip(shares.tolist(), probs.tolist())):
                log_data[f"train/moe/layer_{layer_idx}/expert_{expert_idx}_share"] = share
                log_data[f"train/moe/layer_{layer_idx}/expert_{expert_idx}_mean_prob"] = prob
    if expert_grad_norms is not None:
        for row, layer_norms in enumerate(expert_grad_norms.tolist()):
            for expert_idx, grad_norm in enumerate(layer_norms):
                log_data[f"train/moe/layer_{layer_name(row)}/expert_{expert_idx}_grad_norm"] = grad_norm
    if expert_bias is not None:
        for row, biases in enumerate(expert_bias):
            layer_idx = layer_name(row)
            log_data[f"train/moe/layer_{layer_idx}/bias_spread"] = (biases.max() - biases.min()).item()
            for expert_idx, bias in enumerate(biases.tolist()):
                log_data[f"train/moe/layer_{layer_idx}/expert_{expert_idx}_bias"] = bias
    return log_data


def write_train_stats_jsonl(path, log_data, master_process):
    """Append the exact W&B training record on rank zero when requested."""
    if path is None or not master_process:
        return
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(log_data, allow_nan=False) + "\n")


# -----------------------------------------------------------------------------
# Data-order fingerprint: a cheap on-device running hash of every training batch,
# saved in checkpoint metadata so paired runs (AdamW vs Muon, resumed vs not) can
# be checked to have consumed exactly the same token sequence (spec §8 test 6).

FINGERPRINT_MODULUS = 2**31 - 1
FINGERPRINT_MULTIPLIER = 1_000_003


def _fingerprint_weights(shape, device):
    generator = torch.Generator().manual_seed(20261123)
    return torch.randint(1, 2**20, shape, generator=generator, dtype=torch.int64).to(device)


def update_batch_fingerprint(fingerprint, tokens, weights=None):
    """fingerprint <- (fingerprint * M + sum(tokens * W)) mod P, all int64 on device.

    tokens: (B, T) integer tensor. Order-sensitive across batches and positions.
    Returns (fingerprint, weights) so callers can reuse the weight tensor.
    """
    if weights is None or weights.shape != tokens.shape or weights.device != tokens.device:
        weights = _fingerprint_weights(tokens.shape, tokens.device)
    batch_hash = (tokens.long() * weights).sum() % FINGERPRINT_MODULUS
    fingerprint = (fingerprint * FINGERPRINT_MULTIPLIER + batch_hash) % FINGERPRINT_MODULUS
    return fingerprint, weights
