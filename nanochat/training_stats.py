"""Small, testable helpers for MoE training diagnostics."""

import json
from pathlib import Path

import torch


def accumulate_router_stats(sums, router_stats, micro_steps):
    """Average equal-size microbatch statistics, retaining the layer axis."""
    if not router_stats:
        return sums
    per_layer = {
        key: torch.stack([layer[key] for layer in router_stats]) / micro_steps
        for key in router_stats[0]
    }
    if sums is None:
        return per_layer
    for key, value in per_layer.items():
        sums[key] += value
    return sums


def expert_gradient_squared_norms(model, device):
    """Sum squared gradients of each expert on this rank, before reduction."""
    squared = torch.zeros((model.config.n_layer, model.config.n_experts), device=device)
    with torch.no_grad():
        for layer_idx, block in enumerate(model.transformer.h):
            for expert_idx, expert in enumerate(block.mlp.experts):
                for param in expert.parameters():
                    if param.grad is not None:
                        squared[layer_idx, expert_idx] += param.grad.detach().float().square().sum()
    return squared


def add_router_log_fields(log_data, router_stats, expert_grad_norms):
    """Add the established W&B MoE field names to a training record."""
    if router_stats is not None:
        for layer_idx, (shares, probs, entropy, z_loss) in enumerate(zip(
            router_stats["expert_shares"], router_stats["mean_expert_probs"],
            router_stats["router_entropy"], router_stats["z_loss"],
        )):
            log_data[f"train/moe/layer_{layer_idx}/router_entropy"] = entropy.item()
            log_data[f"train/moe/layer_{layer_idx}/z_loss"] = z_loss.item()
            for expert_idx, (share, prob) in enumerate(zip(shares.tolist(), probs.tolist())):
                log_data[f"train/moe/layer_{layer_idx}/expert_{expert_idx}_share"] = share
                log_data[f"train/moe/layer_{layer_idx}/expert_{expert_idx}_mean_prob"] = prob
    if expert_grad_norms is not None:
        for layer_idx, layer_norms in enumerate(expert_grad_norms.tolist()):
            for expert_idx, grad_norm in enumerate(layer_norms):
                log_data[f"train/moe/layer_{layer_idx}/expert_{expert_idx}_grad_norm"] = grad_norm
    return log_data


def write_train_stats_jsonl(path, log_data, master_process):
    """Append the exact W&B training record on rank zero when requested."""
    if path is None or not master_process:
        return
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(log_data, allow_nan=False) + "\n")
