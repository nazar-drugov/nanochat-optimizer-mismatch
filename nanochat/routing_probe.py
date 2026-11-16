"""Measure top-1 MoE routing on a fixed batch of token IDs."""

from pathlib import Path

import torch


def routing_probe(model, probe_tokens):
    """Return expert IDs with shape (layer, batch, position).

    The full model processes the tokens normally. Hooks only observe each
    router's logits; they do not feed embeddings into later layers or change
    the model's output.
    """
    if probe_tokens.ndim != 2 or probe_tokens.numel() == 0:
        raise ValueError("probe_tokens must be a nonempty (batch, position) tensor")
    if model.config.ffn_type != "moe":
        raise ValueError("routing_probe requires an MoE model")

    blocks = model.transformer.h
    choices = [None] * len(blocks)
    hooks = []
    was_training = model.training

    def capture_router(layer_idx):
        def hook(_router, _inputs, logits):
            if choices[layer_idx] is not None:
                raise RuntimeError(f"router in layer {layer_idx} ran more than once")
            choices[layer_idx] = logits.argmax(dim=-1).detach().cpu()
        return hook

    try:
        for layer_idx, block in enumerate(blocks):
            hooks.append(block.mlp.router.register_forward_hook(capture_router(layer_idx)))
        model.eval()
        with torch.inference_mode():
            model(probe_tokens)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    if any(choice is None for choice in choices):
        raise RuntimeError("the probe forward pass did not visit every router")
    return torch.stack(choices)


def per_layer_change_rates(previous_choices, current_choices):
    """Return each layer's fraction of probe positions that changed expert."""
    if previous_choices.ndim != 3 or previous_choices.numel() == 0:
        raise ValueError("choices must have shape (layer, batch, position)")
    if previous_choices.shape != current_choices.shape:
        raise ValueError("routing snapshots must have the same shape")
    return (previous_choices != current_choices).float().mean(dim=(1, 2))


def save_routing_snapshot(path, step, probe_tokens, choices):
    """Save the fixed inputs and their per-layer choices at one checkpoint."""
    if probe_tokens.ndim != 2 or choices.ndim != 3:
        raise ValueError("expected probe tokens (batch, position) and choices (layer, batch, position)")
    if tuple(choices.shape[1:]) != tuple(probe_tokens.shape):
        raise ValueError("choices and probe tokens must have matching batch and position dimensions")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"routing snapshot already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "probe_tokens": probe_tokens.detach().cpu(),
        "choices": choices.detach().cpu(),
    }, path)


def load_routing_snapshot(path):
    """Load a checkpoint's fixed inputs and choices on CPU."""
    return torch.load(path, map_location="cpu", weights_only=True)


def snapshot_change_rates(previous_snapshot, current_snapshot):
    """Compare snapshots only when they used exactly the same probe tokens."""
    if not torch.equal(previous_snapshot["probe_tokens"], current_snapshot["probe_tokens"]):
        raise ValueError("routing snapshots used different probe tokens")
    return per_layer_change_rates(previous_snapshot["choices"], current_snapshot["choices"])
