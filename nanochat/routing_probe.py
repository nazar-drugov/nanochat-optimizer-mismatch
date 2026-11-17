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
    previous_matmul_precision = torch.get_float32_matmul_precision()

    def capture_router(layer_idx):
        def hook(_router, _inputs, logits):
            if choices[layer_idx] is not None:
                raise RuntimeError(f"router in layer {layer_idx} ran more than once")
            choices[layer_idx] = logits.argmax(dim=-1).detach().cpu()
        return hook

    try:
        # Training configures CUDA matmuls to "high" in compute_init. Keep
        # checkpoint replay independent of the caller's ambient precision.
        if probe_tokens.is_cuda:
            torch.set_float32_matmul_precision("high")
        for layer_idx, block in enumerate(blocks):
            hooks.append(block.mlp.router.register_forward_hook(capture_router(layer_idx)))
        model.eval()
        with torch.inference_mode():
            model(probe_tokens)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
        if probe_tokens.is_cuda:
            torch.set_float32_matmul_precision(previous_matmul_precision)

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


def validate_routing_snapshot(snapshot, expected_step=None, n_experts=None):
    """Reject snapshots that cannot describe top-1 choices for a fixed probe."""
    if not isinstance(snapshot, dict) or not {"step", "probe_tokens", "choices"} <= snapshot.keys():
        raise ValueError("routing snapshot lacks step, probe_tokens, or choices")
    step, tokens, choices = (snapshot[key] for key in ("step", "probe_tokens", "choices"))
    if not isinstance(step, int) or step < 0 or (expected_step is not None and step != expected_step):
        raise ValueError(f"routing snapshot has wrong step: {step}, expected {expected_step}")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.numel() == 0:
        raise ValueError("probe tokens must have nonempty (batch, position) shape")
    if not isinstance(choices, torch.Tensor) or choices.ndim != 3 or choices.shape[1:] != tokens.shape or choices.shape[0] == 0:
        raise ValueError("routing choices must have (layer, batch, position) shape matching probe tokens")
    if tokens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("probe tokens must contain integer IDs")
    if choices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("routing choices must contain integer expert IDs")
    if torch.any(choices < 0) or (n_experts is not None and torch.any(choices >= n_experts)):
        raise ValueError("routing choices contain out-of-range expert IDs")
    return snapshot


def save_routing_snapshot(path, step, probe_tokens, choices, n_experts=None):
    """Save the fixed inputs and their per-layer choices at one checkpoint."""
    if probe_tokens.ndim != 2 or choices.ndim != 3:
        raise ValueError("expected probe tokens (batch, position) and choices (layer, batch, position)")
    if tuple(choices.shape[1:]) != tuple(probe_tokens.shape):
        raise ValueError("choices and probe tokens must have matching batch and position dimensions")
    validate_routing_snapshot({"step": step, "probe_tokens": probe_tokens, "choices": choices}, n_experts=n_experts)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"routing snapshot already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "probe_tokens": probe_tokens.detach().cpu(),
        "choices": choices.detach().cpu(),
    }, path)


def load_routing_snapshot(path, expected_step=None, n_experts=None):
    """Load a checkpoint's fixed inputs and choices on CPU."""
    return validate_routing_snapshot(
        torch.load(path, map_location="cpu", weights_only=True),
        expected_step=expected_step, n_experts=n_experts,
    )


def snapshot_change_rates(previous_snapshot, current_snapshot):
    """Compare snapshots only when they used exactly the same probe tokens."""
    validate_routing_snapshot(previous_snapshot)
    validate_routing_snapshot(current_snapshot)
    if not torch.equal(previous_snapshot["probe_tokens"], current_snapshot["probe_tokens"]):
        raise ValueError("routing snapshots used different probe tokens")
    return per_layer_change_rates(previous_snapshot["choices"], current_snapshot["choices"])
