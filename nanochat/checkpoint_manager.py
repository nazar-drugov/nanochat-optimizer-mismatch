"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import json
import logging
import warnings
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def make_base_experiment_name(args):
    """Name a pretraining run by every factor that differs between cells.

    Call after nanochat.train_config.resolve_optimizer_args/resolve_moe_args so
    defaults that depend on other options are filled in.
    """
    matrix_lr = (
        args.adamw_matrix_lr
        if args.optimizer_name == "adamw"
        else args.muon_matrix_lr
    )
    router_optimizer = getattr(args, "router_optimizer", "same")
    muon_variant = getattr(args, "muon_variant", "keller")
    moe_layers = getattr(args, "moe_layers", "all")
    balancer = getattr(args, "balancer", "aux")

    parts = [
        f"d{args.depth}",
        args.ffn_type,
    ]

    if args.ffn_type == "moe":
        parts.append(f"e{args.n_experts}")
        if moe_layers != "all":
            parts.append("ml" + moe_layers.replace(":", "").replace(",", "."))
        parts.append(f"bal{balancer}")

    parts.append(args.optimizer_name)
    if args.optimizer_name == "muon":
        parts.append(muon_variant)
    if args.ffn_type == "moe" and router_optimizer != "same":
        parts.append(f"r{router_optimizer}")

    parts.extend([
        f"s{args.seed}",
        f"tok{args.train_tokens}",
        f"bs{args.total_batch_size}",
        f"mlr{matrix_lr:g}",
        f"elr{args.embedding_lr:g}",
        f"ulr{args.unembedding_lr:g}",
    ])

    if args.ffn_type == "moe" and args.router_lr is not None:
        parts.append(f"rlr{args.router_lr:g}")

    parts.append(f"wd{args.weight_decay:g}")
    if args.optimizer_name == "muon":
        parts.append(f"ns{args.muon_ns_steps}")
    grad_clip = getattr(args, "grad_clip", None)
    if grad_clip is not None:
        parts.append(f"clip{grad_clip:g}" if grad_clip > 0 else "noclip")

    parts.extend([
        f"wu{args.warmup_steps}",
        f"wdr{args.warmdown_ratio:g}",
    ])

    return "-".join(parts)


def validate_experiment_name(name):
    """Validate a user-provided name before using it as a checkpoint directory."""
    if not name or not name.strip():
        raise ValueError("run name cannot be empty")
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("run name cannot contain '/', '\\', or be '.' or '..'")
    return name


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def validate_checkpoint_seed(meta_data, seed):
    """Ensure a resumed run uses the seed recorded by the original run."""
    checkpoint_seed = meta_data.get("user_config", {}).get("seed")
    if checkpoint_seed is None:
        warnings.warn(
            "Checkpoint metadata does not contain user_config.seed; "
            f"cannot validate requested seed {seed}. Continuing for compatibility with older checkpoints.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if checkpoint_seed != seed:
        raise ValueError(
            f"Checkpoint seed {checkpoint_seed} does not match requested --seed={seed}. "
            f"Resume with --seed={checkpoint_seed} to preserve the original experiment metadata."
        )


def checkpoint_tokenizer_name(meta_data):
    """Tokenizer recorded in checkpoint metadata; checkpoints from before the
    tokenizer option used nanochat's trained BPE."""
    return meta_data.get("tokenizer") or meta_data.get("user_config", {}).get("tokenizer") or "nanochat"


def patch_missing_expert_bias(model, model_data):
    """MoE checkpoints written before the bias balancer existed have no
    expert_bias buffers. Zero bias is exactly the old selection rule."""
    for name, buffer in model.named_buffers():
        if name.endswith("mlp.expert_bias") and name not in model_data:
            model_data[name] = torch.zeros_like(buffer)
            log0(f"Patching missing {name} with zeros (checkpoint predates the bias balancer)")
    return model_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    patch_missing_expert_bias(model, model_data)
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer the checkpoint was trained with (older checkpoints: nanochat's BPE)
    tokenizer = get_tokenizer(checkpoint_tokenizer_name(meta_data))
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.search(r'model_(\d+)\.pt$', f)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = max(int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files)
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
