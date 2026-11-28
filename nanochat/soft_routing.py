"""Compact, paired soft-router evidence over *all scored tokens*.

Full base log-probabilities are transient CPU tensors only. The persisted
schema stores exact (non-sampled) float64 reductions, fixed-bin histograms,
and an optional bounded sample of raw float32 routing probabilities/logits.
No training code, RNG state or routing decisions are changed.

Axes in persisted arrays are layer (in the existing NPZ's `layers` order),
document (compact EvalSet doc ID), metric, expert, sample; see schema().
"""
import math
import hashlib

import numpy as np
import torch

SOFT_ROUTER_SCHEMA_VERSION = 1
DEFAULT_SAMPLE_SIZE = 8192
SAMPLE_SEED = 20261127
HIST_BINS = 32
METRICS = ("entropy_nats", "top2_margin", "chosen_gate", "max_probability",
           "js_base_current_nats", "tv_base_current", "raw_z_loss")
SAMPLE_FORMATS = ("probabilities", "logits", "both")


def schema(sample_size, sample_format="logits"):
    return {
        "version": SOFT_ROUTER_SCHEMA_VERSION, "metrics": list(METRICS),
        "scoring_mask": "EvalSet.doc_ids >= 0; router at input position predicting its scored target",
        "sample_size_requested": sample_size, "sample_format": sample_format,
        "sample_seed": SAMPLE_SEED,
        "sample_selection": "sorted uniform without replacement of scored row-major positions; NumPy PCG64",
        "sample_position_hash": "SHA-256 of little-endian int64 row-major positions",
        "sample_position_index": "flattened row-major index in EvalSet tensors; positions saved in base arrays only",
        "model_level_means": "equal weight over MoE layers of all-scored-token means",
        "scalar_statistics": "per-token scalar float32, aggregated float64 over all scored tokens",
        "log_probability": "float32 log_softmax of raw logits; JS uses its probabilities; TV uses same",
        "gate": "unbiased softmax probability of chosen expert (selection bias may affect choice)",
        "raw_z_loss": "logsumexp(raw logits)^2, including E1 even when excluded from training objective",
        "histogram": "32 equal-width bins per metric; last bin includes values above the maximum edge (raw_z_loss=64)",
        "undefined_e1": ["top2_margin"], "undefined_encoding": "metric_valid=false, count=0, sum/sumsq=0; JSON mean=null",
        "e1_probabilities": "structural all ones; probability sample omitted with explicit empty sample axis",
        "arrays": {
            "soft_metric_valid": ["metric"], "soft_doc_counts": ["document"],
            "soft_layer_count": ["layer", "metric"], "soft_layer_sum": ["layer", "metric"],
            "soft_layer_sumsq": ["layer", "metric"],
            "soft_doc_sum": ["layer", "document", "metric"],
            "soft_doc_sumsq": ["layer", "document", "metric"],
            "soft_prob_sum": ["layer", "expert"],
            "soft_doc_prob_sum": ["layer", "document", "expert"],
            "soft_hist_edges": ["metric", "edge"],
            "soft_hist_count": ["layer", "metric", "bin"],
            "soft_sample_positions": ["sample"], "soft_sample_doc_ids": ["sample"],
            "soft_sample_positions_sha256": [],
            "soft_sample_probs": ["layer", "sample", "expert"],
            "soft_sample_logits": ["layer", "sample", "expert"],
        },
    }


def scored_sample_positions(doc_ids, sample_size):
    """Deterministic selection; does not consume the training/global RNG."""
    if sample_size < 0:
        raise ValueError("sample size must be nonnegative")
    scored = np.flatnonzero(doc_ids.numpy().reshape(-1) >= 0)
    size = min(sample_size, scored.size)
    if size == scored.size:
        return scored
    return np.sort(np.random.Generator(np.random.PCG64(SAMPLE_SEED)).choice(scored, size=size, replace=False))


def token_statistics(logits, choices, base_log_probs=None):
    """Float32 softmax/gates and stable base-current scalar distances.

    Returns logp, actual softmax probabilities, and [N, metric] values. For
    E=1, margin's placeholder is zero and its validity/count is false in
    the reducer. When base_log_probs is None we are scoring the base itself.
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -torch.where(probs > 0, probs * logp, 0.).sum(dim=-1)
    if logits.shape[-1] == 1:
        margin = torch.zeros_like(entropy)
    else:
        top2 = probs.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
    chosen = probs.gather(-1, choices.long().reshape(-1, 1)).squeeze(-1)
    maximum = probs.max(dim=-1).values
    if base_log_probs is None or logits.shape[-1] == 1:
        js, tv = torch.zeros_like(entropy), torch.zeros_like(entropy)
    else:
        old = base_log_probs.to(logp.device)
        q = old.exp()
        p = logp.exp()
        logmix = torch.logaddexp(logp, old) - math.log(2)
        # masked zero probability terms remain zero under extreme finite logits.
        # Roundoff can induce tiny negative JS; the mathematical range is [0, ln2].
        js = .5 * (torch.where(p > 0, p*(logp-logmix), 0.).sum(dim=-1) +
                   torch.where(q > 0, q*(old-logmix), 0.).sum(dim=-1))
        js = torch.clamp(js, min=0., max=math.log(2))
        tv = .5 * (p-q).abs().sum(dim=-1)
        same = (logp == old).all(dim=-1)
        js, tv = torch.where(same, 0., js), torch.where(same, 0., tv)
    z_loss = torch.logsumexp(logits.float(), dim=-1).square()
    metrics = torch.stack((entropy, margin, chosen, maximum, js, tv, z_loss), dim=-1)
    return logp, probs, metrics


class SoftRoutingCapture:
    """Accumulate one base or checkpoint eval in row order, with aligned base cache."""

    def __init__(self, eval_set, layers, n_experts, sample_size, sample_format="logits", base=None):
        if sample_format not in SAMPLE_FORMATS:
            raise ValueError(f"invalid router sample format: {sample_format}")
        self.layers = tuple(layers)
        self.layer_idx = {layer: i for i, layer in enumerate(self.layers)}
        self.E = n_experts
        self.sample_size = sample_size
        self.sample_format = sample_format
        self.base = base
        self.identity = (eval_set.name, eval_set.source_sha256, eval_set.packing,
                         tuple(eval_set.doc_ids.shape), eval_set.n_docs, eval_set.n_scored)
        self.seq_len = eval_set.doc_ids.shape[1]
        self.doc_ids = eval_set.doc_ids
        self.doc_counts = np.bincount(self.doc_ids[self.doc_ids >= 0].numpy(), minlength=eval_set.n_docs)
        self.metric_valid = np.array([True, n_experts > 1, True, True, True, True, True])
        if base is not None and (
            base.identity != self.identity or base.layers != self.layers or base.E != n_experts or
            base.sample_size != sample_size or base.sample_format != sample_format or
            not torch.equal(base.doc_ids, self.doc_ids)
        ):
            raise ValueError("soft router base and current evaluation alignment differ")
        if base is None:
            # Probabilities for E1 are structural ones; logits can be sampled
            # only if explicitly requested. A probability-only E1 sample is empty.
            requested = 0 if n_experts == 1 and sample_format == "probabilities" else sample_size
            self.positions = scored_sample_positions(self.doc_ids, requested)
        else:
            self.positions = base.positions
        L, D, M, E = len(layers), eval_set.n_docs, len(METRICS), n_experts
        self.doc_sum = np.zeros((L, D, M), dtype=np.float64)
        self.doc_sumsq = np.zeros_like(self.doc_sum)
        self.doc_prob_sum = np.zeros((L, D, E), dtype=np.float64)
        self.hist_edges = np.stack([np.linspace(0, limit, HIST_BINS+1) for limit in
                                    (math.log(E) if E > 1 else 1., 1., 1., 1., math.log(2), 1., 64.)])
        self.hist_count = np.zeros((L, M, HIST_BINS), dtype=np.int64)
        probability_count = len(self.positions) if E > 1 and sample_format in ("probabilities", "both") else 0
        logit_count = len(self.positions) if sample_format in ("logits", "both") else 0
        self.sample_probs = np.empty((L, probability_count, E), dtype=np.float32)
        self.sample_logits = np.empty((L, logit_count, E), dtype=np.float32)
        self.log_probs = ({l: torch.empty((eval_set.n_scored, E), dtype=torch.float32) for l in layers}
                          if base is None and E > 1 else None)
        self.positions_sha256 = hashlib.sha256(self.positions.astype("<i8").tobytes()).hexdigest()
        self.next_row = 0
        self.next_scored = 0
        self.next_sample = 0

    def add_batch(self, start, ids, entries):
        if start != self.next_row or not torch.equal(ids, self.doc_ids[start:start+ids.shape[0]]):
            raise ValueError("router capture batch is not aligned with eval positions")
        if {entry["layer"] for entry in entries} != set(self.layers) or len(entries) != len(self.layers):
            raise ValueError("missing/duplicate layer in soft router capture")
        mask = ids >= 0
        documents = ids[mask].numpy()
        count = len(documents)
        end = self.next_scored + count
        first_pos = start * self.seq_len
        next_row = start + ids.shape[0]
        sample_end = np.searchsorted(self.positions, next_row*self.seq_len, side="left")
        sampled = self.positions[self.next_sample:sample_end] - first_pos
        sample_slice = slice(self.next_sample, sample_end)
        # The sample uses physical flattened (row, token) coordinates, not
        # compressed-scored coordinates. Verify selection never includes BOS.
        if not np.all(mask.numpy().reshape(-1)[sampled]):
            raise ValueError("router sample includes an unscored token")
        if count:
            for entry in entries:
                layer = entry["layer"]; i = self.layer_idx[layer]
                logits = entry["logits"].float()
                choices = entry["choices"]
                if logits.shape != (*ids.shape, self.E) or choices.shape != ids.shape:
                    raise ValueError("unexpected router capture shape")
                if self.sample_logits.shape[1] and len(sampled):
                    self.sample_logits[i, sample_slice] = logits.reshape(-1, self.E)[torch.as_tensor(sampled, device=logits.device)].cpu().numpy()
                devmask = mask.to(logits.device)
                if self.E == 1:
                    # Structural zero change, one gate, no top2 margin. Avoid
                    # full soft cache and unnecessary arithmetic.
                    doc_n = np.bincount(documents, minlength=len(self.doc_counts))
                    self.doc_sum[i, :, 2:4] += doc_n[:, None]
                    self.doc_sumsq[i, :, 2:4] += doc_n[:, None]
                    self.doc_prob_sum[i, :, 0] += doc_n
                    for j, index in ((0, 0), (2, HIST_BINS-1), (3, HIST_BINS-1), (4, 0), (5, 0)):
                        self.hist_count[i, j, index] += count
                    # Raw logits may vary although the probability and gate
                    # are structural ones and routing losses are disabled.
                    z = logits[devmask].squeeze(-1).square().cpu().numpy().astype(np.float64)
                    if not np.isfinite(z).all():
                        raise ValueError("nonfinite raw z-loss at a measured checkpoint")
                    self.doc_sum[i, :, 6] += np.bincount(documents, weights=z, minlength=len(self.doc_counts))
                    self.doc_sumsq[i, :, 6] += np.bincount(documents, weights=z*z, minlength=len(self.doc_counts))
                    self.hist_count[i, 6] += self.histogram(z, 6)
                    continue
                selected_logits = logits[devmask]
                selected_choices = choices[devmask]
                base_logp = None if self.base is None else self.base.log_probs[layer][self.next_scored:end]
                logp, probs, vals = token_statistics(selected_logits, selected_choices, base_logp)
                if self.log_probs is not None:
                    self.log_probs[layer][self.next_scored:end].copy_(logp.cpu())
                if self.sample_probs.shape[1] and len(sampled):
                    # Map physical sample positions to their rank among scored
                    # positions in the batch without a second softmax.
                    scored = np.flatnonzero(mask.numpy().reshape(-1))
                    ranks = np.searchsorted(scored, sampled)
                    self.sample_probs[i, sample_slice] = probs[torch.as_tensor(ranks, device=probs.device)].cpu().numpy()
                v = vals.cpu().numpy().astype(np.float64)
                p = probs.cpu().numpy().astype(np.float64)
                if not np.isfinite(v).all() or not np.isfinite(p).all():
                    raise ValueError("nonfinite router statistic at a measured checkpoint")
                for j, column in enumerate(v.T):
                    self.doc_sum[i, :, j] += np.bincount(documents, weights=column, minlength=len(self.doc_counts))
                    self.doc_sumsq[i, :, j] += np.bincount(documents, weights=column*column, minlength=len(self.doc_counts))
                    self.hist_count[i, j] += self.histogram(column, j)
                for e, column in enumerate(p.T):
                    self.doc_prob_sum[i, :, e] += np.bincount(documents, weights=column, minlength=len(self.doc_counts))
        self.next_row, self.next_scored, self.next_sample = next_row, end, sample_end

    def histogram(self, values, metric):
        bins = np.searchsorted(self.hist_edges[metric], values, side="right")-1
        return np.bincount(np.clip(bins, 0, HIST_BINS-1), minlength=HIST_BINS)

    def finish(self):
        if (self.next_row != self.doc_ids.shape[0] or self.next_scored != self.doc_counts.sum()
                or self.next_sample != len(self.positions)):
            raise ValueError("incomplete soft router capture")
        return self

    def arrays(self, prefix, include_positions=False):
        self.finish()
        layer_count = np.broadcast_to(self.doc_counts.sum()*self.metric_valid, (len(self.layers), len(METRICS))).copy()
        if not np.array_equal(self.hist_count.sum(axis=-1), layer_count):
            raise ValueError("router histogram counts disagree with scored token counts")
        out = {
            "soft_sample_positions_sha256": np.asarray(self.positions_sha256),
            "soft_metric_valid": self.metric_valid, "soft_doc_counts": self.doc_counts,
            "soft_layer_count": layer_count, "soft_layer_sum": self.doc_sum.sum(axis=1),
            "soft_layer_sumsq": self.doc_sumsq.sum(axis=1),
            "soft_doc_sum": self.doc_sum, "soft_doc_sumsq": self.doc_sumsq,
            "soft_prob_sum": self.doc_prob_sum.sum(axis=1), "soft_doc_prob_sum": self.doc_prob_sum,
            "soft_hist_edges": self.hist_edges, "soft_hist_count": self.hist_count,
            "soft_sample_probs": self.sample_probs, "soft_sample_logits": self.sample_logits,
        }
        if include_positions:
            out["soft_sample_positions"] = self.positions
            out["soft_sample_doc_ids"] = self.doc_ids.numpy().reshape(-1)[self.positions]
        return {f"{prefix}_{key}": value for key, value in out.items()}

    def layer_means(self, layer):
        """Small JSON summary; exact sums/squared sums and histograms stay in NPZ."""
        n = self.doc_counts.sum()
        i = self.layer_idx[layer]
        return {**{f"soft_{name}_mean": float(self.doc_sum[i, :, j].sum()/n) if valid and n else None
                   for j, (name, valid) in enumerate(zip(METRICS, self.metric_valid))},
                "soft_mean_expert_probs": (self.doc_prob_sum[i].sum(axis=0)/n).tolist() if n else None}

    def across_layer_means(self, prefix):
        """Equal-weight mean of the per-layer token-weighted means."""
        n = self.doc_counts.sum() * len(self.layers)
        return {f"soft_{prefix}_{name}_mean": float(self.doc_sum[:, :, j].sum()/n) if valid and n else None
                for j, (name, valid) in enumerate(zip(METRICS, self.metric_valid))}
