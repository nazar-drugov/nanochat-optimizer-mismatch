# Rich router evidence at intermediate fine-tuning checkpoints

**2026-11-27 · Implementation and verification, before new FT pilots.**

## Approved scope

The initial E=1/4/16 seed-42 LR pilots retain evaluation evidence, not binary model/optimizer states (`checkpoint-retention none`), because full-state storage on the two personal quotas is infeasible. Full-state retention functionality is implemented and tested for later selected-LR reruns; no such rerun is authorized now (including E8). See [storage analysis](finetune_checkpoint_storage_plan.md). Keep the existing log-spaced checkpoints, including decay start, midpoint and endpoint, so existing and future within-run analysis remain comparable.

At every measured checkpoint on both fixed held-out D0 and D1, retain all token expert IDs (including row positions, with the scored-token mask recoverable from the saved eval sets). For every scored token, compute the following statistics from the same unforced router forward pass; statistical accumulations are float64, per-token logits/softmax computation fp32.

| Quantity | Definition / purpose |
|---|---|
| Entropy | −Σ pᵢ log pᵢ, nats; confidence of the whole softmax |
| Top-two margin | p₍₁₎−p₍₂₎; confidence near selection boundary |
| Chosen gate | p of the expert actually selected, even with selection bias |
| Maximum probability | maxᵢ pᵢ; distinguish from chosen gate |
| Soft JS drift | Jensen–Shannon divergence between paired base/current soft distributions, nats |
| Soft TV drift | ½Σᵢ|pᵢ(t)−pᵢ(0)|; bounded paired probability drift |
| Raw z-loss | (logsumexp z)², diagnostic even when coefficient is zero |
| Soft expert load | Σ pᵢ over scored tokens, by expert and document; compare to hard load |

For each scalar retain counts, sums/squared sums by layer and document, and 32-bin layer histograms (last bin includes overflow in the unbounded z-loss). Thus the reported means and SDs for these metrics use all scored tokens rather than sampling; arbitrary exact per-token quantiles are not recoverable from these summaries. Full hard IDs enable offline adjacent flip/retention/transitions, churn, load entropy/CV/Gini, doc-level instability and cross-layer co-flips. Existing D0 replay losses and flip-cause attribution are retained; the added scored-only cause fields use the same mask as flip rate, while legacy fields remain for historical comparability. E1 has probability one, zero entropy, JS/TV, flip and replay; top-two margin is undefined and encoded as null/invalid, raw z may be nonzero because it is only a diagnostic.

A deterministic, uniform without-replacement sample of up to **8,192 scored positions** on each eval set is fixed at step zero, sorted by row-major position, with IDs and SHA-256 stored in shared base arrays. Raw **fp32 logits** for these positions at all layers/checkpoints preserve exact sampled softmax probabilities, confidence/distribution statistics and z scale without duplicating probability vectors. Per-token soft metrics outside this sample cannot retrospectively be reconstructed. Full base log-probabilities are transient CPU cache for paired JS/TV, not stored. Capture doesn't affect model/optimizer training, data order, LR or routing choices; D0/D1 hashes and base model pin identify the reference.

## Storage and provenance

Rich arrays are in the existing atomic `ckpt_<step>.npz`, keys `d0_soft_*` and `d1_soft_*`, with schema version, masks, positions hash; base arrays also store actual sample coordinates. Config/result `soft_router` identifies metrics, axes, scoring conventions and sample format. Shared pre-decay prefixes are hardlinked within one actual attempt. Partial attempts are archived, not overwritten; attempt identifiers prevent claiming a distinct regenerated sibling is a bit-identical state. Scalar across-layer soft means go to W&B, per-layer metrics to JSONL/Parquet; raw arrays stay in authoritative scratch and lightweight/evaluation mirror to pool, not W&B. Estimates and actual quota checks are in the storage report; the workflow reserves conservative uncompressed wave sizes against both quotas, retaining completed reservations until a refreshed quota report accounts for them.

## Correctness gates

The implementation is covered by independent soft math, routing capture, branch/replay equivalence, sample identity, W&B backward compatibility, retention and coordinator regression tests. The PyTorch CPU compiler dependency (`setuptools`) in the local virtualenv was restored for the legacy smoke tests; **the final whole repository suite passed, 408 passed / 13 skipped** (previous missing-dependency failures resolved). Independent reviewers also verified the generated canary and full-grid arguments for all three E, exact retention/policy consistency and storage reservations. A real GPU canary per E is the first approved StarCoderData AdamW→AdamW, one LR (¼× selected AdamW pretraining LR), seed 1, across 60/240/960. It is not an extra scientific run, and the full grid skips that completed identical cell. Before submission of the remaining grid, validate all three results/configs, exact checkpoint schedules and soft schema, full route shapes/IDs, all sample hashes and shapes/dtypes, finite reductions, doc/probability consistency, E1 structural values, all same-attempt shared prefix hardlinks, no binary states, and measured physical NPZ footprint. No new E8 training or FT inference has occurred yet.
