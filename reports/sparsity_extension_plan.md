# Extending the optimizer-mismatch experiments to E = 1, 4, 16

**2026-11-27 · Design and implementation proposal (not authorization to run).**
Review of the paper, agreed protocol, completed E=8 study and current implementation.
No new experiments, model changes, or paper prose edits were made for this review.

## 1. Scientific objective and proposed comparisons

The paper's baseline experiment ([methodology and experiments](../docs/report.tex)) varies
the number of top-1 experts at fixed active parameter count and controls for the learning
achieved by fine-tuning. It asks whether optimizer mismatch grows with sparsity, and then
whether routing stability and replay explain the extra forgetting. E=1 is the crucial
no-routing control: mismatched optimization can cause forgetting in a dense model, so a
positive mismatch penalty at E=1 is not evidence that routing causes it.

For each E, corpus, pretraining optimizer and matched-learning definition, let
`Δ_E = F(mismatched) − F(matched)` on the same base and paired fine-tuning seeds. Report both
directions of optimizer switch, not only an average. The two estimands recently agreed
in [PRETRAINING_PROTOCOL.md](../PRETRAINING_PROTOCOL.md) are:

- **endpoint:** minimum F with learning ≥ target on the length-ordered polyline of fully
  LR-decayed endpoints, each at its selected LR; no extrapolation.
- **within-run:** same constrained minimum on a fixed-length time trajectory, including
  zero; separately for each available horizon.

For sparsity scaling, primary comparisons should use **the same learning target and the
same available training horizons across E**, rather than choosing an independently
convenient target for each E. Compare `Δ_E` against `log₂(E)` (0, 2, 3, 4), and report
`Δ_E − Δ_1` as the excess above the dense control. A descriptive trend line plus all
four points is more honest than presuming log-linear scaling. This difference-in-
differences is not itself causal mediation: changing E also changes capacity and
training dynamics. Inspect paired seed differences and the continuous threshold sensitivity.
Do not count checkpoints or decay branches as independent samples.

**Important support check:** the agreed E=8 targets are 1.50 (StarCoderData) / 0.25
(WikiText) for endpoint and 1.00 / 0.12 for within-run, all in nats/token. An E=1 or E=16
base may not attain these. Before computing any cross-E forgetting contrasts, determine
each E's per-seed learning-only support and the intersection. For endpoint curves
intersect [minimum endpoint learning, maximum endpoint learning] across each selected
seed/arm; for within-run [0, maximum learning] separately per common horizon. Keep the
existing E=8 results as the E=8 analysis. If any target falls outside cross-E support,
report that fact, obtain agreement on an alternative common interior target or extra
lengths chosen by learning only, and show whole-support sensitivity. Do not silently
drop hard-to-learn cells, mix thresholds, or choose targets on D0.

## 2. Architecture and fair dense control

Retain the E=8 recipe: 10 blocks, width 640, 5 × 128 heads, context 2,048, untied GPT-2
embedding/unembedding (50,257 tokens, 50,304 padded rows), each block's SwiGLU expert
hidden width 1,920, one dropless top-1 expert per token. The learned fp32 router uses the
selected expert's unrenormalized softmax probability. Approximate parameter counts are
from the exact formula implemented in `GPT.parameter_counts()` and checked by
`tests/test_pretraining_spec.py::test_parameter_counts_follow_the_stated_conventions`:

| E | Active with embeds (M) | Active non-embedding (M) | Total (M) | Active/total (incl. embeds) |
|---:|---:|---:|---:|---:|
| 1 | 117.644 | 53.254 | 117.644 | 100.0% |
| 4 | 117.663 | 53.274 | 228.255 | 51.5% |
| 8 | 117.688 | 53.299 | 375.736 | 31.3% |
| 16 | 117.740 | 53.350 | 670.700 | 17.6% |

The count is fixed to within 0.10M (<0.1%) because router matrices increase by
6,400 parameters per extra expert across all layers; each extra expert adds 36.864M
FFN parameters. Approximately constant active compute is **not** constant total
capacity, actual wall-clock FLOPs, or expert exposure per step (roughly 1/E under
balanced loads). Keep the shared dimension fixed instead of shrinking E=16's experts
to force exactly equal counts. Record measured throughput, memory and load distributions.
At 3B tokens, a balanced expert receives approximately 3B/E assignments per layer
(3B, 750M, 375M, 188M respectively), not equal exposure across E. A fixed-total-
parameter or equal-per-expert-exposure experiment would answer a different question,
require different widths or token budgets, and is not the paper's primary design.

### E=1 dense control (routing-loss flag agreed 2026-11-27)

Harry agreed to an explicit E=1 flag disabling routing losses (equivalent to
`--ffn-type=moe --n-experts=1 --balancer=none --z-coef=0`; implementation and
verification remain pending), not nanochat's legacy ReLU² `dense` MLP. An E=1 MoE routes
every token to expert 0 with gate 1; its LM and expert weights are exactly the dense
SwiGLU control. This preserves the existing FT evaluation plumbing, which rejects
non-MoE bases (`scripts/finetune.py`), while preventing artificial router-mediated forgetting.

**Do not treat default E=1 MoE as strict dense:** the default auxiliary balance loss is
a constant E×f×P = 1 per layer, but router z-loss is
mean(logsumexp(logits)²) = mean(z²) for E=1. It still backpropagates through the
router and shared hidden features and affects global clipping, despite the LM gate being
identically 1. Disabling both loss terms for E=1 only gives a genuine dense control
with a structurally inert router (zero router weight decay); this is a deliberate
architecture-specific exception to E≥4's aux 0.01/z 0.001. Alternative: run literal
`ffn_type=swiglu`, if the user wants no router tensor whatsoever; that requires
adding an explicit zero-router FT/evaluation path and validating identical predictions.
A default-loss E=1 run could serve as a separate diagnostic for z-loss confounding,
not the primary dense control. Existing CPU tests prove equal initial SwiGLU logits
and zero LM router gradient for E=1 without aux/z, but not multi-step optimizer
equivalence at real precision.

Initialization is already separated into shared, expert and router RNG streams
(`nanochat/gpt.py::GPT.init_weights`): shared weights are paired at a given seed across
E and optimizers; expert/router draws are deterministic with the same distribution,
**not** all identical expert-0 weights across layers when E changes (different draw
counts). Preserve this E=8 implementation. With E≥4, keep the existing balancer,
initialization, gate scale and router optimizer. Run the agreed E=1/16 stability
checks (initial CE ≈ ln 50,257; activation scales; finite gradients; MaxVio at E≥4,
no early divergence) for both optimizers. If instability suggests a change such as
the protocol's attention-init fallback, seek approval and assess whether the entire
cross-E comparison must use that same change; do not alter only a troublesome cell
and call it comparable.

## 3. Pretraining and LR selection

Use the exact existing frozen 100-shard FineWeb-Edu sample and GPT-2 tokenizer;
same deterministic data order for every cell and seed, verify fingerprints and
manifest. 0.5B sweep: 499,908,608 tokens/1,907 steps; full: 3,000,238,080 tokens/
11,445 steps. Global batch 262,144, microbatch 32 × 2,048, four accumulations on one
B200; changing device batch modifies the aux-loss objective. Use 83/496 warmup steps
respectively, final 20% linear decay, same validation shard excluding D0 documents,
D0 only logged. Keep both optimizers' E=8 recipe: AdamW β=(0.9,0.95), ε=1e-8;
Muon Moonlight Nesterov 0.95, 5 Newton–Schulz steps, 0.2√max(m,n) scale; both matrix
weight decay 0.1, router decay zero, global norm clip 1.0. Embedding/unembedding
AdamW unchanged even on Muon cells. The router follows the matrix optimizer when E≥4.

Follow the **existing recorded 0.5B sweep grids**, with independent selection (agreed
2026-11-27): seed 42, five points spaced
×2, centers AdamW 0.002·E^-0.25 and Muon 0.004·2^(-k/4) for k=0,1,2,3 at
E=1,4,8,16. (Centers are starting guesses only, not nominal optima.)

| New E | AdamW grid center | Muon grid center |
|---:|---:|---:|
| 1 | 0.002000 | 0.004000 |
| 4 | 0.001414 | 0.003364 |
| 16 | 0.001000 | 0.002378 |

Factors 1/4, 1/2, 1, 2, 4; extend until the observed best point is interior, log invalid
diverged trials explicitly. For each architecture × optimizer, fit a local parabola
in log₂ LR through its best grid point and the two adjacent points, as at E=8.
The earlier proposed joint regression `val_bpb = a_E + b·(log₂ LR − m₀ − α·log₂ E)²`
would have shared curvature and constrained the LR optima to a power law across E.
Harry instead chose independent fits. A cross-E LR scaling fit is only an optional
descriptive analysis after selection; it must not change the selected LRs. Do not
alter the already selected E=8 bases. Selection always uses the non-D0 pretraining
validation set.

**3B transfer verification is essential:** the E=8 AdamW optimum shifted more than one
octave from the borrowed T^-0.32 transfer; Muon transferred well. Never assume
T^-0.32 at other E. For each new E×optimizer, check ×½, ×1, ×2 of the transferred
LR (extend to bracket), use the existing 0.25-octave rule or pre-agreed improved
method. If not retaining ×1 from the check, train final seed-42 main at validated
vertex. Consider centering an AdamW 3B bracket near its untransferred 0.5B optimum as
an efficiency alternative, but agree that change and keep equal tuning quality
across E. Keep transfer-only checkpoints strictly separate from eligible bases.
When a full-run check is reused for a main base, validate its actual schedule, state,
final result and checkpoint inventory; never alias another LR or pretend an unsaved
checkpoint exists.

Pretraining seed plan originally uses one base seed at E≠8. Recommend seed 42 first,
then replicate seeds 43/44 at all E if the aim is an inferential scaling claim:
3 FT data-order seeds on one base cannot measure pretrained-initialization variation.
The one-seed plan is acceptable as an explicitly exploratory design conditional on
seed 42. For comparisons with a one-seed extension, show the seed-42 E=8 result on
the same plot (not E=8 three-base uncertainty vs other E within-base SD as equal
uncertainties). Pair seeds before comparing optimizers; resample by base rather than
treating nested FT seeds as independent replicates.
The protocol also deferred one E=4 Muon weight-decay-timescale control
(λ_M = 0.1·LR_A/LR_M); keep it optional, distinct from the primary grid.

## 4. Fine-tuning matrix and resource choices

All runs should use the same frozen D0 and two existing D1 datasets (StarCoderData
Python held-out and WikiText-103 near-distribution), exact best-fit packed/document
scoring identities, 65,536-token total batch (32 × 2,048 on one B200), 8 warmup steps,
final 20% decay to zero, fresh optimizer state, flat matrix decay zero, existing
embedding/head AdamW LR ×0.1 and small decays, router WD zero, balancer active except
strict E=1 null. Use data-order seeds 1,2,3; compare fingerprints across optimizer,
LR, E and equal horizon. Fine-tuning LR grid is specific to E, base pretraining
optimizer, FT optimizer, corpus and horizon. Begin on seed-42 bases with the approved
five-point grid of 1/16, 1/8, 1/4, 1/2, 1× the **FT optimizer's** new E-specific 3B
pretraining LR; E=8 selections can inform initial extensions, not substitute for
sweeps. Select lowest mean final D1 loss across FT seeds; extend on either edge. D0
never selects LR. Reuse seed-42 selected LRs on the seed-43/44 bases if replicated,
and report that D1 selection/evaluation reuse is shared across E.
The draft paper still mentions five FT seeds, whereas the actual recorded E=8
protocol selects three; the authors need to reconcile this in their own paper text.

| Option | Matched horizons and purpose | Caveat |
|---|---|---|
| A: initial comparable screen | SC 60/240/960; WT 60/240/720; one pretraining seed each | Cheap, likely catches much of learning; may not support endpoint target in some E, cannot claim between-initialization scaling. Compare E8 on same horizons and seed. |
| B: full primary comparison (recommended for eventual scaling claim) | SC 60/240/960/1920/3200; WT 60/240/720; three pretraining seeds each | Matches existing E8 fully replicated cohorts; more compute. SC longer horizon LR selection must be repeated for each E. |
| C: long-WikiText robustness | Also WT 1920/3200 on seed42 in all E | Multi-epoch overfitting, not needed for primary near-distribution comparison. Separate from replicated results; if making main statistical claim here, replicate WT long at every E including E8. |

Keep the short-run branch equivalence validated (actual 8-step warmup, replay buffer
restoration, per-branch decay, data fingerprint). A longer trunk can reuse early
checkpoints across branches but they are not independent. Use 60/240/960/1920/3200
StarCoderData and 60/240/720 WikiText as common **full-grid** lengths if B is agreed;
derive support after collecting D1. Additional horizons require an agreed scientific
reason; avoid assuming a long extension must be run before support is known.
In particular WikiText's 3200 steps ≈ 4.28 best-fit passes, while SC's ≈ one pass. A
corpus comparison at the same step count is not a same-epoch comparison.

## 5. Router/replay measurements

Existing `scripts/finetune.py` saves tokenwise loss/document sums and routes,
per-layer assignment flips, loads, MaxVio, chance-adjusted flip, TV, churn, load
separation, input/weight flip attribution, expert weight change and expert exposure
at log-spaced checkpoints including decay boundaries and final step. Normal
`F`, fixed-route replay term `F_fixed`, and routing term `F_route` must satisfy
`F=F_route+F_fixed` at actual measured checkpoints; preserve the reverse-order
decomposition as a check. Route replay changes chosen expert IDs, not gate
probabilities, and is an intervention at evaluation, not a training-time causal
mediation proof. Compare `Δ_route = F_route(mismatch) − F_route(match)` with
`Δ_total`, not just a percentage of one arm's F. Fractions need nonzero meaningful
denominators and uncertainty; zero or negative forgetting makes ratios unstable.
Use actual observed checkpoint comparisons at matched targets if possible; report
overshoot, and mark interpolated replay scalars explicitly as interpolation, not
replayed hypothetical model states. If endpoint matches require interpolation
between separately decayed models, causal claims may require targeted extra decay
branches after author approval.

E=1 sanity and semantics:

- Expert counts sum to tokens; all expert IDs=0, loads=[1], entropy=0,
  MaxVio=TV=flip=churn=0, separation=1; both replay orderings reproduce normal
  model loss, `F_route=0` and `F_fixed=F`. Dense forgetting remains meaningful,
  even though routing is absent.
- The normalized flip rate is 0/0, and a top-two probability margin is
  undefined (one expert). Mark unavailable, **not** zero. Do not pool E=1
  observations into route correlations as if they were informative routing variation.
  Current `top2_margin` crashes at E=1 because `torch.topk(...,2)` exceeds the
  dimension; fix before any FT E=1 run and ensure JSON/Parquet/W&B represent NA
  without nonfinite valid measurements.
- Across E≥4 raw flip and MaxVio vary mechanically with E (uniform independent
  chance flip 1−1/E); present both raw and chance-adjusted flip/TV and layerwise
  distributions. Do not mistake low load-TV for no tokenwise rerouting (churn
  can be high), or use ten correlated layers per model as ten independent replicates.
  Lower separation on D0/D1 is suggestive of specialization rather than proof that
  replayed routes remain appropriate.

## 6. Implementation path and pre-compute tests

Actual blockers in the existing paths (inspection, not executed changes):

| Component | Required change / checks |
|---|---|
| [`nanochat/pretrain_sweep.py`](../nanochat/pretrain_sweep.py), [`scripts/pretrain_sweep.py`](../scripts/pretrain_sweep.py) | `RunSpec.n_experts` already exists and emits E in names/arguments, but sweep/main/transfer helpers, job/selection CLI and result pooling hard-code E=8 or key only by optimizer. Parameterize E (include E and seed in manifest/selection schemas), five-point grids and independent local fit, transfer checks, explicit dense-control loss overrides, stage-specific script paths. Preserve existing E=8 semantics/tests; reject wrong E/LR/status/data/schedule and rounded-LR name collisions. |
| [`scripts/base_train.py`](../scripts/base_train.py), [`nanochat/gpt.py`](../nanochat/gpt.py) | Model/optimizer training already supports arbitrary E. Confirm actual resolved E=1 aux/z disabled and zero router gradient/WD, and E≥4 same args. CPU tests: parameter counts at each E; same-seed shared weights and optimizer groups; identical E=1 vs SwiGLU logits, gradients, several paired steps with AdamW and Muon and clipping; routing and zero-norm Muon robustness. Existing E=8 regressions remain required. |
| [`scripts/finetune.py`](../scripts/finetune.py), [`nanochat/finetune_metrics.py`](../nanochat/finetune_metrics.py) | Implement single-expert margin/normalized-flip NA path; all-zero diagnostic/replay identity smoke test through a complete tiny E=1 training/evaluation branch, plus E=4/16 grouped and loop dispatch. Check per-document arrays, token-weighted losses, replay identities, NPZ shapes and no nonfinite defined metrics. Validate new E-specific checkpoints without disabling SHA pins. |
| [`scripts/finetune_jobs.py`](../scripts/finetune_jobs.py) and `FINETUNE_BASE_CHECKPOINTS`, `base_run_seed`, `render_finetune_job` in pretrain_sweep | Currently shared base allowlist has only six E=8 SHA-256 entries and regex is `e8-main-...`; generated jobs pin same root. Extend via validated per-E immutable base manifest (actual path, final step 11,445, E, preopt, seed, full LR, SHA, source owner/provenance); keep existing E8 pins; reject transfer/invalid runs, mismatched selection files, mixed E and missing hashes. Add explicit FT trunk/branch/warmup validation (not inferred from maximum length; `finetune.py` computes warmup from --train-tokens). |
| [`scripts/analyze_finetune.py`](../scripts/analyze_finetune.py), [`scripts/finetune_validation_trajectories.py`](../scripts/finetune_validation_trajectories.py), [`scripts/plot_finetune_endpoints.py`](../scripts/plot_finetune_endpoints.py), matched-learning analysis | Currently E=8 fixed expected bases and/or curve-extraction names; add E schema, cohort and horizon checks, per-E selected-LR files, archived-path rejection, raw D0/D1 data and matched constraints. Support NA route metrics without changing E8 results. Unit tests for missing arms/seeds/support, nonmonotonic crossings, overshoot, paired aggregation, E1 null routing, real provenance. Check cross-E support before contrast; document seed-vs-document bootstrap uncertainty. |
| [`scripts/log_finetune_wandb.py`](../scripts/log_finetune_wandb.py), [`scripts/build_checkpoint_registry.py`](../scripts/build_checkpoint_registry.py) | W&B base regex and groups already recognize E; test each new E and E1 undefined metrics serialization, exact-LR run identity and sync. Pretraining names include E; ensure metadata/group/tag and unique stage/seed IDs in both accounts. Registry regex recognizes E but scientific roots are explicit; include agreed new stage roots and actual Nazar/Harry paths, distinguish invalid backups and transfer checkpoints, report actual weights vs eval NPZ. Crucially, the current classification treats all Nazar seed-42 FT runs as replaybug archive and recognizes only his clean standalone seed-43/44 WT720 as valid; scope this legacy rule to those specific historical paths, not future clean multi-E runs. |

Pre-compute gate: unit/regression suite plus independent subagent review and
address findings. Review generated job content against proven E8 scripts. Prior
benchmarks used 1/4/8/16 experts successfully on B200 grouped dispatch, but new E=1
loss and evaluation paths must pass a small smoke test before training; E=16 memory
is the stress case with two model copies and optimizer snapshots during FT. If a
full-memory batch cannot fit, first diagnose snapshot/activation memory and seek
equivalent approaches, e.g. CPU offload of snapshots; do not silently adjust microbatch
or change the balancing objective. Confirm pretraining activation/gradient/clip
stability, exact token/schedule/data identity and W&B handling in a small authorized
smoke test. No jobs until intent and plan are agreed, reviewed and verified.

## 7. Compute/storage and rollout estimates

E=8 benchmarks (`LOG_CHANGES.md`, job 30414697) on one B200 at microbatch 32 gave
0.38–0.44 sec/step for E=1/4/8/16, same active recipe. E=8's measured completed
3B runs cost ~1.4 GPU-hours and 0.5B sweep ~0.26 GPU-hours each including eval.
Planning estimate for new three E only (not a quota promise):

- 30 new short sweep runs (3 E × 2 optimizers × 5 rates): ~7.8 GPU-h plus edge
  extensions. E=8 is reused.
- 18 minimum full-budget transfer/check runs (6 cells × 3 points): ~25 GPU-h;
  if all six require new train-at-vertex final models, another ~8.4 GPU-h. Thus
  initial seed-42 pretraining ~33–42 GPU-h plus edge runs and evaluation variance.
- Replicate seeds 43/44 if agreed: 12 additional bases, ~17 GPU-h. This is
  ~50–59 GPU-h pretraining total before LR edge surprises or ablations.
- Initial short FT screen: 3 E × 2 preoptimizers × 2 FT optimizers × 2 datasets ×
  5 rates × 3 seeds = 360 nominal trunks with decay branches. E=8 clean pilots
  took ~12 B200-hours including edge extensions, suggesting order ~36 GPU-h for
  corresponding three E, not assuming identical runtime. Long extensions and
  replications are additional; the E=8 WikiText long sweep alone took ~13
  GPU-hours, so extending that option to three E is not minor. Establish budget
  after seed-42 pilot measurements and choose enough branch lengths for support.
- E=8 full base scratch was ~28 GB Muon / ~37 GB AdamW at eight checkpoints per
  run; E=16 total params are ~1.78× E8 (not exact storage multiplier). Account for
  actual base/optimizer and checkout snapshot storage, FT eval NPZ arrays and
  mirrored pool files. Retain required actual final model weights and small
  results; no pruning or moving shared originals without author agreement.
  Reuse identical data manifests and stage once, not per job.

Approximate per-E budget below scales E8's completed-run timings by the historical
per-E mean training-step throughput (AdamW/Muon; 0.405 s/step at E8), only as a
planning proxy. FT estimates are more uncertain because evaluation and optimizer
snapshot overhead also varies with E.

| E | Old B200 step time (AdamW/Muon, ms) | 10 × 0.5B sweep (GPU-h) | 6 × 3B transfer points + up to 2 new finals (GPU-h) | Initial pretraining (GPU-h) | +4 bases if 43/44 replicated | Initial both-corpus short FT screen |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 404 / 376 | 2.5 | 8.1–10.8 | 10.6–13.3 | 5.4 | ~12 |
| 4 | 388 / 405 | 2.5 | 8.2–11.0 | 10.8–13.5 | 5.5 | ~12 |
| 16 | 410 / 438 | 2.7 | 8.8–11.7 | 11.5–14.5 | 5.9 | ~12–13 |

All times exclude additional edge checks and long FT/replication. With two available
GPUs, a 33–42 GPU-hour pretraining estimate alone would be 16.5–21 ideal GPU
wall-hours, before scheduler contention and dependencies; actual concurrent capacity
must be agreed with both accounts, not inferred from the old B200 limit.

Suggested rollout: (1) agree questions below and freeze manifest, (2) code/tests
and independent review, (3) approved smoke and 0.5B sweep, (4) fit/check 3B optima
and train/pin seed42 bases, (5) seed42 FT pilot with per-E/D1 LR selections and
learning-only support, (6) agreed long/replicated cells, (7) matched-learning
analysis and separate router/replay study. Maintain shared handoff and W&B (online
verification/offline sync) at each milestone and refresh inventory. Coordinate
division of labor with Nazar's agents before assigning jobs, use persistent SSH
tunnel, and retain per-owner checkpoint/provenance records.
The recent division put StarCoderData and long branches mostly on Harry's account
and the initial WikiText main cells partly on Nazar's. A tidy possible assignment
for new pretraining is one owner per E (e.g. Harry E16 stress case, Nazar E1/E4),
then reuse both owners' final pinned checkpoints on either account for FT;
alternatively preserve the corpus division for FT. Assignments are proposals only:
confirm with Nazar and Harry, exchange exact run manifests (not copied run names),
avoid duplicate work or assuming either account's scratch mapping, and reconcile
both account roots in CHECKPOINTS.md after milestones.

## 8. Decisions needed from Harry before implementation / compute

1. **Claim and replication (initial scope agreed):** seed42 screen first; decide
   on seeds43/44 and population-level scaling inference afterward.
2. **Dense control (agreed):** E=1 MoE with inert one-expert gate and aux/z disabled
   by an explicit flag. Literal dense SwiGLU is the equivalence test; default
   E=1 z-loss could be used only as a separately approved ablation.
3. **Fine-tuning scope (agreed):** option A, three-horizon pilot first, extend later
   depending on common support and the subsequent scientific aim.
4. **Common target:** keep the already declared E8 targets where supported at all E,
   or select a new cross-E interior target from learning-only common support before
   observing new cross-E forgetting? If support is absent, prefer more lengths vs
   a lower threshold? Use common horizons and both estimands, with sensitivity curves.
5. **LR tuning (independent fit agreed):** use separate E/optimizer local 0.5B fits
   and per-E 3B verification analogous to E=8. Confirm material deviations from the
   existing E=8 transfer procedure if needed; equal token budget at fixed active
   parameters remains the paper's design.
6. **Mechanism stage:** analyze route replay at real observed matching checkpoints
   with overshoot or authorize extra targeted decay lengths only if interpolation is
   inadequate? Is E=4 decay-timescale control in scope now or deferred?

These questions affect the comparison and resource cost; none is inferred as
authorization from the request for a plan. The review is grounded in the current
repository; no new measurement, benchmark or external literature claim was added.
