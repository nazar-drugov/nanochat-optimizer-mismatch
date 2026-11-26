# Pretraining protocol

Decisions and defaults for the pretraining stage of the optimizer-mismatch study,
recorded 2026-11-23. Fine-tuning and analysis are parked; see `EXPERIMENT_SPEC.md`.
Every run writes its fully resolved settings to `config.json` next to its checkpoints.

## Decided

| Setting | Value | Why |
|---|---|---|
| Data | FineWeb-Edu 100B sample (`karpathy/fineweb-edu-100b-shuffle`) | Spec decision. Same dataset family as Ludziejewski et al. (arXiv:2502.05172), who use "highly filtered FineWeb-Edu" without naming the sample or size |
| Tokenizer | GPT-2 (`--tokenizer gpt2`, 50,257 tokens) | Matches Ludziejewski et al. (arXiv:2502.05172); no retraining |
| Routing | Top-1, Switch-style softmax gate over all E, dropless | Spec. Top-1 Switch routing matches Ludziejewski et al. (balance loss 0.01, router z-loss 0.001, the same as our balancer default). **Dropless differs:** they train with a capacity factor and drop overflowing tokens, and are dropless only at evaluation; they list dropless MoE as future work. The expert shape matches: SwiGLU with hidden size 3·d_model (`SwiGLUMLP` in `gpt.py`) |
| Router optimizer | Same as the matrices (`--router-optimizer same`) | Spec decision |
| Muon variant | `moonlight` (`--muon-variant moonlight`): Nesterov 0.95 (constant), Newton–Schulz ×5 (bf16, coefficients 3.4445 / −4.7750 / 2.0315), update scale 0.2·√max(m,n), decoupled weight decay 0.1 (applied as `1 − lr·0.1` with the unscaled LR; the same value as AdamW's matrices). Matches the Moonlight code (Liu et al., arXiv:2502.16982); same as `torch.optim.Muon(adjust_lr_fn="match_rms_adamw")` | Spec recommendation. The update scaling, Newton–Schulz depth and Muon weight decay agree with the Ephemeral-cost paper (arXiv:2610.22650); the momentum, learning rate, warmup and AdamW settings do not, see the comparison below. Optimizer-Model Consistency (arXiv:2605.06654) also uses the 0.2 RMS-aligned scaling. The nanochat/Keller scaling (`keller`) and nanochat's momentum schedule are not used |
| Muon learning rate | Swept separately from AdamW, not copied from it; see the sweep below | Moonlight's scaling makes the two LRs comparable in size, but their optima need not coincide |
| Batch size | 262,144 tokens, constant for the whole run, identical in every cell and both optimizers | Ludziejewski et al.'s final batch (they ramp 64K → 128K → 256K; we do not). DeepSeek LLM's fit gives ~249K at 3B tokens. See "Batch size" below |
| Schedule | 130M-token linear warmup, constant, linear decay to 0 over the last 20% | Ludziejewski et al. Sec. 5.1.3 |
| Gradient clipping | Global norm 1.0 in every cell | Spec recommendation, applied to both optimizers so cells differ only in the optimizer; Hernanz does the same. Confirmed 2026-11-24. The fraction of clipped steps is compared between optimizers |
| Balancer | Auxiliary loss 0.01 + z-loss 0.001 (`--balancer aux`) | Following Ludziejewski et al. (arXiv:2502.05172): Switch balance loss 0.01 and router z-loss 0.001, each **summed over the MoE layers** (not averaged); see "Auxiliary loss" below |
| AdamW (matrices, embeddings, lm_head) | β = (0.9, 0.95), ε = 1e-8 | Decided 2026-11-24. Hernanz (arXiv:2610.37914) uses exactly these with Muon at 0.95 Nesterov, 5 Newton–Schulz steps, 0.2·√max scaling, weight decay 0.1 and clipping 1.0 for both optimizers. Ludziejewski et al. state no optimizer settings. ε is far below the per-element gradient scale (~1e-4) here |
| Weight decay on matrices | λ = 0.1 for AdamW and Muon, applied as `1 − lr·λ` (option A) | Decided 2026-11-24. Each optimizer keeps its standard recipe (Hernanz; Moonlight). Because decay scales with the LR, the decay timescale 1/(η·λ) differs between cells by the ratio of their LRs (~3.4× at the sweep centres; Wang & Aitchison, arXiv:2405.13698). Mitigation: per-layer weight norms are logged, and one E = 4 Muon run at 3B tokens uses λ chosen to match AdamW's η·λ (see the sweep, step 9). Option B (fixing the timescale in every cell) was not chosen |
| Router | LR = the matrix LR of the router's optimizer; router weight decay 0 | Decided 2026-11-24. Neither reference paper tunes the router separately |
| Pretraining seeds | 3 at E = 8, 1 elsewhere (seed changes only the initialization; the data order is fixed) | Decided 2026-11-24 |
| Initialization | As in the code; see "Initialization" below | Decided 2026-11-24 (Nazar's plan): experts match Ludziejewski et al.'s truncated normal at scale 0.1; the router is fp32, as their "high precision" router |
| Validation | LR selection uses the held-out validation shard `shard_01822.parquet` without the documents D0 took from it; D0 is reserved for forgetting measurements and only logged during pretraining | Decided 2026-11-24, so that D0 does not choose the models it later evaluates. D0 is sampled from that shard (`make_heldout`), so validation skips D0's rows to keep the two disjoint |
| GPU reproducibility | Not bitwise; accepted | Decided 2026-11-24. Seeds measure the noise |
| Model | Depth 10 (width 640, 10 blocks): 118M active with GPT-2 vocab | Confirmed 2026-11-24 |
| Token budget | 3,000,238,080 tokens (11,445 steps) | Confirmed 2026-11-24. Optional 1B and 2B decay branches stay possible (see below) |
| Headline cell | E = 8; pretraining may start with E = 8 only | Decided 2026-11-24 to save time; other E later if the fine-tuning results warrant |


## Pretraining settings: all confirmed (2026-11-24)

### Initialization

Kept as in the code because it is the tested configuration, it does not depend on E or on the
optimizer, and no source reviewed gives a derivation that would justify changing it
(Switch: empirical; DeepSeek: flat std 0.006 for all parameters, no reason given; Moonlight:
HF default 0.02; the MoE muP papers 2601.20205, 2508.09752 and 2605.14200 assume sigmoid gates,
Adam, or scale K with E, and 2601.20205 reports that its LR transfer fails for top-1 with growing E).

Checks before the first real run, for both optimizers at E = 1 and E = 16 (E = 16 is the stress
case: Switch-style top-1 at κ = 1/16 was the marginally unstable setting in 2601.20205):
initial loss close to ln 50,257 ≈ 10.8, per-layer activation RMS finite and similar across E,
router max load violation falling in the first few hundred steps, no loss spike.

Fallback if E = 16 is unstable in either optimizer: scale attention q/k/v by √0.1 (Switch s = 0.1) and
rerun; change nothing else. Do not change the init between cells once runs start.

### LR Schedule
Ludziejewski et al. use it.

### Auxiliary loss

Both losses are computed per MoE layer and **summed** over layers, so each layer carries the
full coefficient (0.01 for the Switch balance loss `E · Σ f_i P_i`, 0.001 for the z-loss
`mean(logsumexp(logits)²)`). Earlier code averaged over layers, which at depth 10 was 10× weaker
per layer. The logged `balance` and `z` metrics are now the sums (E = 1 gives `balance` = number of MoE layers).

Evidence: Switch (arXiv:2101.03961) says "for each Switch layer, this auxiliary loss is added to
the total model loss" with α = 10⁻² (swept 10⁻¹ to 10⁻⁵; 10⁻² balanced load without hurting the
main loss). The released Ludziejewski et al. code (`llm-random`, `research/conditional`) applies the
weights inside each layer and then does `torch.sum` over layers for both losses; its z-loss is
the same form as ours. Their paper states only "0.001 and 0.01". The same repo's
`research/template` code uses a mean, but the `constrained_scaling_laws` configs run the
`conditional` code. The bias balancer (`--balancer bias`) is not from that paper and stays an ablation.

Difference that remains: they train with capacity factor 1.0 and drop overflowing tokens; we are
dropless, so imbalance costs speed, not dropped tokens. Check `max_vio` in the early-step checks.


### Token budget

The proposed 1B and 2B **decay branches** are shorter training horizons for a
robustness check. With the same 130M-token warmup and final 20% LR decay, a
1B-token run would begin decaying around 0.8B tokens and finish around 1B;
a 2B-token run would begin decaying around 1.6B and finish around 2B. The
3B-token run would begin decaying around 2.4B. Comparing their final validation
bits-per-byte would show whether an optimizer ranking seen at 3B also holds at
shorter budgets.

"Branch" means reusing a checkpoint from the common constant-LR phase and
continuing copies with different decay endpoints. These runs are a proposed
additional check, not part of the chosen 3B-token budget. The current launcher
does not automate checkpoint branching; it requires a separate, verified resume
procedure before results can be treated as shared-prefix branches.

## E = 8 learning-rate sweep and base models (current plan, 2026-11-24)

E = 8 is the headline cell; pretraining starts with E = 8 only. `scripts/pretrain_sweep.py`
writes the Slurm jobs and chooses the LRs (`nanochat/pretrain_sweep.py` holds the plan).

1. **Sweep:** E = 8, 499,908,608 tokens (1,907 steps), seed 42, one B200 per run.
   AdamW at 1e-3, 2e-3, 4e-3; Muon at 2e-3, 4e-3, 8e-3. The AdamW grid covers our earlier
   centre (1.19e-3) and Ludziejewski et al.'s Eq. 8 carried to 0.5B tokens (~2.5e-3); the Muon
   grid covers Moonlight's "same LR as AdamW" and Hernanz's ~2× AdamW (both of his model sizes).
   Warmup is 4.33% of the run (83 steps, the fraction 130M tokens is of 3B), then constant,
   then linear decay over the last 20%. Only the final checkpoint is kept.
2. **Validation:** the held-out validation shard (`shard_01822.parquet`) minus the documents D0
   took from it, 10,485,760 tokens every 500 steps. D0 is evaluated at the same steps on
   1,048,576 tokens and only logged; it never selects anything.
3. **Selection** (`select`): per optimizer, the lowest final validation bpb; the LR is the vertex
   of the parabola through that point and its two neighbours in log2(LR). If the best point is
   at the edge of the grid, one more point (×2 or ×½) is run first. A run whose training loss
   becomes non-finite or rises 1 nat above its step-0 loss after warmup stops and counts as
   diverged.
4. **Transfer to 3B tokens:** × (3B / 0.5B)^-0.32 = 0.5635. The exponent is not Hernanz's own
   fit: he takes it from Bjorck et al. (AdamW) and Ren et al. (Muon) and applies ×0.7 for a 3×
   longer run. The transfer check is deferred (below).
5. **Base models:** E = 8, AdamW and Muon at their transferred LRs, 3,000,238,080 tokens,
   seed 42 first, then seeds 43 and 44. Checkpoints every 1,526 steps (400M tokens), which
   includes 0.8B and 1.6B tokens, where optional 1B and 2B decay branches would start.

Estimated B200 time (central, range): sweep 1.7 GPU-hours (1.1–2.8), ~0.9 h on two GPUs; each
3B run ~1.4–1.5 h; first AdamW/Muon pair ~2.6 h after the sweep starts; all three seeds
~10 GPU-hours (7–17).

**Sweep extended downwards (2026-11-24).** The first grid put both optima at the low edge
(final validation bpb: AdamW 1e-3 1.0197, 2e-3 1.0294, 4e-3 1.0579; Muon 2e-3 1.0034, 4e-3 1.0176,
8e-3 1.0396). Both ×½ and ×¼ points were run at once (AdamW 5e-4, 2.5e-4; Muon 1e-3, 5e-4); the
selection rule is unchanged (the ×¼ point is used only if ×½ is best, when the rule would ask
for it anyway).

**Selected E = 8 LRs (2026-11-25).** With the extension (AdamW 5e-4 1.0432, 2.5e-4 1.0797;
Muon 1e-3 1.0060, 5e-4 1.0278) both optima are interior: AdamW 1.155e-3 at 0.5B → 6.511e-4 at 3B,
Muon 1.575e-3 at 0.5B → 8.874e-4 at 3B. The base models (seeds 42, 43, 44) use these.

**Transfer check moved forward, at E = 8 (2026-11-25).** The ×(3B/0.5B)^-0.32 transfer is borrowed
(Bjorck et al. for AdamW, Ren et al. for Muon, via Hernanz), fitted on other setups, assumed equal for
both optimizers and extrapolated 6×; using the sweep LR unchanged is no better founded. From the
sweep's curvature (AdamW ~0.017, Muon ~0.008 bpb per octave²), being 0.83 octaves off (the gap between
those two choices) would cost ~0.012 / ~0.006 bpb, unequally between the optimizers. So before seeds 43
and 44 (held), seed 42 is run at 3B at ×0.5 and ×2 the transferred LR for both optimizers (×2 ≈ the
untransferred sweep LR); `scripts.pretrain_sweep transfer-check` fits the three points with the main run.
If the 3B optimum is more than ~0.25 octave from the transferred LR, the base models are rerun at it.

**Transfer check, first result (2026-11-25).** Muon: 3B optimum −0.06 octave from the transferred LR
(implied exponent −0.34), so the transfer holds. AdamW: ×2 the transferred LR is best (0.9130 vs 0.9180
at ×1, 0.9280 at ×0.5), so its 3B optimum is at least 1 octave above the transferred LR and at or above
the 0.5B optimum: the −0.32 exponent does not hold for AdamW here. The check is extended to ×4 and ×8
for AdamW before the AdamW base models are rerun.

**AdamW base models at the 3B optimum (2026-11-25).** The extended check (×4 0.9178, ×8 0.9345)
puts AdamW's 3B optimum at 1.311e-3, +1.01 octaves from the transferred 6.511e-4 (implied exponent
+0.07). By the rule above, the AdamW base models (seeds 42, 43, 44) are trained at 1.311e-3; Muon's
stay at the transferred 8.874e-4. The selection is `lr_selection_3b.json`. The first AdamW base model
(6.511e-4) is kept as the transfer-check ×1 point, not used for fine-tuning. Consequence for the
report (proposed, not yet agreed): the −0.32 exponent should not be applied unchecked at other E; each E would get its own transfer check.

**Device batch stays 32 for the sweep and the base models (2026-11-24).** B200 benchmark
(job 30414697, E = 8, 262,144 tokens per step): AdamW 425 ms/step at device batch 32 vs 386 ms
at 64 (9% faster); Muon 385 vs 375 ms (3% faster); 64 fits (62 GB reserved). It is not adopted
because it is not numerically equivalent for MoE: the Switch balance loss is computed per
micro-batch (expert shares × mean router probabilities over 65,536 tokens at 32, 131,072 at 64),
so the balancer objective would differ between the sweep and the base models. Measured time per
3B run at 32: ~81 min (AdamW) and ~73 min (Muon) of training steps plus evaluation; each sweep
run ~12–14 min.

**Deferred until the first fine-tuning results:** other E; the E = 4 transfer check (the E = 8 one is
run now, above); batch-size robustness; the weight-decay ablation (one Muon run with λ = 0.1·η_AdamW/η_Muon,
so that η·λ matches AdamW); seed-noise runs for the sweep.

### If other E are added: joint fit across E (decided 2026-11-24)

Sweep every (optimizer, E) cell on 5 points spaced ×2 (AdamW centre 2e-3·E^-0.25; Muon centre
4e-3·2^(−k/4) with k = 0, 1, 2, 3 for E = 1, 4, 8, 16, so the four grids interleave), and choose
LRs by one fit per optimizer: y = a_E + b·(log₂ LR − m₀ − α·log₂ E)², with one curvature b,
using points within 0.05 bpb of each cell's minimum. α is reported (AdamW: compare with
Ludziejewski's −0.25; Muon: no prior). Ludziejewski et al. themselves fit one LR formula in N and
E (Eq. 8; fitted at E = 1 and 8, checked at E = 4 and 32) rather than tuning each E.

Option B (decay branches off longer stable phases instead of transferring) is the fallback
if a transfer check fails.

## Fine-tuning: LR selection, checkpoints and length pilot (decided 2026-11-25)

- **Far dataset: StarCoderData, Python** (user decision). Training reads the first file of
  `bigcode/starcoderdata/python` (218,079 files, 1.04B characters, ~490M GPT-2 tokens; the whole
  Python split is 59 files, ~28B tokens). D1 = 20,000 documents sampled (seed 0) from the last file,
  which training never reads. GPT-2 tokenizes code at ~2.1 characters per token (FineWeb-Edu ~4.7),
  so D1 losses are not comparable with D0 losses.
- **Evaluation packing: best-fit, as in pretraining** (user decision, 2026-11-25, after the D0 gap
  was traced to packing): D0 and D1 rows are built exactly as the pretraining loader builds them
  (every row starts with a BOS, no document crosses a row, long documents are cropped), with document
  IDs kept; each document is read once. The old contiguous packing cut the context of ~49% of the
  scored tokens. `--eval-packing contiguous` keeps the old version for reference.
- **StarCoderData D1 composition accepted as is** (user decision, 2026-11-25, option 1 of 2). Under
  best-fit, D1 is scored on 1,446 whole Python files of at most 2,048 tokens (median 663); the 26% of
  D1 files longer than a row are never scored. Fine-tuning training makes the same selection (short
  files whole, openings of long files), and D0 is built the same way, so "learning on StarCoderData"
  means learning on short Python files; the report states this. Rejected: one document per row cut at
  2,048 tokens (length-neutral, but differs from training and would need another pilot rerun).
- **WikiText D1: official validation + test + sampled training articles** (user decision,
  2026-11-25). Sized for the best-fit packing: 60 validation and 62 test articles plus 510 training
  articles drawn with seed 0 (and removed from training) until the packing fills 512 rows: 632
  articles, 2.5M tokens, shuffled; the evaluation scores the first ≤ 2,048 tokens of 627 of them
  (1,048,461 tokens). Training: 28,934 articles (110.6M tokens), which give only 49.0M training tokens
  per pass under best-fit cropping (articles are long), so a 960-step WikiText run would need 1.28
  passes. **Decided 2026-11-25 (user):** the WikiText pilot uses 60 / 240 / 720 steps (3.9M / 15.7M /
  47.2M tokens; 720 steps = 23,040 rows, within one pass), warmup 8 steps as in the StarCoderData pilot.
  StarCoderData: 215.8M per pass, 0.29 passes at 960 steps.
- **LR selection: (A), conventional** (user decision). Per cell (base model, fine-tuning dataset,
  fine-tuning optimizer, E), the LR with the lowest final D1 loss, mean over fine-tuning seeds, as in
  Zhao et al. (arXiv:2610.22650). The learning–forgetting analysis (forgetting at matched learning
  along the time axis, routing statistics at t*) uses the selected LR's runs. The frontier over all
  LRs (lowest forgetting at each learning level over every LR and checkpoint) is reported as a
  robustness check only. Edge rule as in pretraining: if the selected LR is the top or bottom grid
  point, one more point (×2 or ×½) is run. Forgetting is never compared across cells at a shared
  fixed LR. Known bias: the LR is chosen on the same D1 tokens that measure learning (5 grid points;
  small).
- **Checkpoints: log-spaced** (user decision; `--checkpoint-spacing log`, the default): steps 2, 4,
  8, ... below the end of the constant phase, that end, the middle of the decay and the last step.
  For the 240-step runs: 2, 4, 8, 16, 32, 64, 128, 192, 216, 240 (10 evaluations, as before).
- **Fine-tuning length: pilot first** (user decision). Lengths 60, 240 and 960 steps (3.9M, 15.7M,
  62.9M tokens), run as decay branches of one 960-step trunk per (LR, seed): at the end of each shorter
  length's constant phase the model, optimizer state and data position are copied, that length's
  decay (last 20%) is run and measured, and the trunk resumes. Warmup is fixed at 8 steps (3% of 240)
  for every length, so each length is the same as a separate run of that length; the 240-step branch
  is the main-grid cell for that seed (tested on CPU: `test_decay_branches_equal_separate_runs`).
  Scope: the seed-42 base models (AdamW, Muon), both fine-tuning optimizers, the 5-point LR grid,
  fine-tuning seeds 1–3, StarCoderData first, then WikiText (D1 decided above). Outputs in
  `$SCRATCH/finetune_pilot`.
  What the pilot decides: (1) the main fine-tuning length (whether learning and forgetting still
  change a lot between 240 and 960 steps); (2) whether the selected LR depends on the length and on
  the optimizer pair, and whether the 1/16–1× grid brackets it; (3) the seed-to-seed spread of
  forgetting and learning, which sets the number of fine-tuning seeds; (4) whether the first
  checkpoints catch the early change. What it can show about the question: whether a mismatch
  penalty grows, shrinks or vanishes with length (Zhao et al.'s "ephemeral" claim, here for an MoE).
  What it cannot show: anything across base seeds or E (one base model per pretraining optimizer).

### Fine-tuning main grid (decided 2026-11-25, user)

- **Fine-tuning seeds: 3** (1, 2, 3; the spec's 5 is not needed: pilot seeds agree to ≤ 0.005 nats in
  learning and mostly ≤ 0.03 in forgetting).
- **Base models: all three pretraining seeds** (42, 43, 44) for both optimizers at E = 8; that is what
  the three pretraining seeds are for (base-seed spread of the mismatch effect).
- **LRs: reuse the seed-42 choice.** The full 5-point grid (plus edge extensions) runs only on the
  seed-42 base models (the pilot); seeds 43 and 44 run only the LR(s) selected on seed 42 for each
  (pretraining optimizer, fine-tuning optimizer, dataset) cell, one per length if the selection differs
  between lengths (all lengths come from one trunk per LR). Selection waits for all four seed-42 pairs.
- **ℓ0: chosen after seeing the fine-tuning curves** (user). Safeguard: choose it from the learning
  curves (the range every cell reaches) before computing the forgetting contrast at it, and report the
  contrast over the whole common learning range as well, so the choice of ℓ0 cannot steer the result.
- **Main length:** open (see LOG_CHANGES 2026-11-25): the pilot lengths (60 / 240 / 960 steps;
  WikiText 60 / 240 / 720) are the candidate main lengths; a longer StarCoderData length is being
  piloted (below).
- **StarCoderData main grid on seeds 43 and 44 (started 2026-11-25, user; relaunched 2026-11-26 after the
  replay fix):** one trunk per selected LR to 960 steps with 60/240 branches, fine-tuning seeds 1–3, at
  each seed-42 selection of each pair from the clean pilot rerun (identical to the contaminated
  selection): AdamW → AdamW 1.31e-3, 6.55e-4; AdamW → Muon 3.55e-3, 1.77e-3, 8.87e-4 (the 60-step
  pick ×4 is interior: ×8 7.10e-3 gave D1 1.7363 vs 1.5786); Muon → AdamW 3.28e-4, 6.55e-4; Muon → Muon 1.77e-3, 8.87e-4.
  `--pair-lr-fractions adamw:adamw=1,0.5 adamw:muon=2,1 muon:adamw=0.25,0.5 muon:muon=2,1` plus
  `adamw:muon=4` (separate jobs), output root `finetune_main/`. Base checkpoints pinned by SHA-256 like seed 42.
- **WikiText main grid on seeds 43 and 44 (2026-11-26):** seed-42 selections from the clean pilot rerun
  (all interior), 60 / 240 / 720 steps: AdamW → AdamW 3.28e-4 / 6.55e-4 / 6.55e-4; AdamW → Muon 8.87e-4
  at all three; Muon → AdamW 1.64e-4 / 3.28e-4 / 3.28e-4; Muon → Muon 4.44e-4 at all three. The 720-step
  cells reuse Nazar's unbranched 720-step runs (`/orcd/scratch/orcd/010/ndrugov/finetune_pilot/`, seeds 1–3,
  at exactly these four LRs; unaffected by the replay bug; warmup 8, same data fingerprint and checkpoint
  steps as the clean seed-42 720-step trunks; commit 60fc9f5 plus local edits). The 60- and 240-step
  cells run on hgo as 240-step trunks with a 60-step branch at every selected LR (`--pair-lr-fractions
  adamw:adamw=0.25,0.5 adamw:muon=1 muon:adamw=0.125,0.25 muon:muon=0.5 --branch-tokens 3932160`), output
  root `finetune_main/`. Their constant-phase checkpoints (up to step 192) at the shared LRs must agree with
  Nazar's runs at the same steps (reproducibility check of his code).
- **Long StarCoderData length pilot (started 2026-11-25, user), seed 42:** trunk to 3,200 steps
  (209.7M tokens, 97% of one best-fit pass of the training file; 3,840 would need 1.17 passes) with
  branches at 60, 240, 960 and 1,920 steps; 3-point grid per pair at ×1, ×1/2, ×1/4 the 960-step
  selection (the selected LR fell or held from 240 to 960 steps for every pair); same edge rule and
  selection rule (lowest mean final D1 loss over seeds 1–3). Separate output root so its 60–960-step
  branches (a replication of the pilot cells at those LRs) do not overwrite the pilot's.
  Result (2026-11-26): at 1,920 and 3,200 steps every pair's pick is ×1, the top of this grid; edge
  extension at ×2 (AdamW fine-tuning 1.31e-3, Muon 1.77e-3) submitted, same branches and output root.

## Before the first real run

- Download data: `python -m nanochat.dataset --dataset fineweb-edu-100b -n 100` (3B training
  tokens need ~4.6B raw tokens because the loader crops ~35%; ~55M tokens per shard).
- Cache the GPT-2 tokenizer on a node with internet access:
  `python -c "from nanochat.tokenizer import get_tokenizer; get_tokenizer('gpt2')"`.
- Build the held-out set D0 (~2M tokens, MinHash-deduplicated against training shards):
  `python -m scripts.make_heldout --dataset fineweb-edu-100b --tokenizer gpt2 --workers 32` (CPU job).
- Run the throughput benchmark (spec §8.8) and the two-GPU CUDA optimizer test
  (`tests/test_optim.py::test_two_rank_muon_matches_torch_reference`) on the cluster:
  `python -m scripts.bench_moe_throughput --depth 10 --experts 1,4,8,16 --dispatch grouped,loop --device-batch-size 32 --out bench.jsonl`.
  Check that `grouped` has 0 graph breaks and that `torch._grouped_mm` does not fall back to a per-expert
  loop on the B200 (on an RTX 5080 it does, with one host sync per call).
- On ORCD, use `runs/pretrain_study.sh` to download to pool and stage to
  scratch once per dataset version, then launch many runs using that scratch
  copy. Downloads and staging are not part of each training run. See
  `STORAGE_PATHS.md`.
- Validation reads the held-out validation shard `shard_01822.parquet` without the documents
  D0 took from it (`val_excluded_heldout_rows` in `config.json`); D0 is evaluated alongside and
  only logged. Each run ends with `result.json` (status, final validation and D0 bpb, evaluation
  curve, clipped-step fraction, settings).
- The data is frozen in `data_manifests/fineweb_edu_100_v1/` (100 training shards,
  the validation shard, and D0). Every `train` start and resume through the launcher
  hashes all shards against that manifest (about 30 s) and stops on any difference, so
  every cell of the study is guaranteed to read identical bytes. A resumed run also
  confirms that its shard list and tokenizer match the checkpoint. If the data ever
  changes, build a new D0 and a new manifest version; do not mix versions in one study.
- MoE layers use the grouped-GEMM dispatch on CUDA (`--moe-dispatch auto`). The per-expert loop is kept
  as the reference (`--moe-dispatch loop`); it breaks `torch.compile` and was 1.4–2.4× slower than dense at E ≤ 8 on an RTX 5080.
- GPU runs are not bitwise reproducible (two identical runs differ after a few steps, with either
  dispatch); the data stream is identical across runs and cells, and resume restores the exact state,
  but the continuation is only statistically equivalent.

Sweep and base-model jobs (on the ORCD login node, from the repository):

```bash
python -m scripts.pretrain_sweep jobs --stage sweep --out jobs/      # 2 jobs × 3 runs
sbatch jobs/e8-sweep-adamw.sbatch; sbatch jobs/e8-sweep-muon.sbatch
python -m scripts.pretrain_sweep select --out jobs/                  # writes jobs/lr_selection.json
python -m scripts.pretrain_sweep jobs --stage main --selection jobs/lr_selection.json --out jobs/
sbatch jobs/e8-main-adamw-s42.sbatch; sbatch jobs/e8-main-muon-s42.sbatch
```

Each job runs its runs through `runs/pretrain_study.sh train` on one GPU, skips runs that
already have `result.json`, and continues past a failed or diverged run.
