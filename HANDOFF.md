# Handoff: MoE optimizer-mismatch study (state on 2026-11-26, 13:45 EST)

Written for the next agent. Read this first, then `PRETRAINING_PROTOCOL.md` (every decision with its
date; fine-tuning decisions near the end), `LOG_CHANGES.md` (every change and run outcome, newest
last; the 2026-11-26 entries cover today) and `docs/report.tex` (the paper draft; `% NOTE` lines mark
what is not yet agreed). The original `EXPERIMENT_SPEC.md` is not in the repo; the protocol and report
supersede it.

**Where things stand in one paragraph.** All E = 8 base models are trained. Both seed-42 fine-tuning
pilots (StarCoderData, WikiText-103) were rerun after a decay-branch bug (§5) and every LR selection is
final and interior. The seed-43/44 main grids for both datasets are complete at the seed-42 LRs. A long
StarCoderData pilot (to 3,200 steps) is complete. The analysis script handles the main grid and has been
run without ℓ0 for every original dataset and length. **Both long seed-42 pilots (StarCoderData and
WikiText) are complete, with interior LR selections at 1,920/3,200 steps.** All long WikiText jobs
finished successfully 2026-11-26. Both matched-learning estimands and their targets have now been
agreed and analyzed (see below).
Harry's StarCoderData-only long main-grid extension has completed all 120 branch records, including
48 missing long endpoints, with W&B and scratch/pool verification (see §6). All StarCoderData
horizons are six-base replicated. No additional WikiText extension is authorized. The Slurm
controller was temporarily unreachable at the final check; see LOG_CHANGES for accounting status.

New validation trajectory report (2026-11-27): `reports/finetune_validation_trajectories.md`
shows D0/D1 absolute NLL versus steps in eight optimizer-pair/dataset figures, all selected
lengths and LRs, with seed SDs. The enriched endpoint and trajectory CSVs contain all 312
selected run endpoints and 3,768 checkpoint rows including step zero. Actual next-token top-1
accuracy was not logged. D1-only common support over every within-run horizon is 1.095406 for
StarCoderData and 0.132103 for WikiText. Harry selected endpoint targets 1.50/0.25 and within-run
targets 1.00/0.12 respectively; `scripts/matched_learning.py` implements the two estimands using
the paper's minimum forgetting above threshold. The endpoint contrasts are positive in both
directions and datasets; within-run signs depend on direction and horizon. See
`reports/matched_learning_analysis.md` for the matched-learning results and sensitivity. Cross-E design
`reports/sparsity_extension_plan.md`: Harry approved implementing E=1/4/16, seed 42, original three-horizon
pilots with independent LR selections. On 2026-11-27 15:25 EST, all three architectures were
submitted under `hgo` in precedence E=1/4, then unclaimed E=16 at nice 1000. The immutable
checkout is `/home/hgo/nanochat-mismatch-sparsity-909d98b` (909d98b). GPU smoke IDs:
30735417/19 (E1), 30735420/21 (E4), 30735426/28 (E16); CPU staged coordinators
30735438/39/40 respectively. All were pending Priority at the initial verification;
E1/4 smoke checks both optimizers completed 0:0 with verified online W&B, and all three
CPU coordinators are running. At 15:34 EST, E1 scientific sweep 30735929/30 running;
E4 scientific sweep 30736231/32 queued; E16 smoke still queued behind E1/4.
See `reports/sparsity_launch_report.md` and `/home/hgo/jobs/sparsity-20261127/e<E>/state.json`.
Coordinators advance after successful gates; do not submit duplicate sweeps. At 17:14 EST,
E1 0.5B sweep completed all ten trials (all W&B verified); interior local fits AdamW 1.645416e-3,
Muon 3.143392e-3 (provisional transferred 3B rates 9.273228e-4 and 1.771551e-3). Six 3B checks queued;
E4 sweep was 4/10, E16 smoke queued. At 18:14 EST E4's sweep completed 10/10, both minima
interior: AdamW 1.145145e-3, Muon 1.560277e-3 (provisional 3B 6.453803e-4, 8.793402e-4).
E4 six 3B jobs queued (30750304/05/06/07/08/10); E1 two transfer checks running, four queued;
E16 smoke queued. All E1/E4 sweep W&B records verified. See `reports/sparsity_e1_pretraining_lr_sweep.md`
and `reports/sparsity_e4_pretraining_lr_sweep.md`.

At 22:14 EST E1's initial six 3B transfer checks completed (AdamW best 2× edge, Muon interior).
Harry requested actual intermediate FT model/optimizer states; the original CPU controllers were
deliberately cancelled (no GPU jobs cancelled) and resumed on pin 2921362 with `--hold-ft`:
30772131 (E1), 30772137 (E4), 30772138 (E16). These finish approved pretraining and pin bases,
but do not launch FT until a feasible retention policy is agreed. Actual personal storage free
~610 GB scratch + 976 GB pool, versus ~21.5 TB full measured-state (or ~19 TB every 60 steps)
across initial pilots. See `reports/finetune_checkpoint_storage_plan.md`. Harry chose to
continue initial pilots without binary FT states and capture richer router data instead;
selected-LR full-state reruns await storage and a separate plan. Rich router capture has been
implemented and reviewed, whole repository suite 408 passed/13 skipped, with a one-cell GPU canary
gate before each full grid. Harry clarified that the existing project norms supersede the optional
collaboration kit; no additional ratification or launcher requirement applies.
At 22:53 EST the GPU jobs were left untouched and the three CPU coordinators
were replaced on pin a7e06a7 (30777015 E1, 30777060 E4, 30777061 E16),
resuming their same durable ledgers with no binary FT states, rich soft capture
and a mandatory implementation canary before the approved LR pilot grids. See `reports/finetune_rich_router_capture.md`. E1 4× AdamW
extension queued (30772904), E4 two transfer jobs running; E8 rerun not authorized now.

## 1. Goal and where we are

Measure whether fine-tuning a top-1 MoE language model with a different optimizer than it was
pretrained with (AdamW vs Muon) forgets more at matched learning, and how much of that is routing
(route replay). E = 8 is the headline cell.

| Stage | State |
|---|---|
| E = 8 LR sweep (0.5B tokens) | Done |
| LR transfer check at 3B tokens | Done; AdamW moved to the 3B optimum |
| E = 8 base models, AdamW and Muon × seeds 42, 43, 44 (3B tokens) | **All six done** |
| Fine-tuning data | **Both built**: StarCoderData (Python); WikiText-103 with a 1.1M-token D1 |
| Fine-tuning protocol | LR selection, checkpoints, length pilot **decided** 2026-11-25 (§4) |
| Seed-42 pilots (StarCoderData 60/240/960, WikiText 60/240/720 steps) | **Done** (clean rerun 2026-11-26); all selections interior (§6) |
| Seed-43/44 main grids (both datasets, all pilot lengths) | **Done** 2026-11-26 at the seed-42 LRs |
| Long StarCoderData pilot (seed 42, to 3,200 steps) | **Done** 2026-11-26; the 960-step LRs stay optimal |
| Long WikiText pilot (seed 42, to 3,200 steps) | **Done** 2026-11-26; multi-epoch pilot, all LRs bracketed |
| Long StarCoderData main grid (seeds 43/44, both optimizers, 1,920/3,200) | **Done** 2026-11-27 (all 120 result records) |
| Analysis (`scripts/analyze_finetune.py`) | Main-grid support done and reviewed; run without ℓ0 for every dataset/length |
| Endpoint and within-run ℓ0 | **Agreed:** endpoint 1.50/0.25, within-run 1.00/0.12 (SC/WT) |
| Forgetting contrast at ℓ0 | **Done**, paired seed SD and whole-support sensitivity; document bootstrap and routing share not yet run |
| Other E (1, 4, 16) | E1/E4 0.5B sweeps complete, 3B checks running/queued; E16 smoke queued; coordinators healthy |

## 2. Pretraining: design and results

Design (details and reasons in the protocol): nanochat GPT, depth 10, width 640, 5 heads of 128,
sequence 2048; top-1 Switch MoE of SwiGLU experts (hidden 1920) in every block, dropless, softmax
gate, fp32 router; 118M active parameters (53.3M non-embedding), 376M total at E = 8; GPT-2 tokenizer;
FineWeb-Edu (frozen 100-shard snapshot). Balancer: aux 0.01 + z-loss 0.001, summed over layers.
AdamW (0.9, 0.95), ε 1e-8; Muon moonlight (Nesterov 0.95, 5 Newton–Schulz steps, 0.2·√max(m,n));
weight decay 0.1 on matrices for both (as 1 − lr·λ), router 0; clipping 1.0. 262,144 tokens per step;
3,000,238,080 tokens (11,445 steps); warmup 130M tokens, constant, linear decay to 0 over the last 20%.
Validation = `shard_01822.parquet` minus the D0 documents; D0 is never used to select anything.

LR selection (files in `results/e8/`: `lr_selection.json` = 0.5B sweep, `transfer_check.json`,
`lr_selection_3b.json` = **the LRs the base models use; pass this to `finetune_jobs --selection`**):

| | 0.5B sweep (val bpb) | 0.5B vertex | 3B check (val bpb at ×0.5 / ×1 / ×2 / ×4 / ×8 of the transferred LR) | Final 3B LR |
|---|---|---|---|---|
| AdamW | 2.5e-4 1.0797, 5e-4 1.0432, **1e-3 1.0197**, 2e-3 1.0294, 4e-3 1.0579 | 1.155e-3 | 0.9280 / 0.9180 / **0.9130** / 0.9178 / 0.9345 | **1.311e-3** (check vertex, +1.01 octave from the transferred 6.51e-4) |
| Muon | 5e-4 1.0278, 1e-3 1.0060, **2e-3 1.0034**, 4e-3 1.0176, 8e-3 1.0396 | 1.575e-3 | 0.9043 / **0.9015** / 0.9050 | **8.874e-4** (transferred; check vertex −0.06 octave) |

The borrowed transfer ×(3B/0.5B)^−0.32 held for Muon and failed for AdamW (implied exponent +0.07).
Sources of the −0.32 (Bjorck et al., Ren et al.) and why they don't match our setup: protocol, LOG.

Base models (final validation bpb; perplexity = 2^(bpb × 4.6788 bytes/token)):

| Run | Val bpb | Val PPL | D0 bpb |
|---|---|---|---|
| e8-main-adamw-s42 / s43 / s44 | 0.9126 / 0.9127 / 0.9171 | 19.29 / 19.30 / 19.57 | 0.9010 / 0.9008 / 0.9055 |
| e8-main-muon-s42 / s43 / s44 | 0.9015 / 0.9010 / 0.9005 | 18.61 / 18.58 / 18.55 | 0.8902 / 0.8896 / 0.8887 |

Muon leads by 0.013 bpb (mean). The superseded AdamW base at 6.51e-4 is kept as
`e8-xfer-adamw-lr0.000651-s42` (transfer-check ×1 point); do not fine-tune it.
Cost: sweep 2.6 GPU-h, transfer check 8.3 GPU-h, base models 8.5 GPU-h (one B200 each, ~1 h 25 min per 3B run).

## 3. Where everything is (MIT ORCD; all readable by every cluster user)

For the global per-run and per-checkpoint path inventory across both authors, consult
`CHECKPOINTS.md` and `results/checkpoint_registry/`. Refresh it when experiments or paths change.

- Scratch root (`$SCRATCH` in our jobs): `/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/`
  - `base_checkpoints/<run>/` — model weights, `config.json`, `result.json` for every sweep, transfer
    and base run. From another account, point `scripts/finetune.py --base-checkpoints-dir` here.
  - `base_data_fineweb_edu_100b/`, `heldout/fineweb-edu-100b/d0.parquet` (D0).
  - `finetune_data/wikitext103/`, `finetune_data/starcoderdata/` (each: `train_*.parquet`, `val_00000.parquet` = D1,
    `manifest.json`; the whole `finetune_data/` is world-writable at the user's request). `finetune_data/codeparrot/` is obsolete (deletable).
  - `logs/<run>.jsonl|.log` — pretraining logs.
- Pool copy of all small outputs: `/orcd/pool/006/hgo/nanochat-optimizer-mismatch/results/`.
  Note: `meta_*.json` files there are 0.4–0.8 GB each (12 GB total); worth slimming.
- Fine-tuning outputs (scratch root above; all copied to pool `results/` by the jobs):
  - `finetune_pilot/` — **clean** seed-42 pilots, both datasets, every grid LR (12 GB).
  - `finetune_main/` — seed-43/44 main grids (StarCoderData 960-step trunks with 60/240 branches;
    WikiText 240-step trunks with 60 branches), one trunk per distinct selected LR.
  - `finetune_pilot_long/` — long StarCoderData pilot, seed 42, 3,200-step trunks with 60/240/960/1,920
    branches at ×¼, ×½, ×1, ×2 the 960-step picks.
  - `finetune_pilot_long_wikitext103/` — clean long WikiText pilot, seed 42, 3,200-step trunks with
    60/240/720/1,920 branches; ×1, ×½, ×¼, ×⅛, ×1/16 the 720 picks (plus ×1/32 for AdamW → Muon).
  - `finetune_main_long/` — new StarCoderData main-grid extension on seeds 43/44 at selected seed-42
    LRs; 1,920/3,200 plus repeat short branches (not additional independent measurements).
  - `analysis/<dataset>-<T>-pilot/` and `analysis/<dataset>-<T>-main/` — `analyze_finetune` outputs without
    ℓ0; `analysis/staging_nazar_wt/` = symlinks to Nazar's seed-43/44 WikiText runs (see §6).
  - `finetune_pilot_v1_replaybug/`, `finetune_pilot_long_v1_replaybug/` — **invalid** (replay bug), kept for
    the record; never analyse them.
- Nazar's root `/orcd/scratch/orcd/010/ndrugov/finetune_pilot/`: his unbranched 720-step WikiText runs on
  seeds 43/44 (valid, reused as the 720-step main-grid cells) **and his old seed-42 pilot runs (invalid,
  replay bug)**. Never pass this root directly to the analysis; use the staging root.
- Job scripts and Slurm outputs: `/home/hgo/jobs/` (`ft-pilot2/` clean pilots, `ft-main2*/` main grids,
  `ft-long2*/` long pilot). Helper scripts there (not in the repo):
  `run_analysis.sh` (all pilot + main analyses), `pilot_summary.py <root> <dataset> <out.json>` (per-cell
  summary + selection with edge flags), `main_table.py` (s42/s43/s44 comparison at the selected LRs),
  `wt_repro.py` (WikiText reproducibility check vs Nazar), `make_ext_jobs.py` / `make_long_ext_jobs.py`
  (edge-extension jobs through the reviewed `render_finetune_job`).
- hgo checkout: `/home/hgo/nanochat-optimizer-mismatch` (at f4e814e; fine-tuning code unchanged since 6506351).

## 4. Fine-tuning protocol (decided 2026-11-25; protocol section "Fine-tuning: LR selection, checkpoints and length pilot")

- Datasets: near = WikiText-103 (detokenized articles; D1 = official validation + test + 510 training
  articles sampled with seed 0 and removed from training until the best-fit evaluation fills 512 rows:
  632 articles, 2.5M tokens, shuffled, of which the evaluation scores 1,048,461 tokens from 627;
  training 28,934 articles, 110.6M tokens); far = **StarCoderData, Python only**
  (train = first file, 218,079 documents, ~490M GPT-2 tokens; D1 = 20,000 documents sampled with
  seed 0 from the last file, ~43M tokens, first 1,048,576 scored). Built by
  `scripts/make_finetune_data.py --dataset starcoderdata` (gated; needs an HF token with the terms
  accepted, read from `$HF_TOKEN` or `~/.cache/huggingface/token`). Run it on `mit_data_transfer`
  (33 s); `mit_normal` queued it for hours.
- Per cell (base model × dataset × fine-tuning optimizer): 5 LRs = {1/16, 1/8, 1/4, 1/2, 1} × the
  fine-tuning optimizer's 3B pretraining LR (AdamW 8.19e-5 … 1.311e-3; Muon 5.55e-5 … 8.87e-4).
  65,536 tokens per step, 15.7M tokens (240 steps) by default, warmup 8 steps, linear decay over the last
  20%, weight decay 0, embedding/lm_head LR 0.1× pretraining, balancer active, fresh optimizer state.
  Seeds set the data order (paired across cells).
- **LR selection (A):** the LR with the lowest final D1 loss, mean over seeds (as Zhao et al.,
  arXiv:2610.22650). Edge rule: if it is a grid end, run one more point. The frontier over all LRs is a
  robustness check only. Never compare forgetting across cells at a shared fixed LR.
- **Checkpoints: log-spaced** (default): 2, 4, 8, …, end of constant phase, mid-decay, last step
  (240 steps: 2, 4, 8, 16, 32, 64, 128, 192, 216, 240).
- **Length pilot first:** lengths 60 / 240 / 960 steps (3.9M / 15.7M / 62.9M tokens) as decay branches
  of one 960-step trunk per (LR, seed) (`--branch-tokens 3932160 62914560`). Each branch equals a
  separate run of its length (CPU test `test_decay_branches_equal_separate_runs`); the 240-step branch
  is the main-grid cell for that seed. Scope: seed-42 base models, both fine-tuning optimizers, 5 LRs,
  seeds 1–3, StarCoderData. It decides: the main length; whether the selected LR depends on length or
  optimizer pair and whether the grid brackets it; seed spread (→ number of seeds); whether early
  checkpoints catch the first change. It can also show whether a mismatch penalty grows, shrinks or
  vanishes with length (Zhao's "ephemeral" claim, for an MoE). It says nothing across base seeds or E.

## 5. Decay-branch replay bug (fixed; all affected runs redone)

Found 2026-11-25 and fixed in d330586. `ReplayLoader` kept references to the loader's reused buffer,
so after each branch the trunk replayed the branch's *last* batch in place of all of its batches: every
branched run longer than the shortest branch trained on repeated batches. All 240/720/960-step pilot
results from before the fix were invalid. Both seed-42 pilots were rerun from scratch on hgo
(`finetune_pilot/`); unbranched runs (Nazar's 720-step WikiText seed-43/44 runs) were unaffected. The
rerun moved two of 24 selections: StarCoderData AdamW → Muon at 60 steps (to 3.55e-3,
a near-tie in an unaffected branch), and WikiText Muon → AdamW at 240 steps (from 1.64e-4
to 3.28e-4). See `reports/finetune_pilot_report.md` for the verified before/after table. Impact measurement and the fix's
tests: LOG_CHANGES 2026-11-25. Do not launch branched runs from a commit before d330586.

## 6. Results so far and next steps

### Seed-42 LR selections (final; lowest mean final D1 loss over seeds 1–3, all interior)

| pretrain → fine-tune | SC 60 | SC 240 | SC 960 | SC 1,920 / 3,200 | WT 60 | WT 240 | WT 720 |
|---|---|---|---|---|---|---|---|
| AdamW → AdamW | 1.31e-3 | 1.31e-3 | 6.55e-4 | 6.55e-4 | 3.28e-4 | 6.55e-4 | 6.55e-4 |
| AdamW → Muon | 3.55e-3 | 1.77e-3 | 8.87e-4 | 8.87e-4 | 8.87e-4 | 8.87e-4 | 8.87e-4 |
| Muon → AdamW | 3.28e-4 | 6.55e-4 | 6.55e-4 | 6.55e-4 | 1.64e-4 | 3.28e-4 | 3.28e-4 |
| Muon → Muon | 1.77e-3 | 8.87e-4 | 8.87e-4 | 8.87e-4 | 4.44e-4 | 4.44e-4 | 4.44e-4 |

Per-cell means and SDs: `results/e8/finetune_pilot_starcoderdata_summary.json`,
`results/e8/finetune_pilot_wikitext103_summary.json`, `results/e8/finetune_pilot_long_starcoderdata_summary.json`
(lists of {pre, ft, T, lr, n, seeds, learn, forget, d1, learn_sd, forget_sd}). Seeds agree closely
(SD ≤ 0.023 nats at the selected LRs).

### Main grids (seeds 43/44, done)

Every selected cell has 3 completed seeds on both base seeds. Seeds 43/44 track seed 42: forgetting
within ~0.05, learning offset by a roughly constant amount per base model. WikiText 720-step cells are
Nazar's runs; all 36 shared-LR hgo 240-step trunks match them at steps 2–128 (GPU noise level; one
transient at step 32), so his locally edited code is equivalent.

### Long pilot (done)

At ×1 the 960-step pick, per doubling past 960 steps learning gains ~0.05–0.08 nats and forgetting
~0.11–0.19. The ×½ LR forgets much less for nearly the same learning (Muon → Muon at 3,200 steps:
×½ 1.812 / 0.946 vs ×1 1.815 / 1.455 learning / forgetting).

### Long WikiText pilot (completed 2026-11-26)

Harry approved a seed-42-only extension to 1,920/3,200 steps with shorter 60/240/720 branches and
three fine-tuning seeds. Initial grid is ×1, ×½, ×¼ of each 720-step pick; the edge rule applies.
At 3,200 steps WikiText sees approximately 4.28 effective training passes; this is intentional.
Jobs 30622317/30622320/30622321/30622322 (`/home/hgo/jobs/ft-wt-long/`) were generated by the
reviewed generator and diff-checked against `ft-long2/`, with no training code changes.
Both AdamW-pretrained optimizer pairs have a low-edge ×¼ winner at 1,920 steps; ×⅛ edge extensions
30631324/30631325 were submitted with lower priority (`ft-wt-long-x8/`, new W&B-logging generator).
Muon-pretrained ×⅛ jobs 30646301/30646302 also finished; 240 branches verified in W&B. All 1,920-step
LRs are interior, but all 3,200-step winners are still on the low edge. ×1/16 extension jobs
30652107/09/13/15 submitted (`ft-wt-long-x16/`). Harry requested check-ins every 60 minutes.
Output root `finetune_pilot_long_wikitext103/`, separate from the original pilot and StarCoderData.
Final additional AM ×1/32 job 30659709 completed; all 315 branch results complete and every long
winner bracketed. Validated 144 short branch reproductions at shared LRs (48 each at 60, 240, 720),
identical hashes/paired fingerprints; max |Δ learning| 0.00134, max |Δ forgetting| 0.00337.

| Pretrain → fine-tune | WT 1,920 LR | WT 3,200 LR | Learning / forgetting at 1,920 | Learning / forgetting at 3,200 |
|---|---:|---:|---:|---:|
| AdamW → AdamW | 1.638e-4 | 8.192e-5 | 0.4006 / 0.3018 | 0.3901 / 0.3207 |
| AdamW → Muon | 2.218e-4 | 5.546e-5 | 0.4012 / 0.3441 | 0.3833 / 0.2789 |
| Muon → AdamW | 1.638e-4 | 4.096e-5 | 0.3763 / 0.3621 | 0.3672 / 0.2940 |
| Muon → Muon | 1.109e-4 | 5.546e-5 | 0.3843 / 0.2895 | 0.3732 / 0.3151 |

At selected LRs, every pair learns less at 3,200 than at 1,920. Unlike StarCoderData, the long
WikiText runs need dramatically lower LRs. `results/e8/finetune_pilot_long_wikitext103_summary.json`
holds all 105 cells; `reports/finetuning_mismatch_endpoints.md` includes final endpoint plots and
raw seed-level data for both corpora and all lengths. Only base seed 42 was run long. No claims
of a formal matched-learning effect yet.

### Analyses already run (no ℓ0)

`/home/hgo/jobs/run_analysis.sh` runs, per dataset and length, `analyze_finetune` on the seed-42 pilot
(`analysis/<ds>-<T>-pilot/`), then on all six bases with `--lr-selection` pointing at that pilot
analysis (`analysis/<ds>-<T>-main/`). All checks passed. **Common D1-learning support** (largest
admissible ℓ0) over all six bases: StarCoderData 1.095 / 1.375 / 1.576 at 60 / 240 / 960 steps;
WikiText 0.132 / 0.227 / 0.335 at 60 / 240 / 720.
Long WikiText seed-42-only support: 0.374261 / 0.366772 at 1,920 / 3,200 (preliminary analysis
with `--ell0` omitted in `analysis/wikitext103-<T>-pilot-long/`; not common across six bases).

### Long StarCoderData main grid (completed 2026-11-27)

Harry confirmed the intent is to quantify long-horizon persistence of mismatch across pretrained
initializations. Eight jobs 30673385/30673387/30673388/30673389/30673390/30673391/30673394/30673395
(`ft-main-long/`) cover four base runs × two FT optimizers × three FT seeds. All sbatch arguments and
W&B filters checked; training code unchanged. Separate output `finetune_main_long/` with short
reproducibility branches; select only 1,920/3,200 when combining with seed-42 `finetune_pilot_long/`.
No new LR sweeps or WikiText jobs. All 120 branches completed and were verified online in W&B;
all 1,368 evaluation NPZs present in scratch and pool. 72 paired short branches have identical
fingerprints, base and eval identities; the largest 960-step forgetting repeat difference is
0.03886 (Muon seed 43 → AdamW, FT seed 1; differences precede replay windows). Median 960 ΔL/F
is 0.00021/0.00399. Six-base analysis passed at 1,920/3,200 (72 runs, 36 selected each); common
learning support 1.65454/1.70389. Reports/plots/CSV and registry updated. The original short
branches remain the selected main data; only the 48 new long endpoints are incorporated.

### Next steps

1. **User decisions (ask; do not choose):**
   - **Main fine-tuning length.** No more training is needed for any of the five StarCoderData
     horizons (60/240/960/1,920/3,200), or the original WikiText horizons (60/240/720). If 1,920 or
     3,200 WikiText is needed as a fully replicated main length, seeds 43/44 need longer runs;
     this is not currently authorized.
   - **Sparsity extension:** see `reports/sparsity_extension_plan.md` for open design choices, E=1
     control correctness, LR tuning, replication, and estimated resource costs.
2. **Document bootstrap and router analysis** (separate scientific step; confirm intent): for the
   within-run estimand, `run_analysis.sh` can be run with `--ell0 <value>` added to the main
   analyses (bootstrap defaults to 500 repetitions; it reads every checkpoint's per-document arrays, so it
   may be worth a CPU Slurm job rather than the login node). For the "whole range" safeguard, repeat at a
   grid of ℓ0 values up to the common support. Outputs: `contrasts` (per base), `pooled_contrasts` (mean
   over base seeds per pretraining optimizer), `bootstrap.mismatch_ci` / `pooled_mismatch_ci` (the pooled CI
   holds base runs fixed; report the per-base spread beside it), `matched_runs.csv` (F_route / F_fixed
   share), router CSVs and plots.
3. **Completed StarCoderData long-grid generation (for reproducibility):** seed-43/44 jobs used the
   reviewed generator:
   ```
   .venv/bin/python -m scripts.finetune_jobs --selection results/e8/lr_selection_3b.json \
     --base-runs e8-main-adamw-s43 e8-main-adamw-s44 e8-main-muon-s43 e8-main-muon-s44 \
     --dataset starcoderdata=/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/finetune_data/starcoderdata \
     --d0-file /orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/heldout/fineweb-edu-100b/d0.parquet \
     --pair-lr-fractions adamw:adamw=0.5 adamw:muon=1 muon:adamw=0.5 muon:muon=1 \
     --branch-tokens 3932160 62914560 125829120 209715200 --seeds 1 2 3 --time-limit 02:30:00 \
     --out-root /orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/finetune_main_long \
     --job-prefix ftml --out ~/jobs/ft-main-long
   ```
   (fractions are of the fine-tuning optimizer's 3B LR: 0.5 × 1.311e-3 = 6.55e-4, 1 × 8.874e-4.) Use a
   separate out-root: its 60/240/960 branches duplicate `finetune_main/` cells, and the analysis refuses
   duplicates. Analyse 1,920/3,200 with `--root finetune_pilot_long --root finetune_main_long` (seed-42
   selection from a `finetune_pilot_long` pilot analysis at that length, `--min-lrs 4`). Diff one generated
   sbatch against `~/jobs/ft-long2/` before submitting; ~40 min per job. Subagent review first if any code
   changed (§10).
4. **Read the completed local endpoint report** `reports/finetuning_mismatch_endpoints.md` and
   `reports/finetune_replay_bug_impact.md`. The older Claude pilot document and its exported snapshot
   describe the earlier phase; the new local report contains the main grid and both completed long
   pilots. Use the new report rather than treating the older snapshot as current.
5. Later: other E (sweep plan with interleaved grids and joint fit, not implemented; the transfer check
   at other E is a proposal: reuse E = 8's measured 0.5B→3B shift and check only at E = 16).

## 7. Open decisions (raise with the team; do not resolve silently)

- StarCoderData: Python only is agreed; the report text still says "86 languages" (the authors fix the
  wording; agents add `% NOTE` comments to `docs/report.tex`, not prose: the report states the authors
  did not use AI to draft it). D1 composition decided: best-fit, short files (protocol).
- WikiText beyond ~740 steps: multiple epochs are approved for the seed-42 long pilot. Decide whether
  to replicate the longer lengths on seeds 43/44 after examining the pilot; compare the corpora with
  the unequal effective epoch counts in mind.
- **Matched-learning targets and definitions resolved.** Endpoint 1.50/0.25 and within-run 1.00/0.12;
  main lengths are the completed replicated grids, with long WikiText separately conditional on seed 42.
  Formal document bootstrap and router/replay comparisons remain open. All six base checkpoints are
  pinned by SHA-256 in `FINETUNE_BASE_CHECKPOINTS`.
- Other E: the proposed design and decisions are in `reports/sparsity_extension_plan.md`.

## 8. Code map

| What | Where |
|---|---|
| Model, MoE, grouped-GEMM dispatch | `nanochat/gpt.py` |
| Pretraining | `scripts/base_train.py`; launcher `runs/pretrain_study.sh` |
| Sweep plan, LR selection, transfer check, Slurm jobs | `nanochat/pretrain_sweep.py`, `scripts/pretrain_sweep.py` |
| Fine-tuning data (WikiText-103, StarCoderData) | `nanochat/finetune_data.py`, `scripts/make_finetune_data.py` |
| Fixed eval sets with document IDs | `nanochat/eval_set.py` |
| Fine-tuning metrics | `nanochat/finetune_metrics.py` |
| Fine-tuning driver (cells, log checkpoints, decay branches) | `scripts/finetune.py` |
| Fine-tuning Slurm jobs (`--branch-tokens`, `--out-root`, `--job-prefix`, `--pair-lr-fractions`) | `scripts/finetune_jobs.py` |
| Fine-tuning analysis (`--bases`, `--lr-selection`, `--ell0`, bootstrap) | `scripts/analyze_finetune.py` |
| Pilot charts for the reports (→ `results/e8/figures/`) | `scripts/plot_finetune_pilot.py` |
| LR selections and data manifest | `results/e8/` |
| Tests | `tests/` (`.venv/bin/python -m pytest -q tests`) |

Tests: `tests/test_finetune.py`, `tests/test_pretrain_sweep.py` and `tests/test_analyze_finetune.py`
pass. Locally,
`tests/test_base_train_smoke.py` fails 6 tests with `ModuleNotFoundError: setuptools` (environment,
not code); on the B200 the full suite passed at b7edc71.

Output conventions (keep them): every run directory has `config.json` (resolved settings, git state,
data and eval-set SHA-256s) and `result.json` (status and summary). Fine-tuning writes
`metrics.jsonl`/`metrics.parquet` (one row per checkpoint × eval set × layer; layer −1 = whole model;
`ft_steps` = the run's length) and `ckpt_<step>.npz` (per-document sums, route tables).

## 9. Cluster notes

- Partition `mit_normal_gpu`, `--gres=gpu:b200:1`, max 6 h per job, 2 GPUs per user. B200 waits were
  minutes when under the limit; H200 waits are hours. Jobs ask for `-c 16 --mem=128G`.
- For fine-tuning, use idle GPUs beyond the 2-GPU limit with `sbatch -p mit_normal_gpu,mit_preemptable
  --requeue --nice=<N> job.sbatch`: finished cells are skipped on restart. **Pretraining** currently uses
  `mit_normal_gpu --no-requeue`: the launcher refuses an existing config for an interrupted run unless
  explicitly resumed, so blind preemption/requeue is not safe. Users can raise `--nice` (lower priority)
  to order their own jobs; `scontrol top` is denied.
- SSH from the local machine goes through a control socket
  (`-o ControlPath=$HOME/.ssh/cm_socket/orcd-%C hgo@orcd-login.mit.edu`, with `-n -o BatchMode=yes`). If
  it has died ("Permission denied"), the user must reopen it themselves:
  `ssh -fN -o ControlMaster=yes -o ControlPersist=yes -o ControlPath=$HOME/.ssh/cm_socket/orcd-%C hgo@orcd-login.mit.edu`.
- `.venv`: `module load miniforge/26.7.2-0; uv sync --extra gpu --group dev` (torch 2.9.1+cu128).
- Credentials: W&B key in `~/.netrc`, HF token in `~/.cache/huggingface/token` (mode 600). Never print
  or commit them.
- W&B tracking (Harry, 2026-11-26): existing project `hgo-mit/nanochat`; organized groups
  `optimizer-mismatch-e8/pretrain/{sweep,transfer,main}` and
  `optimizer-mismatch-e8/finetune/<dataset>/<stage>`. Fine-tuning historically disabled W&B; backfill
  saved finalized branch histories via `python -m scripts.log_finetune_wandb --root <root> --stage <root-basename>
  --query-existing`. Stable IDs are independent of where the run is mirrored; use the same stage/source
  across accounts for the same run, and a distinct source and `--invalid` for contaminated archives.
  Use `--verify-online` to check the explicit checkpoint `step` history; online reads can lag uploads.
  Future generated jobs upload at completion, with an explicit offline fallback. Do not rerun a model
  to repair tracking, and never upload raw document arrays or weights. See AGENTS.md.

## 10. Conventions for agents

- **Standing rule (user): before any cluster job runs new or changed code, a subagent bug-reviews and
  tests it.** Job scripts regenerated by the already-reviewed generator with only names, LRs or time
  limits changed do not need a new review (diff them against a reviewed one).
- LR selection is mechanical: lowest mean final D1 loss over the seeds; if it is a grid end, run the next
  point (edge rule) until interior. Seeds 43/44 never get their own grid.
- Commit with the attribution lines your session provides and push to `origin master`. Locally there is
  no git identity; commit with `git -c user.name=... -c user.email=...` as the user's setup requires.
- Record every decision with its date in `PRETRAINING_PROTOCOL.md`, every change and run outcome in
  `LOG_CHANGES.md`. Write prose plainly for humans.
- Reports: Markdown snapshots of the Claude artifacts are in `reports/` (README there lists sources and how
  current each is). Tracker dashboard: https://claude.ai/artifact/4tbXf1M5nneqzSYW4NEJYo (stale since 2026-11-24).
- Summary document of the pretraining (hyperparameters, sweep, transfer check, base models):
  https://claude.ai/artifact/TP3GZWWwJHK2EqVu3GHdyk (private to the user until shared).
