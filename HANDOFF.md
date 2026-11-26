# Handoff: MoE optimizer-mismatch study (state on 2026-11-25, 18:40 EST)

Written for the agents taking over on Nazar's compute. Read this first, then
`PRETRAINING_PROTOCOL.md` (every decision with its date, including the fine-tuning decisions near
the end), `LOG_CHANGES.md` (what changed and every run's outcome, newest last) and `docs/report.tex`
(the paper draft; `% NOTE` lines mark what is not yet agreed). The original `EXPERIMENT_SPEC.md` is
not in the repo; the protocol and report supersede it.

## 1. Goal and where we are

Measure whether fine-tuning a top-1 MoE language model with a different optimizer than it was
pretrained with (AdamW vs Muon) forgets more at matched learning, and how much of that is routing
(route replay). E = 8 is the headline cell.

| Stage | State |
|---|---|
| E = 8 LR sweep (0.5B tokens) | Done |
| LR transfer check at 3B tokens | Done; AdamW moved to the 3B optimum |
| E = 8 base models, AdamW and Muon × seeds 42, 43, 44 (3B tokens) | **All six done** |
| Fine-tuning data | **Both built** 2026-11-25: StarCoderData (Python); WikiText-103 rebuilt with a 1.1M-token D1 |
| Fine-tuning protocol | LR selection, checkpoints, length pilot **decided** 2026-11-25 (§4) |
| Fine-tuning length pilot (StarCoderData) | **Done**, all four pairs plus edge extensions; seed-42 LR selection final (LOG_CHANGES 2026-11-25; `results/e8/finetune_pilot_starcoderdata_summary.json`) |
| Fine-tuning length pilot (WikiText-103) | **Assigned to Nazar** (2026-11-25): all four pairs, seed-42 bases, 60 / 240 / 720 steps (§5) |
| Fine-tuning main grid | Waits for the WikiText pilot, the main-length decision and ℓ0 |
| Analysis script | **Built** (`scripts/analyze_finetune.py`, §6 step 6); needs all runs on one evaluation packing (best-fit) |
| Other E (1, 4, 16) | Deferred until fine-tuning results |

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

- Scratch root (`$SCRATCH` in our jobs): `/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/`
  - `base_checkpoints/<run>/` — model weights, `config.json`, `result.json` for every sweep, transfer
    and base run. From another account, point `scripts/finetune.py --base-checkpoints-dir` here.
  - `base_data_fineweb_edu_100b/`, `heldout/fineweb-edu-100b/d0.parquet` (D0).
  - `finetune_data/wikitext103/`, `finetune_data/starcoderdata/` (each: `train_*.parquet`, `val_00000.parquet` = D1,
    `manifest.json`; the whole `finetune_data/` is world-writable at the user's request). `finetune_data/codeparrot/` is obsolete (deletable).
  - `logs/<run>.jsonl|.log` — pretraining logs.
- Pool copy of all small outputs: `/orcd/pool/006/hgo/nanochat-optimizer-mismatch/results/`.
  Note: `meta_*.json` files there are 0.4–0.8 GB each (12 GB total); worth slimming.
- Job scripts and Slurm outputs: `/home/hgo/jobs/` (pilot scripts in `/home/hgo/jobs/ft-pilot/`).
- hgo checkout: `/home/hgo/nanochat-optimizer-mismatch` at 756a9fe or later.

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

## 5. Pilots: StarCoderData done, WikiText assigned to Nazar

> **Decay-branch replay bug (found 2026-11-25, fixed in the commit after d704870).** Before the fix,
> `ReplayLoader` kept references to the loader's reused buffer, so after each branch the trunk
> replayed the branch's *last* batch in place of all of its batches. With `--branch-tokens`, the first
> (shortest) branch is correct; every longer length is not: at 60/240/960 (or 720), the 240-step runs
> trained steps 49–60 on step 60's batch, and the longest runs also steps 193–240 on step 240's batch.
> Runs without `--branch-tokens` (e.g. single 720-step runs) are unaffected. Do not launch branched
> runs from a commit before the fix; pilot results at 240 and 960/720 steps are pending a decision on
> reruns (user).

### StarCoderData pilot (done 2026-11-25)

All four pairs on the seed-42 bases, 60 / 240 / 960 steps, 5-point grid plus 2× and 4× extensions
where the best LR hit the top edge, 3 seeds; 78 cells, all completed, all on the best-fit evaluation.
Selected LRs and every cell's mean learning / forgetting: LOG_CHANGES ("Seed-42 StarCoderData
learning-rate selection final") and `results/e8/finetune_pilot_starcoderdata_summary.json`. Report
for the team: https://claude.ai/artifact/2uN17JfULvGvtcTFzYUgKx. Outputs:

- hgo (AdamW → Muon, Muon → AdamW): `/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/finetune_pilot/`
- Nazar (AdamW → AdamW, Muon → Muon): `/orcd/scratch/orcd/010/ndrugov/finetune_pilot/`

What it taught for WikiText: the first 5-point grid put the best LR at the top edge for Muon
fine-tuning in every case and for AdamW → AdamW at short lengths; AdamW → AdamW at 4× (5.24e-3)
diverged. One 960-step trunk with its branches took ~4.8 min on a B200; one job (5 LRs × 3 seeds)
~75 min; seeds agree to ≤ 0.006 nats in learning.

### WikiText pilot: Nazar runs all four pairs (user decision, 2026-11-25)

Same conventions as the StarCoderData pilot, on Nazar's account. Nothing has been run on WikiText yet
by either account.

1. **Code:** commit ed76421 or later (`git pull`). It includes the best-fit evaluation, the
   seed-42 checkpoint pin and `PYTHONUNBUFFERED=1` in generated jobs.
2. **Data** (built, readable and writable by all users; do not rebuild):
   `/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/finetune_data/wikitext103/`
   (training 28,934 articles, 110.6M tokens; D1 = `val_00000.parquet`, 632 articles: validation + test +
   510 sampled training articles; manifest in `results/e8/wikitext103_manifest.json`).
3. **Generate the four jobs** (5-point grid, seeds 1–3, lengths 60 / 240 / 720 steps):
   ```
   .venv/bin/python -m scripts.finetune_jobs --selection results/e8/lr_selection_3b.json \
     --base-runs e8-main-adamw-s42 e8-main-muon-s42 \
     --dataset wikitext103=/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/finetune_data/wikitext103 \
     --d0-file /orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/heldout/fineweb-edu-100b/d0.parquet \
     --seeds 1 2 3 --branch-tokens 3932160 47185920 \
     --out-root /orcd/scratch/orcd/010/ndrugov/finetune_pilot --job-prefix ftpilot --out ~/jobs/ft-pilot-wikitext
   ```
   Keep the default `--train-tokens` (15,728,640): it fixes warmup at 8 steps, so the 60- and
   240-step runs match the StarCoderData pilot's. **720, not 960, steps:** best-fit cropping gives only
   23,924 rows per pass over WikiText's training articles; 720 steps use 23,040 (one pass), 960 would
   repeat 22% of the data during the decay. Check `--exclude` for your account's bad nodes (ours:
   node5800-c1, node5702-c1). Submit all four (`sbatch ~/jobs/ft-pilot-wikitext/*.sbatch`); with
   2 GPUs per user they run in two waves. Expected ~3.7 min per 720-step trunk, ~1 h per job,
   ~4 GPU-hours, ~2 h of wall time.
4. **Check the first job early:**
   - `config.json` of any run: `data.eval_sets.D0.packing` and `D1.packing` are `"bestfit"`;
     `branch.lengths` is `[60, 240, 720]`; `warmup_steps` is 8.
   - `<out-root>/<base>/wikitext103/base/base_metrics.json`: base D0 loss ≈ 2.94 (AdamW s42) /
     ≈ 2.90 (Muon s42), as in the StarCoderData pilot (same D0 set).
   - The D1 set should score 1,048,461 tokens from 627 articles (`config.json`,
     `data.eval_sets.D1.scored_tokens` / `documents`).
5. **Edge rule** (protocol): for each (pair, length), the selected LR = lowest mean final D1 loss over
   the 3 seeds. If it is the top grid point, run `--lr-fractions 2 4`; if the bottom,
   `--lr-fractions 0.03125 0.015625`; same arguments otherwise, `--job-prefix ftpilot-ext`, and submit
   only the pair that needs it. Repeat until every selection is interior. Given the StarCoderData
   result, expect the Muon pairs (and maybe AdamW → AdamW) to need the upward extension.
6. **Analyse** (CPU, no GPU needed), once per length:
   ```
   .venv/bin/python -m scripts.analyze_finetune --root /orcd/scratch/orcd/010/ndrugov/finetune_pilot \
     --dataset wikitext103 --steps <60|240|720> --seeds 1 2 3 --out /orcd/scratch/orcd/010/ndrugov/finetune_pilot/analysis/wikitext103/T<steps>
   ```
   No `--ell0` yet: ℓ0 is chosen by the user after seeing the curves (protocol, "Fine-tuning main grid").
7. **Record** in LOG_CHANGES (jobs, times, per-cell mean learning / forgetting, selected LR per pair
   and length, any divergence) and save a summary like
   `results/e8/finetune_pilot_starcoderdata_summary.json` as `results/e8/finetune_pilot_wikitext103_summary.json`.
   Update this file's status table. Do not change the evaluation or data between jobs: all WikiText
   cells must share one D1 identity and packing (the analysis refuses mixes).

## 6. Next steps (in order)

1. **Nazar: WikiText pilot** (§5), with edge extensions until every selected LR is interior.
2. **User decisions** once both pilots are in: the main fine-tuning length (pilot lengths, or add a
   longer StarCoderData length; WikiText cannot exceed ~740 steps) and ℓ0 (from the learning curves,
   before computing the forgetting contrast; protocol "Fine-tuning main grid").
3. **Extend the checkpoint pin** (`FINETUNE_BASE_CHECKPOINTS` in `nanochat/pretrain_sweep.py`) to the
   seed-43 and seed-44 base models, with their SHA-256 (`model_011445.pt` in each
   `base_checkpoints/e8-main-{adamw,muon}-s{43,44}/`).
4. **Main grid** on seeds 43 and 44: both datasets, all four pairs, 3 fine-tuning seeds, only the LR(s)
   selected on seed 42 for each pair, dataset and length (one trunk per selected LR gives all lengths;
   run each distinct selected LR once). Seed 42 reuses the pilot runs.
5. **Analysis** across all output roots (`scripts/analyze_finetune.py`, repeated `--root`): forgetting at
   matched learning ℓ0 with the paired seed/document bootstrap, mismatch contrasts, routing share. It
   requires one evaluation packing and identical D0/D1 identity across roots; runs from before
   0c86b24 (contiguous packing) cannot be mixed in.
6. Later: other E (sweep plan with interleaved grids and joint fit, not implemented; the transfer check
   at other E is a proposal: reuse E = 8's measured 0.5B→3B shift and check only at E = 16).

Background for step 2 of §6 in earlier versions (kept for reference): **D0 loss gap explained and fixed
2026-11-25** (LOG_CHANGES, `results/e8/d0_eval_gap/`): the old contiguous evaluation cut the context of
~49% of scored tokens (~0.2 of the ~0.24-nat gap); the evaluation now uses the pretraining best-fit rows.

## 7. Open decisions (raise with the team; do not resolve silently)

- StarCoderData: Python only is agreed; the report text still says "86 languages" (the authors fix the
  wording; agents add `% NOTE` comments to `docs/report.tex`, not prose: the report states the authors
  did not use AI to draft it). D1 composition decided: best-fit, short files (protocol).
- WikiText beyond ~740 steps (only if the main length goes past that): accept a shorter WikiText length
  or split long articles.
- Main fine-tuning length (pilot lengths vs adding a longer one) and ℓ0. Decided 2026-11-25: 3 fine-tuning seeds,
  all three base seeds, seeds 43/44 reuse the seed-42 LRs, ℓ0 after the curves (protocol, "Fine-tuning main grid").
  The seed-42 checkpoint pin in `nanochat/pretrain_sweep.py` must admit seeds 43/44 (with SHA-256) first.
- Transfer check at other E (proposal above).

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
| Fine-tuning Slurm jobs (`--branch-tokens`, `--out-root`, `--job-prefix`) | `scripts/finetune_jobs.py` |
| LR selections and data manifest | `results/e8/` |
| Tests | `tests/` (`.venv/bin/python -m pytest -q tests`) |

Tests: `tests/test_finetune.py` and `tests/test_pretrain_sweep.py` pass (32). Locally,
`tests/test_base_train_smoke.py` fails 6 tests with `ModuleNotFoundError: setuptools` (environment,
not code); on the B200 the full suite passed at b7edc71.

Output conventions (keep them): every run directory has `config.json` (resolved settings, git state,
data and eval-set SHA-256s) and `result.json` (status and summary). Fine-tuning writes
`metrics.jsonl`/`metrics.parquet` (one row per checkpoint × eval set × layer; layer −1 = whole model;
`ft_steps` = the run's length) and `ckpt_<step>.npz` (per-document sums, route tables).

## 9. Cluster notes

- Partition `mit_normal_gpu`, `--gres=gpu:b200:1`, max 6 h per job, 2 GPUs per user. B200 waits were
  minutes when under the limit; H200 waits are hours. Jobs ask for `-c 16 --mem=128G`.
- `.venv`: `module load miniforge/26.7.2-0; uv sync --extra gpu --group dev` (torch 2.9.1+cu128).
- Credentials: W&B key in `~/.netrc`, HF token in `~/.cache/huggingface/token` (mode 600). Never print
  or commit them.

## 10. Conventions for agents

- Commit with the attribution lines your session provides and push to `origin master`.
- Record every decision with its date in `PRETRAINING_PROTOCOL.md`, every change and run outcome in
  `LOG_CHANGES.md`. Write prose plainly for humans.
- Tracker dashboard: https://claude.ai/artifact/4tbXf1M5nneqzSYW4NEJYo (may be stale since 2026-11-24).
- Summary document of the pretraining (hyperparameters, sweep, transfer check, base models):
  https://claude.ai/artifact/TP3GZWWwJHK2EqVu3GHdyk (private to the user until shared).
