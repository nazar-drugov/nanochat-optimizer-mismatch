# Handoff: MoE optimizer-mismatch study (state on 2026-11-25, 14:00 EST)

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
| Fine-tuning length pilot (StarCoderData) | **4 jobs queued on the hgo account, not started** (§5) |
| Fine-tuning main grid | Waits for the pilot (length, seeds) |
| Analysis script | Not built |
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

- Datasets: near = WikiText-103 (detokenized articles; D1 = official validation + test + 146 training
  articles sampled with seed 0 and removed from training: 268 articles, 1,100,545 tokens, shuffled;
  training 29,298 articles, ~112M tokens); far = **StarCoderData, Python only**
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

## 5. In flight, and what Nazar runs

The pilot is split between the two accounts (user decision, 2026-11-25 14:15 EST):

| Job | Base model → fine-tuning optimizer | Account | State |
|---|---|---|---|
| 30493469 | e8-main-adamw-s42 → Muon | hgo | running since 13:49 EST, ends ~15:05 |
| 30493470 | e8-main-muon-s42 → AdamW | hgo | running since 13:51 EST, ends ~15:05 |
| — | e8-main-adamw-s42 → AdamW | **Nazar** | to run (hgo's copy, 30493471, was cancelled) |
| — | e8-main-muon-s42 → Muon | **Nazar** | to run (hgo's copy, 30493472, was cancelled) |

hgo outputs: `/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/finetune_pilot/<base>/starcoderdata/<opt>-lr<lr>-s<seed>-T{60,240,960}/`
(Slurm logs in `/home/hgo/jobs/ft-pilot/`). The decay-branch code works on the B200: every finished
trunk wrote all three lengths with the expected checkpoints (8, 10, 12) and status "completed".
Measured: ~4.8 min per 960-step trunk (with its 60- and 240-step branches) on one B200, so one job
(5 LRs × 3 seeds) takes ~75 min. Seeds agree to ~0.01 nats in learning and forgetting.

To run the two matched pairs on Nazar's account, generate the jobs and submit only the
`-adamw-s42-starcoderdata-adamw` and `-muon-s42-starcoderdata-muon` scripts:

```
.venv/bin/python -m scripts.finetune_jobs --selection results/e8/lr_selection_3b.json \
  --base-runs e8-main-adamw-s42 e8-main-muon-s42 \
  --dataset starcoderdata=/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/finetune_data/starcoderdata \
  --d0-file /orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/heldout/fineweb-edu-100b/d0.parquet \
  --seeds 1 2 3 --branch-tokens 3932160 62914560 --out-root <Nazar's scratch>/finetune_pilot \
  --job-prefix ftpilot --out <jobs dir>
```

Then add `--base-checkpoints-dir /orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/base_checkpoints`
to the `scripts.finetune` line of each script (`finetune_jobs` has no flag for it yet), or copy the two
base checkpoints. Check `--exclude` for your account's bad nodes (ours: node5800-c1, node5702-c1).
Job scripts generated from commit bd74937 on set `PYTHONUNBUFFERED=1`, so progress lines reach the Slurm log
live (hgo's two running jobs predate it: their logs fill only at the end; results stream to scratch).
The analysis must read both output roots (hgo's for the mismatched pairs, Nazar's for the matched).
The WikiText pilot (all four pairs) is not assigned yet.

## 6. Next steps (in order)

1. Run the two matched-pair pilot jobs on Nazar's account (§5).
2. **D0 loss gap: explained 2026-11-25** (LOG_CHANGES, `results/e8/d0_eval_gap/`). The fine-tuning
   evaluation (`nanochat/eval_set.py`) packs documents contiguously, so ~49% of scored tokens sit in a
   document that started in an earlier row and have lost their earlier context (+0.45 nats on those
   tokens); that is ~0.2 of the ~0.24-nat gap to the pretraining (best-fit) evaluation. A fix (score
   document-aligned rows, e.g. reproduce the best-fit rows with document IDs) is **awaiting the user's
   decision**. Changing the evaluation changes every fine-tuning measurement, so all four pilot pairs
   must use the same version: do not start the matched-pair pilot jobs until this is decided.
3. Analyse the pilot: per (base, fine-tuning optimizer, length) the D1-selected LR, learning and
   forgetting curves over log-spaced checkpoints, seed spread; record the chosen main length and number
   of seeds in the protocol with the date.
4. WikiText (D1 decided and built 2026-11-25): run the length pilot on WikiText the same way
   (`--dataset wikitext103=<scratch>/finetune_data/wikitext103`), or fold it into the main grid if the
   StarCoderData pilot already settles length and seeds.
5. Main grid: `scripts.finetune_jobs --selection results/e8/lr_selection_3b.json --base-runs e8-main-{adamw,muon}-s{42,43,44} ...`
   at the length the pilot picks. Pilot cells at that length can be reused for seeds 1–3 of the
   seed-42 bases if the length is one of the pilot lengths (same warmup, same data order).
6. Build the analysis script: forgetting at matched learning ℓ0 (interpolated between checkpoints),
   mismatch penalty, routing share (F_route / forgetting), paired document bootstrap over seeds. ℓ0 is
   not chosen yet. Per-document sums are stored in `ckpt_<step>.npz` for the bootstrap.
7. Later: other E (sweep plan with interleaved grids and joint fit, not implemented; the transfer check
   at other E is a proposal: reuse E = 8's measured 0.5B→3B shift and check only at E = 16).

## 7. Open decisions (raise with the team; do not resolve silently)

- StarCoderData: Python only is agreed; the report text still says "86 languages" (fix the wording).
- ℓ0 for forgetting at matched learning; the number of fine-tuning seeds and main length (from the pilot).
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
