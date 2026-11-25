# Handoff: MoE optimizer-mismatch study (state on 2026-11-24, evening)

Read this first, then `PRETRAINING_PROTOCOL.md` (every pretraining decision, with dates),
`docs/report.tex` (the experiment spec as a paper draft) and `LOG_CHANGES.md` (what changed, newest last).
The original design spec, `EXPERIMENT_SPEC.md`, is not in the repo; `docs/report.tex` and the
protocol supersede it, and its fine-tuning recommendations (§6–§8) are implemented as defaults.

## 1. Goal

Measure whether fine-tuning a top-1 MoE language model with a different optimizer than it was
pretrained with (AdamW vs Muon) forgets more at matched learning, and how much of that is routing
(route replay). Current milestone, given by the user (who asked for autonomous work):

1. Finish the B200 throughput investigation.
2. Run the E = 8 learning-rate sweep for both optimizers, select the LRs.
3. Submit the E = 8 pretraining jobs (AdamW and Muon; seed 42 first, then 43 and 44).
4. While waiting on compute: build and test the fine-tuning infrastructure (done, CPU only).
5. If time remains: sweep LRs for the other E (1, 4, 16).
6. Keep logging consistent and persistent.

## 2. Agreed design (all recorded in PRETRAINING_PROTOCOL.md)

- Model: nanochat GPT, depth 10, width 640, top-1 Switch MoE of SwiGLU experts (hidden 3d) in every
  block, softmax gate over all E (never renormalized), dropless, router in fp32. GPT-2 tokenizer.
  118M active parameters incl. embeddings (53.2M non-embedding). E = 8 is the headline cell.
- Balancer: auxiliary loss 0.01 + z-loss 0.001, **summed** over MoE layers.
- Optimizers: AdamW β = (0.9, 0.95), ε = 1e-8; Muon "moonlight" (Nesterov 0.95, 5 Newton–Schulz steps,
  update scale 0.2·√max(m,n)); weight decay λ = 0.1 for both, applied as 1 − lr·λ (option A); clipping
  1.0 in every cell; routers use the run's optimizer with weight decay 0; embedding/lm_head always AdamW
  (nanochat values, identical in every cell).
- Pretraining: FineWeb-Edu (frozen 100-shard snapshot, `data_manifests/fineweb_edu_100_v1/`), 3,000,238,080
  tokens (11,445 steps) at 262,144 tokens per step, warmup 130M tokens (496 steps), constant, linear decay
  to 0 over the last 20%. One B200 per run, device batch 32 (4 micro-batches per step).
- Validation / LR selection: the held-out validation shard `shard_01822.parquet` **minus the documents D0
  took from it** (D0 is sampled from that shard). D0 (`heldout/fineweb-edu-100b/d0.parquet`, ~2M tokens)
  is evaluated alongside as `heldout/bpb` and never used for selection.
- E = 8 sweep: 0.5B tokens (1,907 steps), warmup 83 steps (4.33%, same fraction as 130M of 3B), AdamW at
  1e-3/2e-3/4e-3, Muon at 2e-3/4e-3/8e-3; LR = vertex of the parabola in log2(LR) through the best point and
  its neighbours (one more point ×2 or ×½ if the best is at an edge); ×(3B/0.5B)^-0.32 = 0.5635 for 3B.
- Other E (later): 5-point grids, AdamW centre 2e-3·E^-0.25, Muon centre 4e-3·2^(−k/4) (k = 0..3 for
  E = 1, 4, 8, 16); one joint fit per optimizer y = a_E + b(log2 LR − m0 − α log2 E)². **Not implemented.**
- Deferred until first fine-tuning results: transfer check (E=4, 0.5×/1×/2× at 3B), batch-size robustness,
  weight-decay ablation (Muon with λ = 0.1·η_AdamW/η_Muon), sweep seed-noise runs, 1B/2B decay branches
  (main runs checkpoint every 1,526 steps = 400M tokens so the branch points exist).
- GPU runs are not bitwise reproducible (accepted). Data order is identical across runs and cells.

## 3. Code map

| What | Where |
|---|---|
| Model, MoE, grouped-GEMM dispatch | `nanochat/gpt.py` (`MoEMLP`, `grouped_segment_ends`) |
| Pretraining | `scripts/base_train.py` (writes `config.json`, checkpoints, `result.json`; divergence stop) |
| Launcher (data hash checks, paths) | `runs/pretrain_study.sh` (`train`, `dry-run`, `paths`, ...) |
| E = 8 sweep plan, LR selection, Slurm jobs | `nanochat/pretrain_sweep.py`, `scripts/pretrain_sweep.py` |
| Throughput benchmark | `scripts/bench_moe_throughput.py` |
| Fine-tuning data | `nanochat/finetune_data.py`, `scripts/make_finetune_data.py` |
| Fixed eval sets with document IDs | `nanochat/eval_set.py` |
| Fine-tuning metrics (spec §2) | `nanochat/finetune_metrics.py` |
| Fine-tuning driver | `scripts/finetune.py` |
| Fine-tuning Slurm jobs | `scripts/finetune_jobs.py` |
| Tests | `tests/` (CPU: `.venv/bin/python -m pytest -q --deselect tests/test_execution.py::test_memory_limit`) |

Output conventions (keep them): every run directory has `config.json` (fully resolved settings, git
state, data and eval-set SHA-256s) and `result.json` (status + summary). Pretraining metrics stream to
`<scratch>/logs/<run>.jsonl`; fine-tuning to `<run>/metrics.jsonl` (+ `metrics.parquet`), one row per
(checkpoint, eval set, layer), layer −1 = whole model. Every job copies `*.json`, `*.jsonl`, `*.log`,
`*.parquet` to `$HOME/orcd/pool/nanochat-optimizer-mismatch/results/` at the end (weights stay on scratch).

## 4. Cluster (MIT ORCD) state

- Access: `ssh -o BatchMode=yes -o ControlPath=$HOME/.ssh/cm_socket/orcd-%C hgo@orcd-login.mit.edu ...`
  through a ControlMaster socket the **user** opens (Duo):
  `ssh -fN -o ControlMaster=yes -o ControlPersist=8h -o ControlPath=~/.ssh/cm_socket/orcd-%C hgo@orcd-login.mit.edu`.
  Check with `ssh -O check -o ControlPath=...`. Use `ssh -n` in loops.
- Partition `mit_normal_gpu`, `--gres=gpu:b200:1`, max 6 h per job, **2 GPUs per user**. B200 nodes are
  short of free CPUs; jobs ask for `-c 16 --mem=128G` (lower `-c` if they sit in `Resources`).
- User's checkout: `~/nanochat-optimizer-mismatch` (submit from here). `.venv` built with
  `module load miniforge/26.7.2-0; uv sync --extra gpu --group dev` (torch 2.9.1+cu128). GPT-2 tiktoken
  cache: `~/.cache/nanochat/tiktoken_cache`. **The checkout is at b7edc71: run `git pull` first.**
- Data (copied from Nazar's scratch, D0 verified against the manifest):
  `~/orcd/scratch/public/nanochat-optimizer-mismatch/base_data_fineweb_edu_100b/` (100 train shards +
  `shard_01822.parquet`) and `.../heldout/fineweb-edu-100b/{d0.parquet,d0_report.json}`.
  `bash runs/pretrain_study.sh paths` prints all paths.
- Job scripts: `~/jobs/` (`e8-sweep-adamw.sbatch`, `e8-sweep-muon.sbatch`, `bench-b200.sbatch`); Slurm
  output files land next to them. Scratch test outputs: `~/slurm_smoke/`, `~/fixtest/` (deletable).
- B200 GPU suite at b7edc71: 267 passed, 13 skipped (FA3, 2-GPU, gloo). `torch._grouped_mm` is native on
  sm_100 (no host sync), unlike the RTX 5080.

## 5. Progress

Done and pushed (master, latest d47c4a0): sweep/base-model infrastructure, validation-shard selection
without D0, `result.json`, divergence stop, persistence, B200 empty-expert fix, launcher `--` fix,
fine-tuning infrastructure (CPU-tested), report spec. CPU suite: 265 passed, 20 skipped.

Jobs submitted:
- `30414694` e8-sweep-adamw, `30414695` e8-sweep-muon: **FAILED** in 40 s (launcher bug, fixed in 3a654e4).
  Nothing was trained; no `config.json` exists, so the same run names can be reused.
- `30414697` bench-b200: submitted, outcome unknown (it does not use the launcher). Results append to
  `~/jobs/bench_b200.jsonl` (E = 8, 1, 4, 16 × AdamW/Muon × device batch 32 and 64).

MoE efficiency findings (RTX 5080 profiling, see conversation summary in LOG_CHANGES/protocol):
dropless costs nothing (≤0.2% padding); MoE overhead is mostly a fixed cost per micro-batch (expert
weight stack/cast/grad accumulation ∝ total params, ~21.6 ms on the 5080), so larger micro-batches
dilute it (expected ~1.08× dense at device batch 32 vs the measured 1.31× at 4); the Muon step scales
with total params (Newton–Schulz 4.7 TFLOP/step at E = 8); the data loader delivers ~3M tokens/s.
Optional later gains: device batch 64 (if memory allows), fusing dispatch copies (~1.5–2%).

## 6. Blockers

1. **The ORCD SSH socket expired** (2026-11-24 ~22:15). Ask the user to reopen it (command above).
2. Nothing else blocks the sweep.

## 7. Next steps (in order)

1. On ORCD: `cd ~/nanochat-optimizer-mismatch && git pull --ff-only`.
2. Check `sacct -j 30414697` and `~/jobs/bench_b200.jsonl`: compare `ms_per_step`/`x_dense` at device batch
   32 vs 64 for E = 8 (both optimizers) and record the result (LOG_CHANGES + dashboard A9). If 64 fits and is
   ≥ 5% faster, change `DEVICE_BATCH_SIZE` in `nanochat/pretrain_sweep.py` for the **main** runs (numerically
   equivalent; the sweep can stay at 32). Redo the wall-clock estimate from measured tokens/s.
3. Regenerate and submit the sweep:
   `.venv/bin/python -m scripts.pretrain_sweep jobs --stage sweep --out ~/jobs && cd ~/jobs && sbatch e8-sweep-adamw.sbatch && sbatch e8-sweep-muon.sbatch`.
4. A few minutes after the first run starts, check `~/orcd/scratch/public/nanochat-optimizer-mismatch/logs/e8-sweep-adamw-lr0.001-s42.log`:
   finite "Validation bpb" and "Held-out bpb" lines at step 0, tok/sec, and in
   `base_checkpoints/<run>/config.json`: `data.val_data_file` ends in `shard_01822.parquet`,
   `data.heldout_data_file` in `d0.parquet`, `data.val_excluded_heldout_rows` > 0 (≈ D0's documents
   from that shard; D0 has 1,881 documents). Don't wait on a stuck job: if a job sits in `Resources` for
   long, lower `-c`/`--mem`; if a run crashes, read the log and fix before resubmitting.
5. When both sweep jobs finish: `.venv/bin/python -m scripts.pretrain_sweep select --out ~/jobs`
   (exit 3 = an optimizer needs one more point; rerun `jobs --stage sweep --extra-lr <opt>=<lr>` and `select`
   with the same `--extra-lr`). Sanity-check the curves (`result.json` `eval_history`) and that no run diverged
   unexpectedly.
6. Main runs: `.venv/bin/python -m scripts.pretrain_sweep jobs --stage main --selection ~/jobs/lr_selection.json --out ~/jobs --seeds 42 43 44`;
   submit seed 42 (both optimizers) first, then 43 and 44 (the 2-GPU limit queues them). ~1.4–1.5 h each.
7. While they run: prepare fine-tuning data on the login node (network needed; untested against the real
   Hugging Face endpoints):
   `.venv/bin/python -m scripts.make_finetune_data --dataset wikitext103 --out ~/orcd/scratch/public/nanochat-optimizer-mismatch/finetune_data/wikitext103`
   and the same with `--dataset starcoderdata` (gated; reads the HF token). Check `manifest.json` (document counts, detokenized text sample).
8. GPU smoke of `scripts/finetune.py` before the base models finish: use a finished **sweep** run's final
   checkpoint as `--base-run` (e.g. `e8-sweep-adamw-lr0.002-s42`), 1 LR × 1 seed, to time a cell and check
   outputs; delete that output afterwards. Expected ~1–2 min per cell (15.7M tokens + 10 evaluations).
9. When the seed-42 pair is done: `.venv/bin/python -m scripts.finetune_jobs --selection ~/jobs/lr_selection.json --base-runs e8-main-adamw-s42 e8-main-muon-s42 --dataset wikitext103=<dir> --dataset starcoderdata=<dir> --d0-file <d0> --out ~/jobs`
   and submit (8 jobs; 25 cells each).
10. If time remains: other E. Needs (a) a sweep plan for E ∈ {1, 4, 16} with the interleaved grids above,
    (b) the joint fit (not implemented; `nanochat/pretrain_sweep.select_lr` is the E = 8 three-point version).
11. Still to build: the analysis script (F at matched learning ℓ0 with interpolation between checkpoints,
    mismatch penalty P, routing share, paired document bootstrap over seeds). Fine-tuning measurements are
    already stored per document for it.

## 8. Open decisions to raise with the team (do not resolve silently)

- Far fine-tuning dataset: **decided 2026-11-25: StarCoderData** (Python subset by default; codeparrot-clean
  removed). Whether to use more languages is open.
- Fine-tuning defaults (spec recommendations, `scripts/finetune.py`): 15.7M tokens at 65,536 per step, 3%
  warmup, 20% decay, weight decay 0, embedding/lm_head LR = 0.1× pretraining values in every cell, LR grid
  1/16…1 × the selected pretraining LR of the fine-tuning optimizer, 5 seeds, 10 checkpoints, balancer active
  (`--balancer-mode frozen` for the control arm). ℓ0 not chosen.
- Fine-tuning stores per-document loss sums (not per-token arrays) for the bootstrap.

## 9. Conventions for agents

- Commit with the attribution lines the session provides; push to `origin master` (the user asked for
  autonomous work and the cluster pulls from GitHub). Never commit secrets.
- Keep the tracker dashboard current: https://claude.ai/artifact/4tbXf1M5nneqzSYW4NEJYo (ArtifactData,
  collections `data/tracker/items`, `data/tracker/implemented`, `data/tracker/plan`; pin writes with
  `if_version`). Open items of note: A9 (benchmark), A24 (E = 8 sweep and base models).
- Write prose for humans plainly; record every decision with its date in PRETRAINING_PROTOCOL.md.
