# MoE Mismatch Tracker (snapshot)

Snapshot of the tracker dashboard https://claude.ai/artifact/4tbXf1M5nneqzSYW4NEJYo, exported 2026-11-26.
The tracker was last updated on 2026-11-24 (latest resolved item), before any cluster run, and is **stale** (e.g. every experiment stage still says "not started"). For the current state see `HANDOFF.md` and `LOG_CHANGES.md`.

## Experiment stages

| # | Stage | Runs | Status | Waiting on |
|---|---|---|---|---|
| 1 | Throughput benchmark | 8 (4 E × 2 optimizers) | not started | Your OK to submit on ORCD |
| 2 | Pretraining LR sweep | 40 × 0.5B tokens (Option A) + 6 transfer checks at 3B | not started | Throughput benchmark · data preparation (A19) |
| 2.5 | Batch-size robustness | 6 × 0.5B tokens (E = 4, both optimizers × 128K/256K/512K) | not started | LR sweep |
| 3 | Base models | 8 at 3B tokens, plus 2 extra seeds per optimizer at E = 8 | not started | LR sweep · A10, A13 |
| 4 | Fine-tuning | Up to 800 (8 bases × 2 optimizers × 2 datasets × 5 LRs × 5 seeds) | not started | Parked |
| 5 | Evaluation and route replay | Every fine-tuning checkpoint | not started | Parked |

## Implementation plan

| # | Phase | Status | Detail |
|---|---|---|---|
| 1 | Strip nanochat to a plain transformer | done | Speedrun tricks and FP8 removed; dense regression tests added. |
| 2 | MoE layer, optimizer choice, budgets and logging | done | MoE layer, router optimizer, balancers, budgets and logging. A1, A3 |
| 3 | Align the code with the spec | done | Implemented in 0ed4cd4; inferred defaults await confirmation (A17). A17 |
| 4 | Correctness gate | in progress | Local tests pass. Remaining: throughput and the 2-GPU test on B200s, and data preparation. A9, A19 |
| 5 | Pretraining: LR sweeps, then base models | not started | Learning-rate sweep (Option A), then base models. Ready once phase 4 is done. |
| 6 | Fine-tuning pipeline and runs | blocked | Parked by decision. A14, A15 |
| 7 | Analysis: mismatch penalty, routing response, route replay | blocked | Parked by decision. A16 |

## Implemented

| # | What | Where | Detail |
|---|---|---|---|
| 1 | Routers use the run's optimizer | nanochat/gpt.py | --router-optimizer same\|adamw\|muon, default same. The router learning rate follows the matrix learning rate. |
| 2 | Muon variants | nanochat/optim.py | plain (default): Moonlight 0.2·√max(m,n) scaling, decoupled weight decay, constant momentum 0.95. keller kept for the earlier runs; nanochat's variant dropped. |
| 3 | Gradient clipping | nanochat/optim.py | Global norm 1.0 on the averaged gradient, correct across GPUs; the pre-clip norm is logged. Fixed a bug where clipping was undone on sharded tensors in multi-process runs. |
| 4 | Load balancers | --balancer | aux (0.01 + z-loss 0.001, default), bias (changes selection only), or none. Loads, MaxVio and the bias are logged per layer. |
| 5 | FineWeb-Edu data and held-out D₀ | nanochat/dataset.py · scripts/make_heldout.py | FineWeb-Edu-100B is the default dataset. make_heldout builds a ~2M-token held-out set with near-duplicates of training documents removed (MinHash). |
| 6 | GPT-2 tokenizer | --tokenizer gpt2 | Default for pretraining and recorded in every checkpoint. Depth-10 is now 118M active, the same row as Ludziejewski et al. |
| 7 | Schedule and batch | scripts/base_train.py | 256K-token batch; 130M-token warmup, constant learning rate, linear decay to 0 over the last 20%. |
| 8 | MoE options and analysis hooks | nanochat/gpt.py | --moe-layers; a dense SwiGLU model identical to E=1; route replay and route capture; cumulative expert counts. |
| 9 | Run records | nanochat/train_config.py | config.json holds every setting, parameter counts, optimizer groups, git commit, tokenizer and hardware. Checkpoints carry data fingerprints. |
| 10 | Tests | tests/ | After rebasing onto the data-prep branch: 248 pass on an RTX 5080 (Linux, CUDA) and 231 on macOS CPU: spec §8 checks 1–7, Muon parity with torch.optim.Muon, a 2-process test, grouped-vs-loop MoE equivalence, launcher data checks, and end-to-end runs showing identical data across optimizers. On macOS the 4 launcher tests fail (GNU find -printf; the launcher targets Linux). Resume is bit-exact on CPU; GPU training is not bitwise reproducible (A20). |
| 11 | Blackwell attention fallback | nanochat/flash_attention.py | FA3 loaded on B200/RTX 50xx GPUs and crashed the process on its first call. Fixed independently on both branches; the merged code keeps Nazar's version (FA3 only on sm 8.0/8.6/8.9/9.0, NANOCHAT_ATTENTION_BACKEND=sdpa override) with our test extended to sm_120 and sm_75. The gpu install also gets setuptools. |
| 12 | Grouped-GEMM MoE dispatch | nanochat/gpt.py (MoEMLP) | Experts run in one torch._grouped_mm per projection instead of a per-expert loop that broke torch.compile. Same routing and maths; the model compiles as one graph. 5080, depth 10: 1.13–1.56× dense (was 1.36–2.36× up to E=8, 6.2× at E=16). --moe-dispatch auto\|grouped\|loop. |
| 13 | Balance and z losses summed over layers | nanochat/gpt.py (GPT.forward) | Per-layer losses were averaged, which made each layer's coefficient 10× weaker at depth 10. Now summed, as in Switch and Ludziejewski et al.'s released code (Nazar, 0ed5322). |
| 14 | Frozen data and cluster launcher | runs/pretrain_study.sh · data_manifests/ | Data manifest with SHA-256 for every shard and D₀, checked on each launcher start and resume; resume also checks the shard list, tokenizer files and validation file. runs/pretrain_study.sh stages data to scratch and launches runs (Nazar, 3cf4ef7). |
| 15 | E=8 sweep and base-model infrastructure | scripts/pretrain_sweep.py · scripts/base_train.py | Validation on the held-out validation shard minus D0's documents (D0 is sampled from that shard); D0 evaluated alongside and only logged. result.json per run, divergence stop, sweep planner/LR selection/Slurm jobs (scripts/pretrain_sweep.py). CPU suite: 253 passed. Not yet committed. |
| 16 | Fine-tuning infrastructure | scripts/finetune.py | scripts/finetune.py runs every (LR, seed) cell per base model, dataset and optimizer from the base weights with fresh optimizer state and paired data order; measures learning, forgetting, replay terms, flip rates, TV/churn, flip causes, rho and tokens per expert on D0/D1 per checkpoint. Data prep for WikiText-103 and codeparrot; Slurm job planner; results copied to pool. CPU-tested only (commit d47c4a0). |
| 17 | B200 fixes | nanochat/gpt.py · runs/pretrain_study.sh | Empty-expert segments gave garbage weight gradients with the native B200 grouped GEMM (fixed, b7edc71; B200 suite 267 passed). The launcher's torchrun call swallowed --run (fixed with --, 3a654e4). |

## Action items: open

| ID | Item | Kind | Status | Spec | Where | Detail | Resolution |
|---|---|---|---|---|---|---|---|
| A24 | Run the E=8 mini-sweep, then the E=8 base models | measure | open | LR sweep (Option A) | PRETRAINING_PROTOCOL.md | First sweep submission (jobs 30414694/5) failed on the launcher bug, now fixed; nothing trained. Data copied to Harry's scratch and verified. Blocked: ORCD SSH session expired, needs the user to reopen it (Duo). Then: git pull on ORCD, regenerate and submit the sweep, select, submit E=8 main runs (seeds 42, 43, 44). See HANDOFF.md. |  |
| A19 | Prepare data on the cluster | measure | open | §5 Data | runs/pretrain_study.sh | Data frozen in data_manifests/fineweb_edu_100_v1 (100 train shards, val shard, D₀, SHA-256s) and the ORCD launcher runs/pretrain_study.sh (Nazar). Remaining: the one-step 2-GPU check that training reads D₀ (top of the launcher script). |  |
| A9 | Benchmark throughput per E and optimizer on B200 | measure | open | §8.8 · §5 Compute | scripts/bench_moe_throughput.py | Needs your OK to submit on ORCD. Run scripts/bench_moe_throughput.py (grouped and loop, both optimizers, E=1–16) and the 2-GPU CUDA optimizer test. Check that torch._grouped_mm has a native kernel on sm_100 (on the 5080 it falls back with a host sync). On the 5080, grouped dispatch costs 1.13–1.31× dense up to E=8 and 1.56× at E=16; B200 wall-clock estimates need redoing from these numbers. |  |
| A14 | Fine-tuning script (parked) | build | open | §6 | scripts/chat_sft.py | Parked by decision. Data preparation for WikiText-103 and codeparrot is built and tested on a side branch. Open question: fine-tuning schedule (decay branches recommended). |  |
| A15 | Choose far dataset, fine-tuning budget, LR grid and ℓ₀ | decide | open | Decisions 13, 15, 16 |  | Spec recommendation: codeparrot-clean, 10–20M tokens, 5 learning rates, ℓ₀ reachable by every cell. |  |
| A16 | Evaluation and replay pipeline (parked) | build | open | §7 | nanochat/routing_probe.py | Parked by decision. The model hooks it needs (route replay, capture, counts) are in place. |  |

## Action items: resolved

| ID | Item | Kind | Status | Spec | Where | Detail | Resolution |
|---|---|---|---|---|---|---|---|
| A17 | Confirm inferred defaults | decide | done | Decisions 1, 4, 5 | PRETRAINING_PROTOCOL.md | Still to confirm: weight decay 0.1 on Muon matrices too (see A21); clipping in Muon runs too; router LR tied to the matrix LR, router weight decay 0; AdamW ε 1e-8. The aux balancer (0.01 + z 0.001, summed over layers) is now decided. | AdamW β = (0.9, 0.95), ε = 1e-8 (Hernanz's recipe); clipping 1.0 in Muon cells too; router LR tied to the matrix LR, router weight decay 0; init kept per Nazar's plan. (2026-11-24) |
| A21 | Decide how weight decay compares across optimizers | decide | done | Decision 5 | PRETRAINING_PROTOCOL.md | Decay is applied as 1 − η·λ, so with λ = 0.1 everywhere its strength follows the LR. At the sweep centres after transfer, Σηλ over 3B tokens ≈ 0.67 (AdamW, E=8) vs 2.27 (Muon): a 3.4× difference in decay. Options: keep λ = 0.1 (Moonlight) and log weight norms plus an E=4 ablation with matched η·λ, or match η·λ in every cell. | Option A: λ = 0.1 for both optimizers. Log per-layer weight norms; one E=4 Muon 3B ablation with η·λ matched to AdamW. (2026-11-24) |
| A22 | Decide the warmup in LR-sweep runs | decide | done | LR sweep (Option A) | PRETRAINING_PROTOCOL.md | The fixed 130M-token warmup is 26% of a 0.5B sweep run but 4.3% of the 3B run, so the T^-0.32 transfer compares different schedule shapes. Proposed: scale warmup to 4.3% (83 steps) in sweeps, not zero (no warmup may bias AdamW's optimum low). First check which warmup Hernanz used for the exponent. | Warmup is 4.33% of every run (83 steps in a 0.5B sweep run, 496 in 3B). Hernanz's T^-0.32 was not his own fit, so the transfer check is essential. (2026-11-24) |
| A23 | Decide whether D₀ may be the validation set for LR selection | decide | done | §5 Data | runs/pretrain_study.sh | The launcher evaluates every run on D₀ (1M tokens), so the LR sweep would pick winners on the same set used to measure forgetting. Alternative: select on the validation shard and keep D₀ for the fine-tuning analysis. | Select LRs on the held-out validation shard; D0 is sampled from that shard, so validation skips D0's documents. D0 is only logged in pretraining and reserved for forgetting. (2026-11-24) |
| A1 | Train routers with the run's matrix optimizer | build | done | §4 Router optimizer | nanochat/gpt.py:519-533 | The spec fixes this [DECIDED]. The code always puts routers in AdamW, even in Muon runs. | Routers follow the matrix optimizer by default (--router-optimizer same). (2026-11-23) |
| A2 | Switch pretraining data to FineWeb-Edu (100B sample) | build | done | §5 Data | nanochat/dataset.py:23 | The code downloads ClimbMix. Also set aside a ~2M-token held-out D₀, MinHash-deduplicated against training data, with document IDs recorded. | FineWeb-Edu-100B is the default dataset; make_heldout builds D₀. Running them is item A19. (2026-11-23) |
| A3 | Add a load-balancer flag: auxiliary loss or bias | build | done | §4 Load balancer · Decision 1 | nanochat/gpt.py:228-230, 630 | The auxiliary loss exists, but its 0.01 / 0.001 weights are hard-coded. The bias balancer (u = 0.001, used for selection only) is missing. Log per-layer loads, loss value or bias vector, and bias spread. Team picks the default. | --balancer aux\|bias\|none with configurable weights; the default is still for the team to confirm (A17). (2026-11-23) |
| A4 | Add a Muon variant flag | build | done | Decision 4 | nanochat/optim.py · scripts/base_train.py:262 | Current Muon: fixed-coefficient Newton–Schulz, no weight decay, √(rows/cols) LR scaling, nanochat momentum schedule 0.85 → 0.97 → 0.90. Spec recommendation: constant momentum 0.95, Moonlight 0.2·√max(m,n) scaling and decoupled weight decay, with nanochat's variant behind the flag. | moonlight (default; renamed from plain on 2026-11-23) and keller variants; nanochat's variant dropped by decision. (2026-11-23) |
| A5 | Finish the AdamW matrix recipe | build | done | Decision 5 | scripts/base_train.py | Add global gradient clipping at 1.0 and log the pre-clip norm (no clipping exists). Default weight decay is 0.28; the spec recommends 0.1. ε is 1e-8; record it. | Global clipping at 1.0 in every cell, pre-clip norm logged; weight decay 0.1. (2026-11-23) |
| A6 | Add the analysis hooks | build | done | §4 Required hooks · Decision 10 | nanochat/gpt.py MoEMLP · nanochat/training_stats.py | Forced-route path (forced_idx); capture of router inputs h and logits z; cumulative per-expert token counts nᵢ (only per-step shares are logged today); configurable set of MoE layers. | Route replay, route capture, cumulative expert counts and --moe-layers. (2026-11-23) |
| A7 | Write a full run record | build | done | §0 rule 1 · §7 | scripts/base_train.py | config.json next to each run's outputs with every resolved setting, data paths, seed and git commit. No git commit is recorded today. | config.json per run, with git commit, tokenizer and data identity. (2026-11-23) |
| A8 | Pass correctness tests 1–7 | build | done | §8 | tests/ | Dense equivalence, self-replay identity, router gradient with the balancer off, counter sanity, scrambled-route positive control, AdamW/Muon batch pairing, balancer behaviour. tests/test_gpt.py covers parts of these. | Spec §8 tests 1–7 pass locally (test 5 checks mechanics; the real check needs a trained model). (2026-11-23) |
| A10 | Confirm depth and token budget | decide | done | Decisions 6, 7 | PRETRAINING_PROTOCOL.md | Recommended: depth 10 (118M active with GPT-2, a measured shape in Ludziejewski et al.) and 3,000,238,080 tokens for every E, with decay branches at 1B and 2B. | Depth 10, 3B tokens confirmed. E = 8 is the headline cell; pretraining may start with E = 8 only (both optimizers). (2026-11-24) |
| A11 | Keep or retrain the tokenizer | decide | done | Decision 8 | scripts/tok_train.py | tok_train.py trains a 32,768-token BPE (9 special tokens) on the first ~2B characters of the local pretraining shards, currently ClimbMix. Moving to FineWeb-Edu means retraining, which needs team sign-off; brace tokens would then get no pretraining gradient. | GPT-2 tokenizer, decided 2026-11-23. No retraining needed. (2026-11-23) |
| A12 | Define the E = 1 dense control | decide | done | §3 · §8.1 | nanochat/gpt.py:140-161 | The dense model uses a ReLU² MLP at 4× width; experts are SwiGLU at 3× width. Test 1 needs either E = 1 MoE as the control or a SwiGLU dense path. | E=1 MoE reproduces a dense SwiGLU model exactly (tested); it is the control. (2026-11-23) |
| A13 | Choose pretraining seeds | decide | done | Decision 9 | PRETRAINING_PROTOCOL.md | Recommended: 3 seeds at E = 8, 1 elsewhere. | 3 seeds at E = 8, 1 elsewhere; the seed changes only the init, the data order is fixed. (2026-11-24) |
| A20 | Decide whether GPU runs must be bitwise reproducible | decide | done | §8 exact resume | scripts/base_train.py | Two identical GPU runs differ after a few steps (with either dispatch), so resumed runs are only statistically equivalent; data order is identical. Options: accept, or try torch deterministic mode and measure its cost. | Accepted: GPU runs are not bitwise reproducible; seeds measure the noise. (2026-11-24) |
| R1 | Switch-style top-1 gate | build | done | Decision 3 | nanochat/gpt.py:175-221 |  | Softmax over all E experts, output scaled by p_i*, no renormalization; router computed in fp32. (2026-11-15) |
| R2 | Dropless routing | build | done | Decision 2 | nanochat/gpt.py:196-221 |  | No capacity limit in training or evaluation. (2026-10-26) |
| R3 | nanochat's Muon extras removed | build | done | §4 Muon variant | nanochat/optim.py |  | Polar Express, NorMuon, MuonEq, Muon+ and cautious weight decay removed; parity tests against torch.optim.Muon. (2026-11-15) |
| R4 | Fixed token and batch budgets | build | done | §5 | scripts/base_train.py |  | --train-tokens and --total-batch-size are required; no hidden LR, batch or weight-decay scaling. (2026-11-15) |
| R5 | Seeded, checked initialization | build | done | §4 hook 4 | scripts/base_train.py |  | --seed drives model init and global RNG, is saved in checkpoints and checked on resume. (2026-11-15) |
| R6 | Fresh optimizer state for fine-tuning | build | done | Decision 11 | scripts/chat_sft.py |  | chat_sft.py never loads pretraining optimizer state. (2026-11-16) |
| R7 | Routing diagnostics in pretraining | build | done | §7 | nanochat/training_stats.py · nanochat/routing_probe.py |  | Per-layer expert shares, routing entropy, mean probabilities, z-loss and expert gradient norms; fixed-probe route snapshots at checkpoints. (2026-11-16) |
| R8 | Exact dataloader resume | build | done | §4 hook 4 | tests/test_dataloader_resume.py |  | Resumed runs replay the same packed batches. (2026-11-16) |
| R9 | FLOP estimates count only active experts | build | done | §5 Compute | nanochat/gpt.py:411-425 |  | num_matmul_params excludes the unused experts. (2026-11-16) |
| R10 | Speedrun tricks removed | build | done | §4 Architecture | nanochat/gpt.py |  | Sliding-window attention, smear/backout, value embeddings, residual scalars, QK ×1.2, logit soft-cap and FP8. (2026-10-26) |
