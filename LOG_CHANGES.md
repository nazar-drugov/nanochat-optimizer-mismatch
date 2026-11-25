## Commit 1: Use full causal attention by default

- Changed `GPTConfig.window_pattern` default from `"SSSL"` to `"L"`.
- Changed the `--window-pattern` default in `scripts/base_train.py` from `"SSSL"` to `"L"`.

## Commit 2: Remove token smear and backout

- Removed `smear_gate` and `smear_lambda`.
- Removed smear initialization and forward-pass logic.
- Removed smear-specific KV-cache state from `engine.py`.
- Removed `backout_lambda`.
- Removed backout initialization, intermediate-layer caching, and residual subtraction.
- Updated parameter counting and optimizer grouping.
- Removed the now-unused `n_layer` variable from `GPT.forward()`.

## Commit 3: Remove value embeddings

- Removed `has_ve()`, `value_embeds`, `ve_gate_channels`, and `ve_gate`.
- Removed value-embedding and gate initialization.
- Removed value embeddings from attention and block forward methods.
- Removed value embeddings from dtype conversion, parameter counting, and optimizer grouping.
- Updated the related optimizer comment.
- Left the unrelated contraction-matching regex in `tokenizer.py` unchanged.

## Commit 4: Restore standard residual connections

- Removed learnable per-layer `resid_lambdas`.
- Removed learnable input-mixing `x0_lambdas`.
- Removed their custom initialization.
- Removed the initial-embedding skip path.
- Restored residual behavior through `Block` only.
- Removed residual scalars from parameter counting and optimizer groups.
- Removed the `scalar_lr` optimizer argument and `--scalar-lr` training option.
- Removed obsolete checkpoint patching for these parameters.
- Verified construction, forward/backward, optimizer setup, and strict state-dict loading.
- Full tests: 43 passed, 14 skipped; only the unrelated macOS memory-limit test failed.

## Commit 5: Remove sliding-window attention

- Removed `window_pattern` from `GPTConfig` and training arguments.
- Removed `_compute_window_sizes()` and `self.window_sizes`.
- Removed `window_size` arguments from attention and block methods.
- Made training and cached inference always use full attention with `window_size=(-1, 0)`.
- Simplified training, prefill, decode, and KV-cache calculations for full attention.
- Removed obsolete checkpoint compatibility patching.
- Removed remaining references from SFT metadata, inference output, and CPU training scripts.
- Focused tests: 13 passed, 14 skipped.

## Commit 6: Remove additional query-key scaling

- Removed the manual `1.2` multiplication applied to queries and keys.
- Kept QK normalization unchanged.
- This removes arbitrary attention sharpening while preserving the stability benefit of QK normalization.

## Commit 7: Remove logit soft-capping

- Removed the fixed `softcap = 15` value.
- Removed the `tanh` transformation that constrained logits to `[-15, 15]`.
- Kept logits converted to FP32 before cross-entropy computation.
- This restores standard language-model logits and avoids modifying gradients during dense-versus-MoE comparisons.

## Commit 8: Add dense-model regression tests

- Added forward and backward-pass coverage for the simplified dense GPT.
- Verified that optimizer groups contain every model parameter exactly once.
- Added strict state-dict round-trip coverage.
- Verified that training and KV-cache inference always request full causal attention.
- Added checks for training, prefill, decode, and KV-cache cost calculations.
- Combined focused tests: 18 passed, 14 skipped.

## Commit 9: Remove FP8 training

- Removed the `--fp8` and `--fp8-recipe` pretraining arguments.
- Removed FP8 linear-layer conversion and filtering from `scripts/base_train.py`.
- Removed the `disable_fp8()` evaluation context manager and now run validation, CORE evaluation, and sampling directly in the configured compute dtype.
- Removed the `--fp8` option from `runs/speedrun.sh`.
- Deleted the now-unused `nanochat/fp8.py` implementation.
- Kept Flash Attention's FP8 compatibility comment because it describes the external attention kernel rather than FP8 model training.
- Python and shell syntax checks passed.
- Combined focused tests: 18 passed, 14 skipped.

## Commit 10: Add dense and top-1 MoE feed-forward variants

- Added `ffn_type` and `n_experts` to `GPTConfig`, defaulting to `"dense"` and four experts.
- Added `MoEMLP` with a bias-free FP32 router and top-1 token-choice routing.
- Kept every expert architecturally identical to `DenseMLP`: `n_embd -> 4 * n_embd -> n_embd` with ReLU-squared activation.
- Routed every token without a capacity limit or token dropping, and multiplied each selected expert output by its routing probability.
- Kept the MoE free of shared experts and auxiliary routing losses.
- Updated `Block` to select `DenseMLP` or `MoEMLP` from `ffn_type` while leaving attention, normalization, and residual logic unchanged.
- Split initialization into independent shared, feed-forward, and router random-number streams.
- Initialized dense and expert matrices with the same rules: `c_fc` uses the existing 0.4-scaled uniform initialization and `c_proj` starts at zero.
- Initialized router matrices with a small 0.1-scaled uniform distribution to break top-1 routing ties while keeping initial probabilities near uniform.
- Added tests for feed-forward selection, MoE forward/backward behavior, complete top-1 token routing, routing-probability scaling, FP32 router computation, full causal attention, and bit-identical shared dense/MoE initialization.
- Focused GPT tests: 11 passed.

## Commit 11: Add AdamW and Muon training options

- Added a choice between AdamW and Muon training.
- AdamW mode uses AdamW for every model parameter.
- Muon mode uses Muon for attention and feed-forward matrices. Embeddings, the language-model head, and MoE routers still use AdamW.
- Kept MoE router parameters out of the Muon parameter groups.
- Added separate learning rates for AdamW matrices, Muon matrices, and MoE routers.
- Made the AdamW beta values and epsilon configurable.
- Made the Muon beta2 value and Newton-Schulz step count configurable.
- Added checks that every model parameter belongs to exactly one optimizer group.
- Added training arguments for the optimizer, feed-forward type, and number of experts.
- Updated SFT and RL to use the renamed Muon learning-rate argument.
- Added `ffn_type` and `n_experts` to SFT checkpoint metadata.
- Added optimizer-group tests for dense and MoE models with both optimizer choices.
- Focused GPT tests: 14 passed.
- One forward, backward, and optimizer step passed for dense and MoE models with both AdamW and Muon.

## Commit 12: Use the distributed optimizer for AdamW and Muon

- Renamed `MuonAdamW` to `DistributedOptimizer` because the class handles both AdamW and Muon parameter groups.
- Changed `GPT.setup_optimizer()` to always create `DistributedOptimizer`.
- In AdamW mode, every parameter group uses the AdamW update. No Muon update is run.
- In Muon mode, attention and feed-forward matrices use Muon. Embeddings, the language-model head, and MoE routers use AdamW.
- This keeps AdamW training safe across multiple GPUs. nanoChat does not wrap the model in PyTorch DDP, so the optimizer must synchronize gradients between ranks.
- Kept the single-GPU path unchanged: distributed communication is skipped when only one rank is active.
- Updated imports, optimizer tests, the SFT comment, README, and the change plan to use the new class name.
- Focused tests: 14 passed and 4 GPU-only optimizer tests were skipped on the test machine.
- Python syntax and diff checks passed.
- A true multi-GPU synchronization test is still needed.

## Commit 13: Restore original Muon

- Replaced Polar Express with the original fixed-coefficient Newton-Schulz iteration.
- Removed MuonEq row equilibration, Muon+ renormalization, NorMuon second-moment scaling, and cautious weight decay from Muon updates.
- Kept Nesterov momentum, five configurable Newton-Schulz steps, original aspect-ratio learning-rate scaling, and BF16 orthogonalization.
- Removed the obsolete Muon beta2 argument and second-moment optimizer state.
- Kept the existing distributed reduce-scatter, sharded momentum state, and all-gather path.
- Added single-rank parity tests against `torch.optim.Muon` for wide and tall matrices across multiple momentum values.
- Added a two-rank CUDA parity test with different local gradients and a parameter count that is not divisible by the world size.
- Focused GPT tests: 14 passed.
- Optimizer tests: 7 skipped because CUDA was unavailable; the new two-rank test requires at least two CUDA devices.
- A direct CPU parity check differed from `torch.optim.Muon` by at most `4.1e-4` after one step.

## Commit 14: Add configurable and checked training seeds

- Added a `--seed` option to `scripts/base_train.py`.
- The default seed is `42`.
- The training script now uses this seed for the shared PyTorch random state.
- The training script also passes this seed directly to model initialization.
- The seed is saved in the checkpoint settings with the other training options.
- Added tests that create two models with the same seed and check that every parameter is identical.
- Added tests that create two models with different seeds and check that their parameters are different.
- Ran these tests for both dense and MoE models.
- Kept the existing test that checks that dense and MoE models have identical shared weights when they use the same seed.
- Added a check when training resumes from a checkpoint.
- Training continues when the requested seed matches the seed stored in the checkpoint.
- Training stops with a clear error when the two seeds do not match.
- Older checkpoints may not contain a saved seed. These checkpoints show a warning and continue loading.
- Added tests for matching, mismatched, and missing checkpoint seeds.
- Focused tests: 21 passed.
- Python syntax and diff checks passed.

## Commit 15: Use one safe name for W&B and checkpoints

- Added automatic experiment names for pretraining runs.
- Automatic names include the model depth, feed-forward type, optimizer, seed, and the main optimizer settings.
- MoE names also include the number of experts and the router learning rate.
- AdamW names include matrix weight decay.
- Muon names include the number of Newton-Schulz steps.
- The same experiment name is now used for the W&B run and the checkpoint directory.
- A custom `--run` value replaces the automatic name in both places.
- W&B logging now starts by default on the main process.
- Other distributed ranks continue to use the no-op W&B logger.
- The resolved experiment name is included in the W&B configuration and checkpoint settings.
- Added validation for custom run names before creating the checkpoint directory.
- Empty names, whitespace-only names, `/`, `\\`, `.`, and `..` are rejected.
- Added tests for accepted and rejected custom names.
- Focused tests: 29 passed.
- Python syntax and diff checks passed.

## Commit 16: Use fixed token and batch budgets for pretraining

- Removed FLOP-based and parameter-count-based training budgets from `scripts/base_train.py`.
- Replaced the old training-horizon options with required `--train-tokens` and `--total-batch-size` values.
- Training now stops after the requested number of tokens.
- The token budget must be positive and divisible by the total batch size.
- Removed automatic batch-size selection.
- Removed automatic learning-rate scaling based on batch size.
- Removed automatic AdamW weight-decay scaling based on model size.
- The optimizer now receives the learning rates and weight decay supplied by the user without hidden scaling.
- Original Muon matrices continue to use no weight decay.
- Removed FLOP and MFU values from pretraining output and W&B logs.
- Added `tokens_seen` to validation, CORE, and training logs.
- Kept wall-clock time, step time, tokens per second, and peak memory reporting.
- Added the token budget and total batch size to automatic experiment names.
- Added experiment-name tests for the training budget.
- Updated the main pretraining examples and launch scripts to use explicit token and batch budgets.
- Focused tests: 31 passed and 7 CUDA-only tests were skipped.
- Python syntax, command-line help, and diff checks passed.

## SwiGLU MoE experts, routing losses, and training diagnostics

- Replaced the MoE experts' ReLU-squared MLPs with SwiGLU MLPs using three bias-free `Linear` projections: up, gate, and down. The hidden width is three times the model width.
- Added explicit fan-in-scaled truncated-normal initialization for the three SwiGLU matrices in `GPT.init_weights()`; dense-model initialization remains separate.
- `MoEMLP.forward()` now returns its expert output, Switch-style balance and z-losses, and detached routing statistics for every MoE call.
- MoE scalar training losses always include the mean layer balance loss at weight 0.01 and mean layer z-loss at weight 0.001. Dense losses and per-token cross-entropy requests keep their existing behavior.
- `base_train.py` logs cross-entropy, balance, z, total loss, and per-layer routing statistics without an auxiliary-loss CLI switch.
- Every 100 training steps, `base_train.py` also logs each expert's gradient norm after accumulation and unscaling, before the optimizer step. Distributed logs report the RMS local norm across ranks.
- Focused model and experiment-name tests passed. Compiled GPU throughput remains unverified.

## Fixed-probe routing stability at pretraining checkpoints

- Added `nanochat/routing_probe.py` to run a saved token batch through the full, uncompiled MoE model. Temporary hooks capture each layer's actual router logits and record one top-1 expert ID for every batch and sequence position. The helper removes its hooks and restores the model's training mode, including if the forward pass fails.
- Added per-layer change rates: for each layer, divide the number of probe positions whose selected expert differs from the preceding snapshot by the number of probe positions. The first snapshot is a baseline and has no change rate.
- Store `routing_<step>.pt` beside each MoE pretraining checkpoint with the checkpoint step, fixed probe token IDs, and per-layer choices. Snapshot comparison rejects changed probe inputs or mismatched choice shapes, and saving refuses to overwrite an existing snapshot.
- Rank 0 selects one validation batch for a new run, runs the probe at each checkpoint, and logs the per-layer change rates to W&B. On resume it reloads the saved inputs and choices for the requested checkpoint; a missing snapshot fails explicitly instead of silently selecting a new batch. Resuming from the final step does not rewrite its checkpoint or snapshot.
- Added focused tests for capture shape and expert range, evaluation and training mode restoration, hook cleanup after failure, per-layer rates, snapshot round trips, changed-probe rejection, and dense-model rejection. The routing-probe and GPT suites passed: 25 tests.
- `--save-every=-1` creates only a final baseline. A positive checkpoint interval is required for a pretraining stability time series. Full training and distributed probe execution remain unverified.

## Pretraining readiness (2026-11-23)

Decisions and the learning-rate sweep protocol are in `PRETRAINING_PROTOCOL.md`.

- **Router optimizer:** `--router-optimizer same|adamw|muon`, default `same`, so routers use the run's matrix optimizer as the spec decides. The router LR defaults to that optimizer's matrix LR. The hidden 1/√(d/768) scaling now applies only to the embedding and lm_head groups.
- **Muon variants:** `moonlight` (default; renamed from `plain` after the first version of this entry: Moonlight 0.2·√max(m,n) update scaling plus decoupled weight decay) and `keller` (the previous original-Muon scaling). Weight decay applies in both. Muon momentum is constant 0.95 by default; nanochat's schedule is available with `--muon-momentum-schedule nanochat`.
- **Gradient clipping:** global-norm clipping (`--grad-clip`, default 1.0) on the reduced gradient, correct across ranks. The pre-clip norm and clip coefficient are logged.
- **Bug fix:** the optimizer waited on `Work.get_future()`, which gloo lacks, and gloo re-copies reduce-scatter outputs on every `wait()`. It now waits on work handles once, so the multi-rank path runs on CPU and clipping is not undone on sharded tensors.
- **Load balancing:** `--balancer aux|bias|none` with configurable `--balance-coef`, `--z-coef` and `--bias-update-rate`. The bias changes selection only, never the gate value. Bias updates use step loads summed over micro-batches and ranks.
- **MoE layers:** `--moe-layers all|every:K|i,j,...`. Non-MoE layers in an MoE model, and the new `--ffn-type swiglu`, use a dense SwiGLU MLP identical to one expert.
- **Analysis hooks:** `GPT.forward(forced_routes=..., routing_capture=...)` for route replay and capture of router inputs, logits and the routes actually used. `routing_probe` now records the routes actually used, including the bias.
- **Data:** a dataset registry with FineWeb-Edu-100B as the default. `--dataset`/`--data-dir` reach the loader, which also gains seeded shuffling with exact resume. `scripts/make_heldout.py` builds a MinHash-deduplicated held-out D0.
- **Tokenizer:** `--tokenizer gpt2|nanochat`, default `gpt2`, recorded in checkpoints and used on reload. The engine tolerates tokenizers without chat tokens, and pretraining samples use the BOS id.
- **Schedule:** `--warmup-tokens` (default 130M) and `--warmdown-ratio 0.2` decaying to 0. The default batch is 256K tokens. Runs whose warmup plus decay exceed the run are rejected.
- **Records:** `config.json` (and `config_resume_<step>.json`) contain every argument, the model config, parameter counts under each convention, every optimizer group, data and tokenizer identity, git commit and dirty state, and hardware. Checkpoints store cumulative expert counts and per-rank data fingerprints.
- **Tests:** 200 passed and 18 CUDA-only tests skipped (`uv run --group dev python -m pytest -q`). They include spec §8 tests 1–7, optimizer parity with `torch.optim.Muon`, a two-rank gloo test, and end-to-end CPU runs of `base_train.py`. Those runs show that AdamW and Muon cells see identical data and that resuming is bit-exact. The only failure is the known macOS `test_execution.py::test_memory_limit`.
- **Not yet verified:** CUDA throughput, the two-GPU NCCL path and the CUDA-only optimizer tests. They need the cluster.
- **Parked:** the fine-tuning data scripts, the LM fine-tuning script and the evaluation pipeline. `chat_sft.py` and `base_eval.py` were not updated for GPT-2 checkpoints.

## Data preparation and a launcher for the cluster (2026-11-23)

This entry covers the work done after the "Pretraining readiness" commit (`0ed4cd4`). The main goal was to get the training data ready on the cluster and to make sure every training run measures its results on the same, unchanging held-out data. Some terms used below:

- **D0** is a small held-out set of text (about 2 million tokens) that training never sees. We use it to measure how good each model is. "Bits per byte" (bpb) is the score; lower is better.
- **Shards** are the pieces the training text is split into. Each one is a `.parquet` file of about 94 MB.
- **Pool** and **scratch** are two storage areas on the cluster. Pool is for keeping big datasets between jobs. Scratch is fast storage for files that running jobs read and write. Neither is backed up.
- **SHA-256** is a short "fingerprint" of a file. If even one byte of the file changes, the fingerprint changes, so we can tell when a file was modified.

### The data is now downloaded and D0 is built

- Downloaded 100 FineWeb-Edu training shards (`shard_00000` to `shard_00099`) plus the validation shard (`shard_01822`). They total about 9.5 GB. One copy is in pool and one is in scratch.
- The GPT-2 tokenizer is cached in `~/.cache/nanochat/tiktoken_cache/`, so jobs without internet can still use it.
- Built D0 from the validation shard. D0 has 1,881 documents and 2,000,142 GPT-2 tokens.
- While building D0, every candidate document was compared with all 100 training shards to find near-copies. 136 candidates were found to be near-copies of training text and were thrown out. This keeps D0 fair: the model is not tested on text it has already seen. The full list is saved in `d0_report.json`.
- Both D0 files live in `/home/ndrugov/orcd/scratch/public/nanochat-optimizer-mismatch/heldout/fineweb-edu-100b/`.

### A saved "snapshot" of the data (`data_manifests/fineweb_edu_100_v1/`)

This folder is a written record of exactly which files we used. The launcher checks the files against it before every training start and resume (see "A single script" below), and anyone can repeat the check by hand. It contains:

- `shards.sha256`: the fingerprint of each of the 101 shard files.
- `shard_sizes.tsv`: the size in bytes of each shard file.
- `heldout.sha256`: the fingerprints of `d0.parquet` and `d0_report.json`.
- `README.md`: explains what the snapshot covers and gives the exact commands for checking the files. It also says that if the shards ever change, a new D0 and a new snapshot version must be made. The fingerprints detect changes, but they do not stop anyone from making them.

### Training can now be told to use D0 (`scripts/base_train.py`)

- Added a `--val-data-dir` option. It points to a folder that holds exactly one parquet file, and training uses that file for its validation scores. Without the option, training works as before and uses the last shard of the training folder.
- The training script reads the validation file, computes its SHA-256, prints it, and saves it (with the file path and folder) in `config.json`, in each checkpoint's metadata, and in the run record.
- When a run is resumed from a checkpoint, the script compares the current validation file with the one saved in the checkpoint. If the file or its fingerprint differs, it stops with a clear error instead of silently scoring against different data. It also refuses to resume with `--val-data-dir` if the checkpoint has no saved fingerprint.
- Added a check on `--eval-tokens`. It must now be a multiple of (per-GPU batch size x sequence length x number of GPUs). Before, a badly chosen value could silently cause fewer tokens to be scored than requested. The study's value, 1,048,576, passes this check.

### A single script to prepare data and start runs (`runs/pretrain_study.sh`)

This is a new script with several modes. The first lines of the file explain how to use it:

- `paths` prints where everything lives.
- `download` fetches the shards into pool. It needs internet.
- `stage` copies the shards from pool to scratch.
- `heldout` builds D0 (this is heavy on the CPU).
- `dry-run` prints the full training command without starting anything, so you can check it first.
- `train` starts a training run on the GPUs.

Download, stage and heldout are one-time setup steps. `train` never downloads anything. It always reads the scratch copy, and it always passes D0 to the training script. Safety checks in `train`:

- It stops if the shards or D0 are missing.
- It checks D0 and its report against the saved fingerprints and stops if they differ.
- It checks the training shards in two steps, on every start and every resume. First, the list of `.parquet` files in the data folder must match the manifest exactly. The training code treats every file except the last as training data, so one stray extra file would silently become training data, and a missing file would silently shorten it. Second, every shard's SHA-256 must match `shards.sha256`, which catches an edited file even if its size did not change. On this node hashing all 100 shards took about 26 seconds from a cold start and about 13 seconds when the files were already cached. If anything differs, the script prints which file and stops before any GPU time is used. `dry-run` does not do these checks.
- The shard check follows symbolic links, as the Python data loader does. (My first version did not, and wrongly reported all 101 shards as missing when the data folder was a symbolic link. A test now covers this.)
- It refuses to reuse a run name that already has results, unless you are resuming a run.
- You cannot override `--run`, `--data-dir`, `--val-data-dir` or `--train-stats-jsonl` from the command line, because the script sets them. This stops two runs from accidentally sharing output or using different data.
- It insists on an explicit `--train-tokens` value, and sets `--eval-tokens=1048576` unless you pass your own.
- The header also describes a short one-step test for the two-GPU node. That test confirms the cluster really reads D0 before the learning-rate sweep starts. **This test has not been run yet.**

### A map of where files go (`STORAGE_PATHS.md`)

New document listing every location: the code and virtual environment in home, the dataset in pool and scratch, D0, checkpoints, logs, and W&B files, plus where SFT, RL and evaluation files go. It says which folders are backed up (home) and which are not (pool, scratch), so final results get copied somewhere safe. It also explains that the paths in it are defaults and patterns, not proof that those files exist.

### Documentation updates

- `PRETRAINING_PROTOCOL.md`: added an explanation of the proposed 1B- and 2B-token "decay branch" checks. These runs are a proposed extra check and are not part of the chosen 3B-token budget. Also added the pool/scratch workflow and changed the example command to use the new launcher. The sections "Batch size" and "Auxiliary loss" are currently empty headings, and "LR Schedule" is one short sentence. Longer drafts of these (the batch-size reasoning and a table of balance-loss values from papers) were written and then removed from the file, so these sections still need their final text.
- `CHANGES_PLAN.md`: rewritten as an ordered to-do list: confirm the settings, prove the new code works on the cluster, run the learning-rate sweep, run the final comparisons, and do the fine-tuning study. The "prepare the data" stage was removed because that work is done. The stage numbers were not renumbered, so they go 1, 2, 4, 5, 6. The two-GPU check that training reads D0 is still open and is listed under "Settle the protocol before spending GPU time" and in the "Still to do" list below.
- `README.md`: rewritten to describe this study instead of the original nanochat "GPT-2 speedrun" project. It now points to the new launcher and the storage map.
- `dev/LEADERBOARD.md`: added notes saying that its commands are historical and are not the instructions for this study.

### Small fixes so the code runs on the cluster

- `nanochat/flash_attention.py`: the code now only tries to load the Flash Attention 3 add-on on GPUs it was built for (compute capability 8.0, 8.6, 8.9 and 9.0). Other GPUs use PyTorch's built-in attention instead. Setting `NANOCHAT_ATTENTION_BACKEND=sdpa` forces the built-in attention on any GPU.
- `nanochat/optim.py`: the optimizer update functions are now only compiled with `torch.compile` when their data is on a GPU. On a CPU they run the same math without compiling. This is what lets the tests run on CPU.
- `pyproject.toml` and `uv.lock`: the `gpu` install option now also requires `setuptools`, as the `cpu` option already did.
- `nanochat/tokenizer.py`: only a comment changed. It no longer points to `tok_train.py`, which was deleted (see below).

### Tests added

- `tests/test_attention_fallback.py`: one test checks that choosing the built-in attention skips loading Flash Attention 3. Another checks that an unsupported GPU (compute capability 10.0) falls back to the built-in attention.
- `tests/test_base_train_smoke.py`: one new end-to-end test trains a tiny model on CPU. Its training folder contains a deliberately broken validation shard, so the run can only succeed if D0 is really used. The test then checks that the D0 path and fingerprint were saved in `config.json` and in the checkpoint. Finally it changes D0 and checks that resuming fails with the "Validation file or SHA-256 differs" error.
- `tests/test_checkpoint_manager.py`: 16 new fast tests for the resume checks. They cover identical shards, a moved folder, a missing shard, an extra shard, a changed size, a different order, a long list of problems, an old checkpoint without a shard list, and the tokenizer rules (same tokenizer, different name, old checkpoint without a name, changed file, missing file, added file, name-only check).
- `tests/test_base_train_smoke.py`: one more end-to-end test. It trains a tiny model, checks that the shard list and tokenizer fingerprints were saved in the checkpoint and in `config.json`, and then tries to resume after: adding a training shard, replacing a shard, removing a shard, changing the tokenizer name in the checkpoint, and changing a tokenizer file fingerprint in the checkpoint. Every one must fail with the right message. (The earlier resume test still passes, so a normal resume gets through the new checks.)
- `tests/test_pretrain_study_launcher.py` (new file, 6 tests, under a second): runs the launcher's `train` mode on tiny fake shards with a fake manifest. It stops before any training would start. It checks that matching data passes, that a shard edited without changing its size is rejected, that a missing shard and an extra shard are rejected, that a data folder that is a symbolic link works, and that `dry-run` skips the check.
- **Full test run** (`python -m pytest tests`, on a CPU-only compute node, about 4 minutes): **227 passed, 18 skipped, 0 failed**. The previous entry reported 200 passed and 18 skipped, so these are the 27 new tests. The 18 skipped tests are the same CUDA-only tests as before; this node has no working GPU, so they still have not been run. The macOS-only memory-limit test that failed before passes on this Linux node.
- I also ran the launcher's data check once against the real 100 shards and the real manifest (using a temporary output folder, so nothing real was written). It passed in about 13 seconds.

### Files removed

These came from the original nanochat project and are not needed for this study, so they were deleted. Anyone who wants them can find them in earlier commits.

- Original launch scripts: `runs/speedrun.sh`, `runs/miniseries.sh`, `runs/scaling_laws.sh` and `runs/runcpu.sh`.
- Chat and inference tools: `scripts/chat_cli.py`, `scripts/chat_rl.py` and `scripts/infer_bench.py`.
- Tokenizer tools: `scripts/tok_train.py` and `scripts/tok_eval.py`. The study uses the existing GPT-2 tokenizer, so nothing here trains a new one. The `--tokenizer nanochat` option still exists but now needs a tokenizer trained elsewhere or from an earlier commit.

### Resumed runs now check the training data and the tokenizer

The D0 check above only protected the validation data. When a run was resumed, nothing confirmed that the **training** data was the same as when the checkpoint was saved. The checkpoint stores the reading position as just "file N, row group M", so if shards had been added, removed or edited in between, the run would have carried on from that position in different data without any error. The tokenizer name saved in the checkpoint was not compared with `--tokenizer` either. This is now fixed in two layers:

- **Inside the training code (`scripts/base_train.py`, `nanochat/checkpoint_manager.py`):**
  - Every checkpoint now saves the list of training shards with their sizes (`train_shards`, in the order the loader reads them). The same list is written to `config.json`.
  - On resume, the current list is compared with the saved one. The run stops with a clear message that names the problem and the files: "missing shards", "extra shards", "shards with a different size", or "same shards, but in a different order". Long lists are shortened to five names.
  - Only names, order and sizes are compared, not the folder path. So moving the same data (for example from pool to scratch) is allowed, but changing the data is not.
  - Every checkpoint now also saves the SHA-256 of each tokenizer file (`tokenizer_identity`). On resume, the tokenizer name must match, and every saved tokenizer file must still exist with the same fingerprint. Extra files added to the tokenizer folder later are allowed, because the GPT-2 cache folder may legitimately gain other tokenizers.
  - Checkpoints made before this change have no saved shard list, so they only produce a warning and continue (the same way old checkpoints without a seed are handled). Old checkpoints without a saved tokenizer fingerprint are checked by name only.
- **Inside the launcher (`runs/pretrain_study.sh train`):** the byte-level check against the manifest described above. The Python check only sees sizes, so it cannot catch a file edited without changing its size. The launcher check can. Resuming through the launcher therefore confirms that the data matches the manifest, and the Python check confirms it matches this particular checkpoint.

The running data fingerprint saved in checkpoints is unchanged. It is still only for comparing runs afterwards (for example, to show that an AdamW run and a Muon run saw the same examples in the same order). It was never meant to detect a changed shard.

One limit remains: if you start a run with the training code directly (not through the launcher), the byte-level hash check does not happen. Only the size comparison on resume does. Use the launcher for real runs.

### Still to do

- Run the one-step, two-GPU check that training reads D0 (described at the top of `runs/pretrain_study.sh`).
- Run the CUDA-only tests and the two-GPU training checks from `CHANGES_PLAN.md`.
- Confirm the settings listed as "Defaults that still need team confirmation" in `PRETRAINING_PROTOCOL.md`.
- Fix the reference to `EXPERIMENT_SPEC.md`, which is not in this checkout.

## GPU test fixes (2026-11-23)

Found by running the full suite and a short CUDA `base_train` on an RTX 5080 (sm_120): 209 passed, 13 skipped.

- **Bug fix:** Flash Attention 3 was loaded from the kernel hub on any CUDA GPU that had a build, including Blackwell (sm_100 B200, sm_120 RTX 50xx), where the first attention call aborts the process ("no kernel image is available for execution on the device"). The data-preparation work above fixed the same bug independently; the rebased code keeps that version (FA3 only on compute capability 8.0, 8.6, 8.9 and 9.0, plus `NANOCHAT_ATTENTION_BACKEND=sdpa`). `test_unsupported_cuda_architecture_uses_sdpa` now covers sm_100, sm_120 and sm_75 and checks that the kernel hub is never queried.
- **Environment:** the `gpu` extra now installs `setuptools`, which torch inductor imports when compiling CPU code (several tests failed without it; the data-preparation branch added the same line).

## Grouped-GEMM MoE dispatch (2026-11-23)

- **Problem:** the per-expert loop gathers tokens with `x[mask]`, whose size depends on the data. Under `torch.compile` this breaks the graph inside the layer loop of `GPT.forward`, so the whole forward ran uncompiled (embedding, lm_head and the fp32 cross-entropy included), and each expert was compiled separately, recompiling for every new token count until the recompile limit. Peak memory doubled (fused cross-entropy lost) and the step slowed down by 1.4× (E=1) to 2.4× (E=8) relative to a dense SwiGLU model with the same active FLOPs.
- **Fix:** `MoEMLP` sorts tokens by expert and runs all experts with one `torch._grouped_mm` per projection on the stacked expert weights, with each expert's segment padded to 16 rows. Shapes are static and nothing syncs with the host, so the model compiles as one graph. Routing, gates, losses, statistics, route replay and capture are unchanged, and so are parameters and checkpoints.
- **Option:** `--moe-dispatch auto|grouped|loop` (`GPTConfig.moe_dispatch`). `auto` picks grouped for bf16 on CUDA and the loop otherwise (CPU tests, fp32). The resolved choice is printed and stored in `config.json` (`moe_dispatch_resolved`).
- **Tests (`tests/test_moe_dispatch.py`):** grouped equals the loop on outputs and all gradients (bitwise on CPU, ≤2e-3 relative on CUDA) for E ∈ {1, 4, 8}, empty experts, unaligned counts, a single token and the bias balancer; zero graph breaks; on CUDA, full-model loss and gradients match, results are deterministic across calls, and a compiled training step is one graph. Mutating the gate or the scatter makes 4–5 tests fail.
- **Benchmark:** `scripts/bench_moe_throughput.py` (spec §8.8) times compiled forward+backward and optimizer steps separately and projects a 262,144-token step. RTX 5080, depth 10, micro-batch 4×2048, time relative to dense:

  | | E=1 | E=4 | E=8 | E=16 |
  |---|---|---|---|---|
  | AdamW, grouped | 1.17× | 1.25× | 1.31× | 1.56× |
  | AdamW, loop | 1.44× | 1.87× | 2.36× | 6.2× (exceeds 16 GB) |
  | Muon, grouped | 1.13× | 1.24× | 1.31× | — (exceeds 16 GB) |
  | Muon, loop | 1.36× | 1.81× | 2.25× | — |

  On the 5080 `torch._grouped_mm` has no native kernel and falls back to one GEMM per expert with a host sync per call, which accounts for the remaining E=1 gap (GPU time equals dense). The B200 numbers need the cluster run.
- **Found while testing:** GPU training is not bitwise reproducible with either dispatch (two identical runs differ after a few steps); the data stream is identical.

## E = 8 sweep and base-model infrastructure (2026-11-24)

Decisions in `PRETRAINING_PROTOCOL.md`: λ = 0.1 for both optimizers (option A), AdamW (0.9, 0.95, 1e-8), clipping 1.0 in every cell, depth 10 and 3B tokens confirmed, E = 8 as the headline cell, LR selection on the held-out validation shard.

- **Validation and D0** (`scripts/base_train.py`, `nanochat/dataloader.py`, `runs/pretrain_study.sh`): validation now reads the validation shard `shard_01822.parquet`, not D0. D0 is sampled from that shard, so validation skips the rows D0 took (`exclude_rows`, matched by D0's `source_shard`/`row_index`); `config.json` records how many. D0 is evaluated alongside via `--heldout-data-dir` / `--heldout-eval-tokens` (logged as `heldout/bpb`, stored in checkpoints, SHA-256 checked on resume) and never used for selection. The launcher passes D0 that way and defaults `--eval-tokens` to 10,485,760.
- **`result.json`** at the end of every run: status, steps and tokens trained, final and minimum validation bpb, final D0 bpb, the evaluation curve, step-0 and final training loss, clipped-step counts and fractions (counted on the GPU, no extra sync), wall time, peak memory and the resolved settings. The evaluation curve and clip counts survive resume.
- **Divergence stop:** a run whose training loss is non-finite, or more than 1 nat above its step-0 loss after warmup, or whose validation bpb is non-finite, stops, writes `result.json` with status `diverged` and exits 0, so a job running several sweep points continues.
- **Sweep plan and jobs** (`nanochat/pretrain_sweep.py`, `scripts/pretrain_sweep.py`): the six E = 8 sweep runs (0.5B tokens, 83-step warmup), LR selection (parabola vertex in log2 LR through the best point and its neighbours; one more point if the best is at an edge; × 0.5635 for 3B), the 3B base-model runs (checkpoints every 1,526 steps) and Slurm job scripts for one B200 per run that skip finished runs and continue past failures.
- **Launcher portability:** the shard-list check no longer uses GNU-only `find -printf`, so the launcher tests also pass on macOS.
- **Tests:** `tests/test_pretrain_sweep.py` (plan, selection, job scripts against a fake launcher, the command line end to end), validation row exclusion in `tests/test_dataloader_data_dir.py`, D0 logging / exclusion / `result.json` / resume check and the divergence stop in `tests/test_base_train_smoke.py`, launcher arguments in `tests/test_pretrain_study_launcher.py`.

## B200 fixes and fine-tuning infrastructure (2026-11-24)

- **Empty-expert gradient on B200** (`nanochat/gpt.py`, commit b7edc71): torch 2.9's native grouped GEMM on sm_100 returns garbage weight gradients for an empty group. Every expert segment now has at least one 16-row block. Full GPU suite on a B200: 267 passed, 13 skipped.
- **Launcher** (commit 3a654e4): torchrun parsed `--run=` as its own `--run-path`; base_train options now follow `--`.
- **Persistence:** every Slurm job written by `nanochat/pretrain_sweep.py` copies small result files (`*.json`, `*.jsonl`, `*.log`, `*.parquet`) from scratch to `$POOL/results/` at the end.
- **Fine-tuning** (`scripts/finetune.py`, `nanochat/finetune_metrics.py`, `nanochat/eval_set.py`, `nanochat/finetune_data.py`, `scripts/make_finetune_data.py`, `scripts/finetune_jobs.py`): spec §2/§6/§7 measurements. One process runs every (LR, seed) cell of a base model, dataset and fine-tuning optimizer, restarting from the base weights with fresh optimizer state; paired data order by seed; per checkpoint on D0 and D1: losses, learning/forgetting, replay terms F_route/F_fixed and the reverse order, per-layer loads, MaxVio, flip rate (raw and chance-normalized), TV, churn, flip causes, expert weight change rho and fine-tuning tokens per expert; per-document loss sums and route tables (uint8) for bootstrapping. Base model: separation S, two-halves reference, router margins. Outputs: config.json, metrics.jsonl/parquet, ckpt_*.npz, result.json. Defaults (spec REC, not yet agreed): codeparrot-clean as the far dataset, 15.7M tokens at 65,536 per step, 3% warmup, 20% decay, weight decay 0, embedding/lm_head LRs 0.1x pretraining, LR grid 1/16..1 x the pretraining LR, 5 seeds, 10 checkpoints. Tested on CPU only (`tests/test_finetune.py`).
- **Report spec:** `docs/report.tex`, the 2026-11-24 draft with the agreed design changes; unagreed implementation defaults are `% NOTE` comments.

## B200 benchmark and E = 8 sweep submission (2026-11-24, night)

- **B200 throughput** (job 30414697, `~/jobs/bench_b200.jsonl` on ORCD, depth 10, grouped dispatch, 0 graph breaks everywhere). ms per 262,144-token step, with time relative to dense in brackets:

  | | E=1 | E=4 | E=8 | E=16 | dense |
  |---|---|---|---|---|---|
  | AdamW, device batch 32 | 404 (1.27×) | 388 (1.21×) | 425 (1.33×) | 410 (1.28×) | 320 |
  | AdamW, device batch 64 | 385 (1.19×) | 374 (1.15×) | 386 (1.19×) | 403 (1.24×) | 325 |
  | Muon, device batch 32 | 376 (1.32×) | 405 (1.43×) | 385 (1.36×) | 438 (1.54×) | 284 |
  | Muon, device batch 64 | 366 (1.25×) | 411 (1.41×) | 375 (1.28×) | 430 (1.47×) | 292 |

  Peak reserved memory ≤ 44 GB at 32 and ≤ 65 GB at 64. The overhead relative to dense stays at 1.2–1.5×, more than the ~1.08× expected from the RTX 5080 profile; single measurements, so differences of a few percent (e.g. E = 4 vs E = 8) are within noise.
- **Device batch stays 32** for the main runs (protocol entry of the same date): 64 would be 9% (AdamW) / 3% (Muon) faster, but the balance loss is computed per micro-batch, so it changes the balancer objective relative to the sweep. The handoff's "numerically equivalent" was wrong for MoE.
- **E = 8 sweep resubmitted** with the launcher fix: jobs 30416147 (AdamW) and 30416148 (Muon).
- **Bad node node5800-c1:** job 30416147 landed there and every run stopped with "Stage the dataset": that node mounts `/orcd/scratch/orcd/014` but not `hgo/` inside it (checked with `srun -w node5800-c1`); node5700-c1 and node5701-c1 see it. Job scripts now carry `#SBATCH --exclude=node5800-c1` (`SlurmSettings.exclude`, `--exclude` on `scripts.pretrain_sweep jobs` and `scripts.finetune_jobs`; `''` for none) and stop at once, naming the node, when the scratch directory is not visible. Muon job 30416148 was cancelled before it started.
- **W&B offline by default:** the next attempt (job 30416809, node5700-c1) stopped at `wandb.init` ("api_key not configured (no-tty)"); ORCD has no W&B login. The launcher now exports `WANDB_MODE=offline` unless the caller sets it (files under `$WANDB_DIR` on scratch, syncable later); the metrics JSONL remains the record. Muon job 30416810 was cancelled before it started.
- **W&B online with the cluster key:** the W&B API key is now in `~/.netrc` (machine api.wandb.ai, mode 600) on ORCD and locally, and compute nodes reach api.wandb.ai. The launcher defaults to `WANDB_MODE=online` when `WANDB_API_KEY` is set or `~/.netrc` (or `$NETRC`) has an api.wandb.ai entry; without a key it stays offline. The first run of each sweep job (`e8-sweep-adamw-lr0.001-s42`, `e8-sweep-muon-lr0.002-s42`) started before this and logs offline under `$SCRATCH/wandb/`; sync with `wandb sync` if wanted.
- **Sweep running** (jobs 30417229 AdamW on node5501-c1, 30417230 Muon on node5700-c1): step-0 validation bpb 3.338, D0 bpb 3.327; ~415 ms/step AdamW, ~400 ms/step Muon (~13 min per run); `config.json` shows the validation shard `shard_01822.parquet` with 1,881 D0 documents excluded.
- **E = 8 sweep, first grid finished** (jobs 30417229/30417230, 47/44 min, all six runs completed, none diverged, 2–3% of steps clipped). Final validation bpb at 0.5B tokens: AdamW 1e-3 **1.0197**, 2e-3 1.0294, 4e-3 1.0579; Muon 2e-3 **1.0034**, 4e-3 1.0176, 8e-3 1.0396 (D0 bpb ~0.015 lower throughout, same order). Both optima sit at the low edge, so `select` asks for ×½. Run in one go: ×½ and ×¼ for both (AdamW 5e-4, 2.5e-4; Muon 1e-3, 5e-4; jobs 30421535/30421537). The ×¼ point enters the parabola only if ×½ is best, in which case the protocol would ask for it next anyway, so the selection is unchanged and a round trip is saved.

## E = 8 LR selection and base-model submission (2026-11-25)

- **Extension results** (jobs 30421535/30421537): AdamW 5e-4 1.0432, 2.5e-4 1.0797; Muon 1e-3 1.0060, 5e-4 1.0278. The ×½ points lost, so the original low-edge points are the optima and both curves are convex over 4 octaves.
- **Selection** (`select --extra-lr adamw=0.0005 adamw=0.00025 muon=0.001 muon=0.0005`, `~/jobs/lr_selection.json`): parabola vertex in log2 LR through the best point and its neighbours. AdamW 1.155e-3 at 0.5B → **6.511e-4** at 3B; Muon 1.575e-3 at 0.5B → **8.874e-4** at 3B. Muon's optimum is ~1.4× AdamW's (Muon's curve is flat between 1e-3 and 2e-3: 1.0060 vs 1.0034).
- **Base models submitted** (E = 8, 3B tokens, device batch 32): seed 42 AdamW 30425713, Muon 30425714; seed 43 30425715/30425716; seed 44 30425717/30425720 (the 2-GPU limit runs two at a time).
- **Fine-tuning data on the login node failed:** both `make_finetune_data` processes stalled for ~50 min after downloading (a shard written, then no progress, processes in uninterruptible I/O, even `import huggingface_hub` from the venv took > 60 s); everything was fast again once they were killed, consistent with the login node's per-user memory limit. Rerun as a CPU Slurm job (`~/jobs/make-finetune-data.sbatch`, `mit_normal`, 64 GB, job 30425733), reusing the raw downloads.
- **Hugging Face token** installed at `~/.cache/huggingface/token` (mode 600) locally and on ORCD; it authenticates and can read the gated `bigcode/starcoderdata` (terms accepted). Our loader (`nanochat/finetune_data.py`) downloads without auth, so StarCoderData would need a loader change; the far-dataset choice (codeparrot-clean vs StarCoderData) is still open.
- **Fine-tuning data prepared** (job 30425733, `mit_normal`; 38 s and 16 s; peak RSS 6.6 GB each, so the two login-node processes together exceeded the per-user limit). WikiText-103: train 29,444 articles (518M characters), held-out D1 = official validation, 60 articles (1.1M characters); detokenized text reads cleanly ("Little Rock's", "U.S."). codeparrot-clean: train 2 files, 98,534 files of code (1.0B characters); D1 = codeparrot-clean-valid, 61,373 files (596M characters). Note: WikiText's D1 has only 60 documents, which limits a document-level bootstrap on it.
- **Seed-42 base models running** (node5600-c1, node5701-c1): ~400 ms/step, ETA ~75 min; seeds 43/44 queued (QOSMaxCpuPerUserLimit).
- **Transfer check at E = 8** (decision in the protocol): seeds 43/44 held (`scontrol hold`, jobs 30425715–30425720); 3B runs at ×0.5 and ×2 the transferred LR, seed 42, final checkpoint only: AdamW 3.256e-4 (30427007), 1.302e-3 (30427014); Muon 4.437e-4 (30427015), 1.775e-3 (30427016). New: `RunSpec` stage `xfer`, `transfer_check_runs`, `check_transfer` (vertex, offset in octaves, implied exponent), `jobs --stage transfer` and `transfer-check` in `scripts/pretrain_sweep.py`, with tests.
- **Seed-42 base models finished** (1 h 26 min AdamW, 1 h 24 min Muon; 0% of post-warmup steps clipped). Final validation bpb: AdamW **0.9180**, Muon **0.9015** (D0: 0.9063, 0.8902). Eight checkpoints each (every 1,526 steps plus the final one), 37 GB / 28 GB on scratch; results copied to pool.
- **Second bad node, node5702-c1:** transfer-check job 30427014 landed there and stopped after 1 s with the new "Scratch directory ... is not visible" message. Added to `SlurmSettings.exclude` (now `node5800-c1,node5702-c1`) and to the queued and held jobs (`scontrol update ExcNodeList=`); AdamW ×2 resubmitted.
- **Fine-tuning smoke on a B200** (job 30445993, one cell: base `e8-main-adamw-s42`, WikiText-103, AdamW 1.628e-4, seed 1): 240 steps, 104 s for the cell, 2 min 20 s for the job, peak RSS 16 GB; base D0 loss 3.1245, D1 3.5073; final learning 0.204, forgetting 0.119, F_route ~0.001; all outputs written (config, metrics.jsonl/parquet, ten ckpt_*.npz, result.json, 19 MB). The first attempt (job 30434333) stopped at once: WikiText's D1 file has only 236,411 tokens, fewer than the default `--eval-tokens` 1,048,576, so the smoke ran with `--eval-tokens 235520` (115 rows). The real WikiText cells need a decision on D1 (see open questions). Output deleted afterwards.
- **Transfer check, first round** (`transfer-check`, `~/jobs/transfer_check.json`), final validation bpb at 3B tokens, seed 42:

  | | ×0.5 | ×1 (base model) | ×2 |
  |---|---|---|---|
  | AdamW (6.511e-4) | 0.9280 | 0.9180 | **0.9130** |
  | Muon (8.874e-4) | 0.9043 | **0.9015** | 0.9050 |

  Muon: vertex 8.54e-4, −0.06 octave from the transferred LR, implied exponent −0.34 (assumed −0.32); the transfer holds. AdamW: ×2 (1.302e-3, above the 0.5B optimum 1.155e-3) is best, so the 3B optimum is at or beyond it, more than 1 octave from the transferred LR, and the AdamW base model is ≥ 0.005 bpb off its optimum (the seed-42 AdamW–Muon gap is 0.0165). New `--extra-factor` (`jobs --stage transfer` and `transfer-check`) extends the check past an edge; submitted AdamW ×4 (2.604e-3, job 30446667) and ×8 (5.209e-3, job 30446668). Seeds 43/44 stay held.
- **Transfer check, AdamW extended** (jobs 30446667/30446668): ×4 (2.604e-3) 0.9178, ×8 (5.209e-3) 0.9345. AdamW at 3B over the five points: 0.9280, 0.9180, **0.9130**, 0.9178, 0.9345; vertex **1.311e-3**, +1.01 octaves from the transferred 6.511e-4, implied exponent +0.07 (assumed −0.32): AdamW's 3B optimum is slightly above its 0.5B optimum (1.155e-3). Muon unchanged (−0.06 octave).
- **AdamW base models rerun at 1.311e-3** (rule in the protocol: offset > 0.25 octave). `transfer-check` now also writes `lr_selection_3b.json` (`selection_after_transfer`: the transferred LR, or the vertex beyond 0.25 octave). The superseded seed-42 AdamW base model (6.511e-4, final val 0.9180) was moved, on scratch and in the pool (checkpoints, results, logs), to `e8-xfer-adamw-lr0.000651-s42`, where `transfer-check` now looks for the ×1 point first; its eight checkpoints are kept. New jobs from `jobs --stage main --selection ~/jobs/lr_selection_3b.json --seeds 42 43 44` (`~/jobs/main-v2/`): AdamW s42/s43/s44 30452701–30452703; the held AdamW s43/s44 (30425715/30425717, at the old LR) were cancelled; Muon s43/s44 (30425716/30425720, LR unchanged, scripts identical to the new ones apart from the exclude list, which was updated in place) released with Nice=100 so AdamW s42 runs first.
- **AdamW base models at 1.311e-3, seeds 42 and 43 finished** (jobs 30452701/30452702, 1 h 27 min each, 0% of post-warmup steps clipped, eight checkpoints each): final validation bpb **0.9126** and **0.9127** (D0 0.9010, 0.9008). The transfer-check run at 1.302e-3 gave 0.9130, so the seed-to-seed spread looks like ~0.0005 bpb. Seed-42 gap to Muon (0.9015): 0.011 bpb, against 0.0165 with the transferred AdamW LR. AdamW s44 and Muon s43/s44 are queued (Priority). Scratch at 214 GB.
- **AdamW s44 and Muon s43 finished** (jobs 30452703/30425716, node5600-c1/node5601-c1, 1 h 25 min / 1 h 23 min, 0% clipped): AdamW s44 **0.9171** (D0 0.9055), Muon s43 **0.9010** (D0 0.8896). AdamW s44 trails s42/s43 by ~0.004 bpb from step 1,000 to the end (evaluation history), a steady offset rather than a late instability, so AdamW's seed spread is ~0.004 bpb, not the ~0.0005 suggested by s42/s43 alone. Per-seed Muon lead so far: 0.011 (s42), 0.012 (s43). Muon s44 running (job 30425720, step ~9,700 of 11,445 at 10:18).
- **Config check across the six base models** (`config.json`, all flattened keys): within each optimizer the runs differ only in seed, run name, host, job id, timestamps and git commit (Muon s42 at 87b45c6, the rest at cfc69da; the diff between them touches only the sweep tooling, tests and docs). Optimizer groups are identical apart from the matrix/router LR of the optimizer in use.
- **LR choice on held-out D0 instead of validation** (D0 never selects anything; this only checks that the choice does not depend on the eval set): the parabola vertices agree to ≤ 0.015 octave in all four comparisons (0.5B sweep AdamW 1.155e-3 vs 1.163e-3, Muon 1.575e-3 vs 1.574e-3; 3B transfer check AdamW 1.311e-3 vs 1.320e-3, Muon 8.54e-4 vs 8.63e-4). D0 bpb is lower by a nearly constant 0.014 (0.5B) / 0.012 (3B) at every LR.
- **All six E = 8 base models done.** Muon s44 (job 30425720, 1 h 23 min, 0% clipped): **0.9005** (D0 0.8887). Final validation bpb, seeds 42/43/44: AdamW (1.311e-3) 0.9126 / 0.9127 / 0.9171, mean 0.9141, sd 0.0026; Muon (8.874e-4) 0.9015 / 0.9010 / 0.9005, mean 0.9010, sd 0.0005. Per-seed Muon lead 0.011 / 0.012 / 0.017, mean 0.013. Eight checkpoints each; results in the pool; scratch at 306 GB.
- **Scratch opened to all cluster users (2026-11-25, at the user's request):** `chmod o+rx /orcd/scratch/orcd/014/hgo` (was `drwx------`; `public/` is its only entry and everything below was already world-readable, umask 0002). Before opening, every file except weights/parquet/npz was scanned for the W&B and HF tokens: none found. Pool (`/orcd/pool/006/hgo/nanochat-optimizer-mismatch`) and `~/jobs` were already readable by path. Note: the pool copy of `base_checkpoints` is 12 GB because the `meta_*.json` files are 0.4–0.8 GB each; worth checking what fills them.
- **Far fine-tuning dataset: StarCoderData (2026-11-25, user decision)**, replacing codeparrot-clean (`nanochat/finetune_data.py`, `scripts/make_finetune_data.py`). `prepare_starcoderdata` reads one language directory of the gated `bigcode/starcoderdata` (default `python`, 59 files of ~0.4 GB): training = the first `--starcoder-train-files` files (default 1); held-out D1 = `--starcoder-val-docs` (default 20,000) documents sampled with a fixed seed from the language's last file, which training never reads. StarCoderData has no validation split; the sample is shuffled because `build_eval_set` scores D1 from the start of the file and the source files are not in random order. The token comes from `$HF_TOKEN` or `~/.cache/huggingface/token` and is sent as an unredirected header, so the redirect to the storage CDN does not carry it. The dataset name in fine-tuning jobs becomes `starcoderdata`. Open: Python only vs a multi-language mix (the report text says 86 languages). Local `tests/test_base_train_smoke.py` fails 6 tests before and after this change (`ModuleNotFoundError: setuptools` in the local venv); unrelated.
- **Fine-tuning protocol decisions (2026-11-25, user):** LR chosen per cell by lowest final D1 loss (mean over seeds; the matched-learning frontier is a robustness check only); log-spaced checkpoints; the fine-tuning length ablation runs first as a pilot. Recorded in PRETRAINING_PROTOCOL.md ("Fine-tuning: LR selection, checkpoints and length pilot") and the report's LR sentence. StarCoderData Python split measured: first file 218,079 documents, 1.04B characters, ~490M GPT-2 tokens (0.466 tokens per character on 4,380 sampled documents); whole split 59 files, 23.2 GB, ~28B tokens (scaled by file size).
- **`scripts/finetune.py`: log-spaced checkpoints and decay branches.** `--checkpoint-spacing log` (default; `linear` with `--n-checkpoints` keeps the old spacing): 2, 4, 8, ... below the end of the constant phase, that end, mid-decay, last step. `--branch-tokens L1 L2 ...` runs further lengths as decay branches of one trunk (the longest length): at each shorter length's branch point the model state, optimizer state, expert counts and data fingerprint are copied, the branch's decay is run on the batches the trunk would see (a replay buffer hands them back to the trunk), then the trunk is restored. Warmup is fixed in steps from `--train-tokens`, so a branch equals a separate run of its length; checkpoints before a branch point are measured once and shared (rows rewritten per branch, `.npz` hard-linked). Outputs per length in `<cell>-T<steps>/`; rows gain `ft_steps`. `scripts/finetune_jobs.py` gains `--branch-tokens`, `--out-root` and `--job-prefix`. Tests: `test_log_spaced_checkpoints`, `test_decay_branches_equal_separate_runs` (a 4-step branch and the 8-step main length match separate runs in data fingerprint, curves and expert counts), and the job pass-through.
- **StarCoderData built** (job 30493369 on `mit_data_transfer`, 33 s, 5.6 GB peak; the same job sat for hours in `mit_normal`'s queue and was cancelled as 30490178): train 218,079 documents (1.04B characters) from `python/train-00000-of-00059.parquet`; D1 20,000 documents (91.8M characters) sampled from `python/train-00058-of-00059.parquet`. Manifest copied to `results/e8/starcoderdata_manifest.json`.
- **Length pilot submitted** (hgo account, 2026-11-25 13:48 EST): jobs 30493469–30493472 (`/home/hgo/jobs/ft-pilot/`), seed-42 base models × fine-tuning optimizer, StarCoderData, 5 LRs × seeds 1–3, lengths 60/240/960 steps as decay branches; outputs `$SCRATCH/finetune_pilot/`. Pending at handoff.
- **Handoff to Nazar's agents** (2026-11-25): HANDOFF.md rewritten with the current state; LR selection files (`lr_selection.json`, `transfer_check.json`, `lr_selection_3b.json`) copied from `/home/hgo/jobs/` to `results/e8/` so other accounts can pass them to `finetune_jobs --selection`.
- **WikiText D1 enlarged (2026-11-25, user decision):** D1 = official validation (60 articles) + test (62) + training articles sampled with seed 0 and removed from training (`assemble_heldout`, `--wikitext-d1-tokens`, default 1.1M GPT-2 tokens incl. one BOS per article), shuffled with seed 1. Rebuilt (job 30495806, `mit_data_transfer`, 43 s): D1 268 articles, 1,100,545 tokens, 5.1M characters; training 29,298 articles (146 moved), 515M characters. Checked on the login node: `build_eval_set` at the default 1,048,576 tokens gives 512 rows, 1,048,317 scored tokens, 260 articles. Before this, D1 was the validation split alone (60 articles, 0.24M tokens), too small for the default evaluation. `finetune_data/` made world-writable (`chmod -R o+rwX`) at the user's request; pretraining data left read-only.
- **Pilot split between accounts (2026-11-25 14:15 EST, user):** hgo's queued matched-pair jobs 30493471 (AdamW→AdamW) and 30493472 (Muon→Muon) cancelled; Nazar runs those two. hgo's mismatched pairs 30493469/30493470 keep running (started 13:49/13:51). Branch code verified on the B200: all finished trunks wrote 60/240/960-step results with 8/10/12 checkpoints; ~4.8 min per trunk, ~75 min per job. Fine-tuning job scripts now `export PYTHONUNBUFFERED=1` (the running jobs' Slurm logs were empty because stdout was block-buffered).
- **D0 loss gap explained (2026-11-25; `dev/d0_eval_gap.py`, job 30498480 on an L40S in `mit_preemptable`; results in `results/e8/d0_eval_gap/`).** The pretraining held-out evaluation (best-fit packer) and the fine-tuning evaluation (`nanochat/eval_set.py`, contiguous packing in file order) score different tokens with different context. Reproduced exactly: best-fit gives D0 bpb 0.9010 / 0.8903 / 0.9064 (AdamW s42 / Muon s42 / old AdamW 6.51e-4), matching pretraining; the contiguous set gives 3.176 / 3.165 / 3.195 nats/token (the smoke's 3.1245 is the old AdamW on the first 115 rows: 3.1245 here too). Gap for AdamW s42: 3.176 − 2.932 = 0.244 nats. Cause: in the contiguous set 49% of scored tokens (515,625 of 1,047,645) belong to a document that began in an earlier row, so their earlier context is cut off; they average 3.377 nats vs 2.982 for tokens whose document starts in the same row; the first 16 positions of a row average 8.0 nats, positions 16–127 4.0. On the 426,985 (document, position) pairs both evaluations score, the context-cut pairs cost +0.450 nats (3.343 vs 2.892) and the intact pairs only +0.016 (2.951 vs 2.935); cut tokens are 49% of the set, so the context cut accounts for roughly 0.2 of the 0.24 nats, and the choice of tokens (the contiguous set scores the first 932 documents whole, incl. 297K tokens at document positions ≥ 2048; best-fit favours documents that fit a row and crops the rest) for the remainder. Same pattern for Muon (gap 0.268) and the old AdamW (0.246). Neither number is wrong; the fine-tuning D0 and D1 losses are measured on partly context-starved tokens. Forgetting (a difference on identical tokens) is still consistent within the study; a fix is proposed, not applied (it would change the pilot's measurements).
- **Fine-tuning evaluation switched to best-fit packing (2026-11-25, user decision: "fix it now and rerun the pilot").** `nanochat/eval_set.py`: `build_eval_set(..., packing="bestfit")` (default) builds the pretraining loader's rows (`bestfit_rows`: same 128-document read chunks, 1,000-document buffer, largest-fit / crop-shortest) while keeping each token's document (`doc_rows` = file row per compact document ID); documents are read once instead of wrapping. `--eval-packing` in `scripts/finetune.py` (recorded in `config.json`); `contiguous` keeps the old packing. Test `test_bestfit_eval_set_reproduces_the_pretraining_loader_rows` (rows identical to the loader's; every scored target is the right token of its document; `bestfit_capacity` exact). On the real D0: rows 0–349 identical to the pretraining held-out evaluation; from row 350 the pretraining loader had wrapped to the start of D0 (1,881 documents) and scored documents a second time, which the new set does not: 1,047,614 scored tokens from 1,474 documents. StarCoderData D1: 1,047,642 tokens, 1,446 documents.
- **WikiText D1 resized for best-fit** (job 30499948, `mit_data_transfer`): with best-fit each row holds about one article's beginning, so the 268-article D1 filled only 220 rows. `prepare_wikitext103(d1_rows=512)` now adds seeded training articles (10 at a time) until `bestfit_capacity` reaches 512 rows: 632 articles (60 validation + 62 test + 510 training), 2.50M tokens; the evaluation scores 1,048,461 tokens from 627 articles. Training 28,934 articles, 110.6M tokens. Manifest in `results/e8/wikitext103_manifest.json`.
- **Best-fit throughput of the fine-tuning training sets** (lengths simulated with `bestfit_capacity`, file order): WikiText one pass = 23,924 rows = 49.0M training tokens (55% of its tokens are cropped away; articles average ~3,800 tokens), so the 960-step (62.9M-token) length needs 1.28 passes; StarCoderData one pass = 105,372 rows = 215.8M tokens (0.29 passes). Open: how to handle WikiText at 960 steps.
- **Pilot rerun with the new evaluation:** the old pilot jobs 30493469/30493470 cancelled at 14:36 (5 of 15 trunks done each); their outputs moved to `$SCRATCH/finetune_pilot_v0_contiguous_eval/` (deletable). Resubmitted the mismatched pairs as 30500290 (AdamW s42 → Muon) and 30500292 (Muon s42 → AdamW), scripts in `/home/hgo/jobs/ft-pilot-v1/`, outputs `$SCRATCH/finetune_pilot/`. Nazar runs the matched pairs from commit 0c86b24 or later.

- **Seed-42 fine-tuning checkpoint pin (2026-11-25):** `scripts.finetune` defaults to the shared HGO base checkpoint directory and step 11,445, accepts only the AdamW/Muon seed-42 run names from that directory, checks metadata and SHA-256 of `model_011445.pt` before loading, and records the model path/hash in each fine-tuning `config.json`. Generated Slurm jobs pass the same directory and step explicitly. Explicit custom checkpoint directories remain available for tests or future studies.

- **Fine-tuning analysis script (2026-11-25):** `scripts/analyze_finetune.py` validates completed seed-42 E=8 grids and paired data identity, selects each arm's LR by mean final D1 loss across seeds, flags grid-edge winners, finds the minimum interpolated forgetting at a predeclared common learning level, calculates both optimizer-mismatch contrasts, reads per-layer routing diagnostics from JSONL, and saves CSV/JSON tables plus plots. Its paired seed/document bootstrap holds selected LRs fixed and checks per-document arrays against result curves. Four targeted tests cover threshold geometry, complete LR grids, paired resampling, and end-to-end output. The HGO pilot is still incomplete; no scientific result has been reported from it.

- **Analysis across the split pilot:** `scripts/analyze_finetune.py` accepts repeated `--root` arguments, combines hgo's mismatched-pair outputs with Nazar's matched-pair outputs, and checks common evaluation data, training data, and paired data fingerprints. A synthetic end-to-end test splits the two base models across separate output roots.

- **Analysis script checks the evaluation packing (2026-11-25):** `scripts/analyze_finetune.py` reads the `packing` field (`bestfit` or `contiguous`) from each run's evaluation identity and requires every combined run, across repeated `--root` arguments, to share one packing and the same D0 and D1 identity (file hash, rows, sequence length, documents, scored tokens). A run without the field is treated as the original `contiguous` packing and is refused in a mix with `bestfit` runs; mixed packings or an unknown value stop the analysis with a plain message. The per-document arrays are checked against the config (document count, scored tokens) and, when `base/eval_set_D0.pt` and `eval_set_D1.pt` are present, against the saved sets (compact document IDs, matching per-document counts, one increasing file row per document, and no best-fit document split across rows). `analysis.json` and the printed summary now state the packing and how many runs lacked the field. Tests: synthetic grids for both packings, legacy and mixed cases, corrupted saved sets, and real sets built by `build_eval_set`.
- **Restarted pilot sanity check passed:** AdamW s42 base D0 loss under the new evaluation 2.9386 nats/token (pretraining 2.932; the 0.007 difference is the pretraining evaluation scoring some D0 documents twice), StarCoderData D1 2.8495; `config.json` records `packing: bestfit` for D0 and D1, 1,474 D0 documents.
- **StarCoderData D1 composition (2026-11-25, user: option 1):** keep best-fit; D1 = 1,446 whole Python files of ≤ 2,048 tokens (median 663; 26% of D1's files are longer than a row and never scored; 301 short files are cropped only to fill row ends). WikiText D1 by contrast scores 627 of 632 articles, 509 of them by their first 2,048 tokens (81% of scored tokens). Recorded in the protocol and as a `% NOTE` in the report.
- **Report prose:** reverted my added clause in the report's fine-tuning LR item to the authors' original text and moved the content into a `% NOTE` comment (the report states the authors did not use AI to draft it).
- **WikiText pilot length (2026-11-25, user):** 60 / 240 / 720 steps (`--branch-tokens 3932160 47185920`) instead of 960, so the longest run stays within one best-fit pass of WikiText's training split (23,040 of 23,924 rows); 960 steps would have drawn its last 6,796 rows (22%, from about step 748, i.e. most of the decay) from a second pass. The 60- and 240-step runs equal the StarCoderData pilot's. Not yet assigned or submitted.
- **StarCoderData pilot, mismatched pairs done (best-fit evaluation):** 30500290 (AdamW s42 → Muon, 1 h 17 min) and 30500292 (Muon s42 → AdamW, 1 h 09 min), all 90 runs completed (15 trunks × 3 lengths each). Mean over seeds 1–3, learning / forgetting (nats/token) at 60 / 240 / 960 steps:
  AdamW s42 → Muon: 5.55e-5 0.767/0.062, 1.142/0.205, 1.455/0.521; 1.11e-4 0.839/0.072, 1.247/0.220, 1.541/0.509; 2.22e-4 0.943/0.091, 1.362/0.249, 1.617/0.507; 4.44e-4 1.076/0.129, 1.458/0.305, 1.678/0.563; **8.87e-4** 1.194/0.216, 1.516/0.460, 1.710/0.802 (lowest D1 loss at every length = top of the grid → edge).
  Muon s42 → AdamW: 8.19e-5 1.043/0.150, 1.331/0.299, 1.534/0.541; 1.64e-4 1.111/0.205, 1.387/0.368, 1.584/0.601; 3.28e-4 **1.150**/0.319, 1.425/0.520, 1.620/0.787; 6.55e-4 1.123/0.635, **1.436**/0.860, **1.631**/1.206; 1.31e-3 0.792/1.757, 1.387/1.620, 1.606/1.897 (selected 3.28e-4 at 60 steps, 6.55e-4 at 240 and 960; interior).
  Seeds agree to ≤ 0.005 in learning and mostly ≤ 0.03 in forgetting (AdamW 1.31e-3 at 60 steps is unstable: learning 0.65–0.88). Learning still rises 0.2–0.3 nats from 240 to 960 steps while forgetting roughly doubles. For AdamW fine-tuning the D1-selected LR buys little learning for much forgetting (240 steps: 6.55e-4 vs 3.28e-4, +0.011 learning, +0.34 forgetting). No mismatch comparison yet (needs Nazar's matched pairs).
- **Edge extension submitted** (protocol edge rule): job 30512566, AdamW s42 → Muon at ×2 and ×4 the Muon pretraining LR (1.775e-3, 3.550e-3), seeds 1–3, same branches and output root; scripts in `/home/hgo/jobs/ft-pilot-ext/` (commit 65155fa; numerically the same fine-tuning code as 0c86b24 plus Nazar's checkpoint pin).
- **Main-grid decisions (2026-11-25, user):** 3 fine-tuning seeds; fine-tune all three base-model seeds (42, 43, 44); seeds 43 and 44 reuse the LRs selected on seed 42 (no grid); ℓ0 chosen after seeing the curves (protocol adds a safeguard: pick it from the learning curves before computing the forgetting contrast, and also report the contrast across the whole common learning range). Main fine-tuning length still open. Note: `FINETUNE_BASE_CHECKPOINTS` (Nazar's pin) admits only the seed-42 bases; seeds 43/44 need adding (with their SHA-256) before the main grid.
- **Seed-42 StarCoderData learning-rate selection final (2026-11-25):** extensions 30512566 (hgo, AdamW → Muon), 30515353 (Nazar, Muon → Muon) and 30521188 (hgo, AdamW → AdamW) all at 2× and 4×, 29–30 min each. All 78 cells have 3 completed seeds (`results/e8/finetune_pilot_starcoderdata_summary.json`: mean learning, forgetting, D1 loss per pair, length, LR). Selected (lowest mean final D1 loss), 60 / 240 / 960 steps: AdamW → AdamW 1.31e-3 / 1.31e-3 / 6.55e-4; AdamW → Muon 1.77e-3 / 1.77e-3 / 8.87e-4; Muon → AdamW 3.28e-4 / 6.55e-4 / 6.55e-4; Muon → Muon 1.77e-3 / 8.87e-4 / 8.87e-4; all inside their grids. Close calls: AdamW → Muon at 60 steps (1.77e-3 vs 3.55e-3, 0.0008 nats of D1 loss) and AdamW → AdamW at 960 (6.55e-4 vs 1.31e-3, 0.002). AdamW → AdamW at 5.24e-3 diverged at 60 and 240 steps (learning −1.9 and −0.2, forgetting 6.5 and 5.0). These are the rates the seed-43/44 main grid uses. Pilot report: https://claude.ai/artifact/2uN17JfULvGvtcTFzYUgKx (private to the user until shared).
- **WikiText pilot assigned to Nazar (2026-11-25, user):** HANDOFF.md §5 now gives the full procedure on Nazar's account, following the StarCoderData pilot's conventions: commit ed76421 or later; shared WikiText data (no rebuild); `finetune_jobs` with `--branch-tokens 3932160 47185920` (60 / 240 / 720 steps), seeds 1–3, 5-point grid, output root `/orcd/scratch/orcd/010/ndrugov/finetune_pilot`; early checks (best-fit packing, branch lengths, 8 warmup steps, base D0 ≈ 2.94 / 2.90, D1 1,048,461 tokens from 627 articles); edge extensions until interior; `analyze_finetune` per length; summary to `results/e8/finetune_pilot_wikitext103_summary.json`. §6 next steps renumbered (pin seeds 43/44, main grid, analysis).
