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
- **Muon variants:** `plain` (default: Moonlight 0.2·√max(m,n) update scaling plus decoupled weight decay) and `keller` (the previous original-Muon scaling). Weight decay applies in both. Muon momentum is constant 0.95 by default; nanochat's schedule is available with `--muon-momentum-schedule nanochat`.
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
