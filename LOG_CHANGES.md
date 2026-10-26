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
