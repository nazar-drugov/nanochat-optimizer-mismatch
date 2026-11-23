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
| Gradient clipping | Global norm 1.0 in every cell | Spec recommendation, applied to both optimizers so cells differ only in the optimizer |
| Balancer | Auxiliary loss 0.01 + z-loss 0.001 (`--balancer aux`) | Following Ludziejewski et al. (arXiv:2502.05172): Switch balance loss 0.01 and router z-loss 0.001, each **summed over the MoE layers** (not averaged); see "Auxiliary loss" below |


## Defaults that still need team confirmation

| Setting | Default | Alternative |
|---|---|---|
| Weight decay on matrices | 0.1 for AdamW and Muon | The spec gives no Muon value. Moonlight uses 0.1 in all training stages (paper) and as the default in its code, shared by Muon and AdamW |
| Router learning rate | Same as the matrix LR; router weight decay 0 | Sweep separately |
| AdamW | β = (0.9, 0.95), ε = 1e-8 | ε = 1e-10 |
| Model | depth 10 (width 640, 10 blocks): 118M active with GPT-2 vocab | depth 12 costs ~1.55× |
| Token budget | 3,000,238,080 tokens (11,445 steps) for every E | Optional 1B- and 2B-token runs can test whether the result depends on the training horizon; see below. |
| Pretraining seeds | 3 at E = 8, 1 elsewhere | |



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

## Learning-rate sweep (Option A)

Chosen 2026-11-23: sweep on short runs, then transfer to the full budget.

1. **What is tuned:** only the peak LR of the matrices and routers, separately for each
   (optimizer, E): 8 cells. Embedding and lm_head LRs stay at their defaults in every cell.
2. **Grid:** 5 points spaced ×2 around a center (c/4, c/2, c, 2c, 4c). AdamW center
   2e-3 · E^-0.25 (Ludziejewski et al. Eq. 8): 2e-3, 1.41e-3, 1.19e-3, 1e-3 for E = 1, 4, 8, 16.
    Muon center 4e-3 (Hernanz), swept independently of AdamW's result. Extend the grid if the best point is on an edge.
3. **Sweep length:** 499,908,608 tokens (1,907 steps) with the full schedule shape
   (130M-token warmup, 20% decay).
4. **Selection:** lowest final validation bits-per-byte, one seed. If neighbouring points are
   within seed noise (measure once with 2 seeds at the best point), take the smaller LR.
5. **Transfer to 3B tokens:** multiply by (3 / 0.5)^-0.32 = 0.564 (Hernanz's T^-0.32 scaling).
6. **Transfer check:** at E = 4, for both optimizers, train 3B-token runs at 0.5×, 1× and 2×
   the transferred LR. If 1× is not best, redo the transfer for all cells with the measured factor.
7. **Batch-size robustness (agreed):** at E = 4, 0.5B tokens, both optimizers × {128K, 256K, 512K}
   at the swept best LR.

Option B (decay branches off longer stable phases instead of transferring) is the fallback
if the transfer check fails.

## Before the first real run

- Download data: `python -m nanochat.dataset --dataset fineweb-edu-100b -n 100` (3B training
  tokens need ~4.6B raw tokens because the loader crops ~35%; ~55M tokens per shard).
- Cache the GPT-2 tokenizer on a node with internet access:
  `python -c "from nanochat.tokenizer import get_tokenizer; get_tokenizer('gpt2')"`.
- Build the held-out set D0 (~2M tokens, MinHash-deduplicated against training shards):
  `python -m scripts.make_heldout --dataset fineweb-edu-100b --tokenizer gpt2 --workers 32` (CPU job).
- Run the throughput benchmark (spec §8.8) and the two-GPU CUDA optimizer test
  (`tests/test_optim.py::test_two_rank_muon_matches_torch_reference`) on the cluster.
- On ORCD, use `runs/pretrain_study.sh` to download to pool and stage to
  scratch once per dataset version, then launch many runs using that scratch
  copy. Downloads and staging are not part of each training run. See
  `STORAGE_PATHS.md`.
- The data is frozen in `data_manifests/fineweb_edu_100_v1/` (100 training shards,
  the validation shard, and D0). Every `train` start and resume through the launcher
  hashes all shards against that manifest (about 30 s) and stops on any difference, so
  every cell of the study is guaranteed to read identical bytes. A resumed run also
  confirms that its shard list and tokenizer match the checkpoint. If the data ever
  changes, build a new D0 and a new manifest version; do not mix versions in one study.

Example cell preview (0.004 is the Muon sweep center, not a selected final LR):

```bash
RUN_NAME=moe-e8-muon-s42 bash runs/pretrain_study.sh dry-run \
    --depth=10 --ffn-type=moe --n-experts=8 --optimizer-name=muon \
    --muon-matrix-lr=0.004 --train-tokens=3000238080 --save-every=1000 \
    --eval-tokens=1048576 --core-metric-every=-1 --seed=42
```

Use `train` instead of `dry-run` only after the chosen learning rate and
remaining protocol defaults have been confirmed and the dataset staged.
