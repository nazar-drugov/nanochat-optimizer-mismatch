# E=8 MoE pretraining: hyperparameters, LR sweep, transfer check and base models

2026-11-25 · Harry

Exported 2026-11-26 from https://claude.ai/artifact/TP3GZWWwJHK2EqVu3GHdyk (doc revision 6).

All six E=8 base models (AdamW and Muon, seeds 42–44, 3B tokens) are trained. Muon reaches lower loss at every seed, by 0.013 bpb on average.

## Agreed hyperparameters

Every cell shares one recipe; AdamW and Muon differ only in the optimizer used for the matrices and router and in its learning rate.

| Setting | Value |
|---|---|
| Model | nanochat GPT, depth 10, width 640, 5 heads of 128, context 2,048; 118M active parameters (with embeddings), 376M total |
| MoE | E = 8 experts in all 10 layers; top-1 Switch softmax gate, dropless; SwiGLU experts, hidden 1,920 (3 × width); fp32 router |
| Balancing | Switch balance loss 0.01 + router z-loss 0.001, each summed over the 10 layers |
| Data and tokenizer | FineWeb-Edu 100B sample; GPT-2 tokenizer (50,257 tokens) |
| Token budget | Base models 3,000,238,080 tokens (11,445 steps); sweep 499,908,608 tokens (1,907 steps) |
| Batch | 262,144 tokens per step, constant (device batch 32 × 2,048, 4 accumulation steps) |
| Schedule | Linear warmup over 130M tokens (496 steps; 83 in the sweep, the same 4.33%), constant, linear decay to 0 over the last 20% |
| Matrix LR (base models) | AdamW 1.311e-3; Muon 8.874e-4 (from the sweep and transfer check below) |
| AdamW | β = (0.9, 0.95), ε = 1e-8 |
| Muon | Moonlight variant: Nesterov momentum 0.95, 5 Newton–Schulz steps, update scale 0.2 · √max(m, n) |
| Weight decay | 0.1 on matrices for both optimizers, applied as 1 − lr · λ; router 0; embedding 0.001; unembedding 0.01 |
| Router | Same optimizer and LR as the matrices |
| Embedding / unembedding | AdamW in both cells; LR 0.329 / 0.00876 (0.3 / 0.008 scaled by (640/768)^−0.5) |
| Gradient clipping | Global norm 1.0 for both optimizers |
| Seeds | 42, 43, 44; the seed changes the initialization only, the data order is fixed |
| Validation (selects LRs) | shard_01822 minus the 1,881 documents D0 took from it; 10,485,760 tokens every 500 steps |
| D0 (forgetting set) | 1,048,576 tokens, evaluated at the same steps and only logged |
| Hardware | One B200 per run, bf16 compute |

## LR sweep (0.5B tokens)

The best LR at 0.5B tokens is 1.155e-3 for AdamW and 1.575e-3 for Muon, and Muon reaches lower loss at every LR. Each optimizer is fitted with a parabola in log₂(LR) through the best point and its two neighbours.

| Matrix LR | AdamW, final val bpb | Muon, final val bpb |
|---|---|---|
| 2.5e-4 | 1.0797 | – |
| 5e-4 | 1.0432 | 1.0278 |
| 1e-3 | **1.0197** | 1.0060 |
| 2e-3 | 1.0294 | **1.0034** |
| 4e-3 | 1.0579 | 1.0176 |
| 8e-3 | – | 1.0396 |
| **Fitted optimum** | **1.155e-3** | **1.575e-3** |

- The first grid (AdamW 1e-3 to 4e-3, Muon 2e-3 to 8e-3) put both optima at its low edge, so it was extended down by ×½ and ×¼.
- No run diverged; 1–3% of steps were clipped.
- Seed 42, one run per point; 15–16 min per run, 2.6 GPU-hours in total.
- Choosing on held-out D0 instead of validation moves either optimum by at most 0.01 octave.

## LR transfer check (3B tokens)

The standard transfer from 0.5B to 3B tokens held for Muon but not for AdamW. AdamW's 3B optimum is about 2× the transferred LR, so the AdamW base models were retrained at 1.311e-3.

The transferred LR is the sweep optimum × (3B / 0.5B)^−0.32 = ×0.5635. The −0.32 exponent is taken from Bjorck et al. (AdamW) and Ren et al. (Muon), via Hernanz. To test it, seed 42 was trained for the full 3B tokens at multiples of the transferred LR; ×1 is the original base model.

| Multiple of transferred LR | AdamW LR | AdamW val bpb | Muon LR | Muon val bpb |
|---|---|---|---|---|
| ×0.5 | 3.26e-4 | 0.9280 | 4.44e-4 | 0.9043 |
| ×1 (transferred) | 6.51e-4 | 0.9180 | 8.87e-4 | **0.9015** |
| ×2 | 1.30e-3 | **0.9130** | 1.77e-3 | 0.9050 |
| ×4 | 2.60e-3 | 0.9178 | – | – |
| ×8 | 5.21e-3 | 0.9345 | – | – |
| **Fitted 3B optimum** | **1.311e-3** | | **8.54e-4** | |
| Offset from transferred LR | +1.01 octave | | −0.06 octave | |
| Implied exponent (assumed −0.32) | +0.07 | | −0.34 | |

- **Rule agreed beforehand:** retrain the base models if the 3B optimum is more than 0.25 octave from the transferred LR. AdamW was retrained at 1.311e-3 (all three seeds); Muon kept 8.874e-4.
- **Effect at seed 42:** AdamW improved from 0.9180 to 0.9126 bpb, which shrank Muon's lead from 0.0165 to 0.011.
- **AdamW's 3B optimum lies above its 0.5B optimum** (1.155e-3), so its LR does not fall with training length here.
- **Fitting on held-out D0 gives the same optima** within 0.015 octave.
- **Cost:** 6 extra runs, 8.3 GPU-hours, plus 1.4 GPU-hours for the superseded AdamW base model.
- **Caveat:** one seed only. AdamW's seed spread (sd 0.0025) is comparable to the ×1–×2 gap, but the five AdamW points form one smooth minimum.

## Base models (3B tokens)

Muon is better at every seed: mean validation perplexity is 18.58 for Muon and 19.39 for AdamW, a gap of 0.013 bpb.

| Optimizer (matrix LR) | Seed | Val bpb | Val perplexity | D0 bpb | D0 perplexity | Training time |
|---|---|---|---|---|---|---|
| AdamW (1.311e-3) | 42 | 0.9126 | 19.29 | 0.9010 | 18.76 | 78 min |
| AdamW (1.311e-3) | 43 | 0.9127 | 19.30 | 0.9008 | 18.75 | 78 min |
| AdamW (1.311e-3) | 44 | 0.9171 | 19.57 | 0.9055 | 19.04 | 77 min |
| **AdamW mean ± sd** | | **0.9141 ± 0.0025** | **19.39 ± 0.16** | **0.9025 ± 0.0027** | **18.85 ± 0.16** | |
| Muon (8.874e-4) | 42 | 0.9015 | 18.61 | 0.8902 | 18.11 | 74 min |
| Muon (8.874e-4) | 43 | 0.9010 | 18.58 | 0.8896 | 18.07 | 74 min |
| Muon (8.874e-4) | 44 | 0.9005 | 18.55 | 0.8887 | 18.02 | 74 min |
| **Muon mean ± sd** | | **0.9010 ± 0.0005** | **18.58 ± 0.03** | **0.8895 ± 0.0008** | **18.07 ± 0.05** | |

- **Muon's lead per seed:** 0.011, 0.012 and 0.017 bpb.
- **AdamW varies about 5× more between seeds than Muon.** AdamW seed 44 trails seeds 42/43 by a steady ~0.004 bpb from step 1,000 on.
- **All six runs were stable:** none clipped any step after warmup, each saved 8 checkpoints (every 1,526 steps plus the final one), and each took about 1 h 25 min of wall clock including evaluation.
- **Superseded AdamW model:** the first AdamW model, at the transferred LR 6.51e-4 with seed 42, reached val 0.9180 bpb (perplexity 19.63). It is kept only as the ×1 point of the transfer check and will not be fine-tuned.
- **Perplexity definition:** per GPT-2 token, perplexity = 2^(bpb × bytes per token), computed on the same eval tokens as bpb (BOS excluded). The validation set averages 4.679 bytes per token and D0 4.694.
- **Where the data is:** each run's `result.json` and `config.json` under `/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/base_checkpoints/`, readable by all ORCD users. The LR selections are in `~hgo/jobs/lr_selection.json`, `transfer_check.json` and `lr_selection_3b.json`. The full log is in LOG_CHANGES.md in the repository.
