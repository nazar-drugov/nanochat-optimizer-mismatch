# Sparsity pretraining: comparison with E=8

**Snapshot: 2026-11-27, 23:19 EST.** Seed 42 for the new architectures. All
three staged coordinators are running. No E=1/4/16 fine-tuning has started.

| Experts | 0.5B sweep | 3B transfer check | Final base selection |
|---:|---|---|---|
| 1 | 10/10 complete | 6 complete; extra AdamW ×4 queued | Muon selected, AdamW pending |
| 4 | 10/10 complete | 2 AdamW running, 4 queued | Pending |
| 8 | 10/10 complete historically | 8 optimizer/LR points including original Muon base | Both selected; seeds 42–44 complete |
| 16 | Smoke jobs queued | Not started | Pending |

The new jobs use the same ~118M active parameters and data as E=8. Total
parameters increase from 117.6M to 228.3M, 375.7M, and 670.7M, respectively.
E=1 router losses are disabled. E=16 has not yet produced any measurement.

## Measured losses

These are **measured best grid points**, not predicted parabola minima. Val
is the intact LR-selection validation distribution (D0 documents excluded);
D0 is the forgetting set. Both lower is better. Perplexity is approximately
2^(bpb × bytes/token), using 4.679 for validation and 4.694 for D0. Actual
top-1 next-token accuracy was never logged, and cannot be inferred from loss.

### 0.5B tokens (all completed cells, seed 42)

| E | Optimizer | Matrix LR | Val bpb | Val PPL | D0 bpb | D0 PPL |
|---:|---|---:|---:|---:|---:|---:|
| 1 | AdamW | 0.002 | 1.025406 | 27.82 | 1.011332 | 26.86 |
| 1 | Muon | 0.004 | 1.008154 | 26.30 | 0.993679 | 25.36 |
| 4 | AdamW | 0.00141421 | 1.024723 | 27.76 | 1.010555 | 26.79 |
| 4 | Muon | 0.00168179 | 1.007523 | 26.25 | 0.992998 | 25.30 |
| 8 | AdamW | 0.001 | 1.019728 | 27.31 | 1.005563 | 26.36 |
| 8 | Muon | 0.002 | 1.003395 | 25.90 | 0.989075 | 24.98 |

E=8 beats E=1 and E=4 by about 0.004–0.006 val bpb. E=1 and E=4 are
separated by <0.001 val bpb for either optimizer, but these are only one
seed and measured grid minima (not repeat-seed estimates or fitted optima).
Muon beats AdamW by about 0.016–0.017 bpb in every architecture so far.

### 3B tokens (completed measurements only, seed 42)

| E | Optimizer | LR | Val bpb | Val NLL (nats/token, approx.) | Val PPL | D0 bpb | D0 PPL |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | AdamW | 0.00185465 | 0.933796 | 3.0285 | 20.67 | 0.920963 | 20.01 |
| 1 | Muon | 0.00177155 | 0.926242 | 3.0040 | 20.17 | 0.913711 | 19.55 |
| 8 | AdamW | 0.00131076 | 0.912642 | 2.9599 | 19.30 | 0.901014 | 18.76 |
| 8 | Muon | 0.000887386 | 0.901511 | 2.9238 | 18.61 | 0.890245 | 18.11 |

E=1 AdamW is currently the best measured transfer trial, not the selected
base. E=1 Muon is the verified selected LR. E=8's 3-seed mean ± sample SD
is 0.9141 ± 0.0025 bpb for AdamW and 0.9010 ± 0.0005 for Muon. The seed-42
E=1 deficits relative to E=8 are 0.02115 and 0.02473 bpb, respectively,
larger than the observed E=8 seed spread but not a substitute for replicating E=1.
Cannot yet infer a 3B trend across all sparsities.

E=4 live W&B measurements (not final, not used for selection):
[AdamW ×0.5](https://wandb.ai/hgo-mit/nanochat/runs/4vj5rhwu) at step 10,000:
val 0.964033, D0 0.951330; [AdamW ×1](https://wandb.ai/hgo-mit/nanochat/runs/8aq7xd9n)
at step 9,000: val 0.976671, D0 0.963314. Same-step E=8 final-LR AdamW
comparison is 0.948660 / 0.936093 at 10,000 and 0.971938 / 0.959008 at 9,000.
All runs have 11,445 steps and the final decay is still underway.
W&B verified finished: E1 verification 2, sweep 10, transfer 6; E4 verification 2,
sweep 10; E4 transfer 2 running. No E16 records yet.

## Matrix LR selections

The local log₂-LR quadratic is fitted separately per architecture and
optimizer at 0.5B; the starting 3B LR is multiplied by 6^(-0.32) ≈ 0.56358.
We then check ×0.5, ×1, ×2 with edge extensions; retain the transferred LR
if the fitted 3B vertex differs by at most 0.25 octave, otherwise use the
fitted full-length LR. Never select on D0. Rates below in units of 10^-3.

| E | Optimizer | 0.5B fitted | 3B transferred | 3B fitted/selected | State |
|---:|---|---:|---:|---|---|
| 1 | AdamW | 1.645416 | 0.927323 | — | ×2 best, ×4 trial queued |
| 1 | Muon | 3.143392 | 1.771551 | 1.771551 selected (vertex 2.022075) | Confirmed |
| 4 | AdamW | 1.145145 | 0.645380 | — | Check running |
| 4 | Muon | 1.560277 | 0.879340 | — | Check queued |
| 8 | AdamW | 1.155320 | 0.651115 | 1.310763 selected | Complete |
| 8 | Muon | 1.574554 | 0.887386 | 0.887386 selected | Complete |
| 16 | Both | — | — | — | Smoke queued |

E=1 transfer validation bpb:

| Factor | AdamW LR | AdamW val bpb | Muon LR | Muon val bpb |
|---:|---:|---:|---:|---:|
| 0.5 | 0.000463661 | 0.947206 | 0.000885776 | 0.928548 |
| 1 | 0.000927323 | 0.937094 | 0.001771551 | **0.926242** |
| 2 | 0.001854646 | **0.933796** | 0.003543103 | 0.927274 |
| 4 | 0.003709291 | Queued | — | — |

E=1 Muon's vertex offset is +0.191 octave, within the tolerance; reuse its
completed center checkpoint. E=1 AdamW echoes E=8's upward deviation from the
token-budget transfer rule; its optimum is not yet bracketed. E=4 and E=8
short-run optima nearly coincide; E=1's Muon optimum is roughly twice the
E=8 optimum and its AdamW optimum about 1.42×. No cross-E regression was
used for selection.

Sources: cluster base checkpoint result.json files and matching W&B records,
[initial E1 transfer report](../results/e1/transfer_check.json),
[provisional E1 3B selection](../results/e1/lr_selection_3b.json),
[E=8 summary](pretraining_summary.md), [E1 sweep](sparsity_e1_pretraining_lr_sweep.md)
and [E4 sweep](sparsity_e4_pretraining_lr_sweep.md).
