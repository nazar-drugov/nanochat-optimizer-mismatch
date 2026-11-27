# E=1 dense control: 0.5B-token LR sweep

**2026-11-27, 17:14 EST · Complete; 3B transfer checks queued.**

Seed 42, 0.5B tokens, fixed active size, router loss disabled (both aux and z), the same pretraining data and WSD schedule as E=8. Five points per optimizer, selected solely by final held-out validation bpb (D0 excluded). All 10 results completed and all 10 corresponding W&B runs are `finished` in `optimizer-mismatch-e1/pretrain/sweep`.

| Optimizer | Peak LR | Final validation bpb |
|---|---:|---:|
| AdamW | 0.0005 | 1.055944 |
| AdamW | 0.001 | 1.032033 |
| AdamW | 0.002 | **1.025406** |
| AdamW | 0.004 | 1.049115 |
| AdamW | 0.008 | 1.141949 |
| Muon | 0.001 | 1.019215 |
| Muon | 0.002 | 1.009312 |
| Muon | 0.004 | **1.008154** |
| Muon | 0.008 | 1.014598 |
| Muon | 0.016 | 1.027548 |

(The table values are rounded; see the [exact selection JSON](../results/e1/lr_selection.json).)

| Optimizer | Fitted 0.5B LR | Transferred 3B LR (provisional) |
|---|---:|---:|
| AdamW | 0.001645416246 | 0.000927322821 |
| Muon | 0.003143392431 | 0.001771551450 |

Both grid minima are interior; the local log₂-LR quadratic uses the best grid point and its two neighbors. The 3B LRs above are provisional transferred values, not final selections. The coordinator submitted six independent 3B transfer checks at ×½, ×1, and ×2 (30743795, 30743796, 30743797, 30743798, 30743800, 30743801), queued behind the ongoing E=4 sweep. E=4 is 4/10 completed and E=16 smoke is queued at lower priority. No forgetting or cross-sparsity inference follows from pretraining LR validation alone.
