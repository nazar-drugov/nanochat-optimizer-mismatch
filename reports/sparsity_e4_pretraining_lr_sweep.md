# E=4: 0.5B-token pretraining LR sweep

**2026-11-27, 18:14 EST · Complete; 3B transfer checks queued.**

Seed 42, same active-size architecture, frozen training/validation distribution and WSD schedule as E=8; default router coefficients (balance .01, z .001). Five independent LR trials per optimizer; final non-D0 validation bpb selects the optimum. All 10 trials completed and 10 W&B runs were verified `finished` in `optimizer-mismatch-e4/pretrain/sweep`.

| Optimizer | Peak LR | Final validation bpb |
|---|---:|---:|
| adamw | 0.000353553391 | 1.066326 |
| adamw | 0.000707106781 | 1.031329 |
| adamw | 0.00141421356 | **1.024723** |
| adamw | 0.00282842712 | 1.051904 |
| adamw | 0.00565685425 | 1.064100 |
| muon | 0.000840896415 | 1.013545 |
| muon | 0.00168179283 | **1.007523** |
| muon | 0.00336358566 | 1.016870 |
| muon | 0.00672717132 | 1.028148 |
| muon | 0.0134543426 | 1.203461 |

| Optimizer | Fitted 0.5B peak LR | Provisional transferred 3B LR |
|---|---:|---:|
| adamw | 0.001145145118 | 0.000645380282 |
| muon | 0.001560277173 | 0.0008793401869 |

Both minima are interior. These are separate log₂-LR local quadratic fits, not constrained by E1/E8. The transferred LRs are not final 3B selections until checked at ×½, ×1 and ×2. The coordinator already submitted the six checks (30750304, 30750305, 30750306, 30750307, 30750308, 30750310). E1 transfer checks have two running and four queued; E16 verification remains lower priority. See [full selection data](../results/e4/lr_selection.json). No optimizer mismatch or cross-E conclusion is drawn from these pretraining results.
