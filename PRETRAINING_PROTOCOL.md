# Pretraining protocol

Decisions and defaults for the pretraining stage of the optimizer-mismatch study,
recorded 2026-11-23. Fine-tuning and analysis are parked; see `EXPERIMENT_SPEC.md`.
Every run writes its fully resolved settings to `config.json` next to its checkpoints.

## Decided

| Setting | Value | Why |
|---|---|---|
| Data | FineWeb-Edu 100B sample (`karpathy/fineweb-edu-100b-shuffle`) | Spec decision |
| Tokenizer | GPT-2 (`--tokenizer gpt2`, 50,257 tokens) | Matches Ludziejewski et al. (arXiv:2502.05172); no retraining |
| Routing | Top-1, Switch-style softmax gate over all E, dropless | Spec |
| Router optimizer | Same as the matrices (`--router-optimizer same`) | Spec decision |
| Muon variant | `plain`: Nesterov 0.95 (constant), Newton–Schulz ×5, update scale 0.2·√max(m,n), decoupled weight decay | Spec recommendation; nanochat's variant dropped |
| Batch size | 262,144 tokens, identical in every cell | Ludziejewski's batch at this scale; DeepSeek LLM's fit gives ~240K |
| Schedule | 130M-token linear warmup, constant, linear decay to 0 over the last 20% | Ludziejewski et al. Sec. 5.1.3 |
| Gradient clipping | Global norm 1.0 in every cell | Spec recommendation, applied to both optimizers so cells differ only in the optimizer |

## Defaults that still need team confirmation

| Setting | Default | Alternative |
|---|---|---|
| Balancer | Auxiliary loss 0.01 + z-loss 0.001 (`--balancer aux`) | Bias balancer (`--balancer bias`, u = 0.001) as an ablation |
| Weight decay on matrices | 0.1 for AdamW and Muon | The spec gives no Muon value |
| Router learning rate | Same as the matrix LR; router weight decay 0 | Sweep separately |
| AdamW | β = (0.9, 0.95), ε = 1e-8 | ε = 1e-10 |
| Model | depth 10 (width 640, 10 blocks): 118M active with GPT-2 vocab | depth 12 costs ~1.55× |
| Token budget | 3,000,238,080 tokens (11,445 steps) for every E | Decay branches at 1B and 2B for robustness |
| Pretraining seeds | 3 at E = 8, 1 elsewhere | |

## Learning-rate sweep (Option A)

Chosen 2026-11-23: sweep on short runs, then transfer to the full budget.

1. **What is tuned:** only the peak LR of the matrices and routers, separately for each
   (optimizer, E): 8 cells. Embedding and lm_head LRs stay at their defaults in every cell.
2. **Grid:** 5 points spaced ×2 around a center (c/4, c/2, c, 2c, 4c). AdamW center
   2e-3 · E^-0.25 (Ludziejewski et al. Eq. 8): 2e-3, 1.41e-3, 1.19e-3, 1e-3 for E = 1, 4, 8, 16.
   Muon center 4e-3 (Hernanz). Extend the grid if the best point is on an edge.
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

Example cell:

```
torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --depth=10 --ffn-type=moe --n-experts=8 --optimizer-name=muon \
    --muon-matrix-lr=<swept> --train-tokens=3000238080 --save-every=1000 \
    --eval-tokens=10485760 --core-metric-every=-1 --seed=42
```
