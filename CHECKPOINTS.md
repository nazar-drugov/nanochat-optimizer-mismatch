# Experiment and checkpoint registry

This is the canonical cross-author path map for the optimizer-mismatch study. It supersedes the historical path examples in `STORAGE_PATHS.md` for completed experimental outputs. The cluster is the source of the actual files; the checked-in CSV registry is a snapshot of their paths, not a copy of the model weights.

- [Run registry](results/checkpoint_registry/runs.csv): full run name and absolute directory, owner, stage, scientific validity/status, base identity, hyperparameters and paths where available.
- [Checkpoint registry](results/checkpoint_registry/checkpoints.csv): actual checkpoint step and file paths. Includes model weights, optimizer state, metadata, routing probes and evaluation snapshots as applicable.
- [W&B project](https://wandb.ai/hgo-mit/nanochat): organized pretraining and fine-tuning histories. Tracking records are not model checkpoints.

**Inventory snapshot: 2026-11-26, 22:35 EST.** 1,816 registry records (including 1,757 fine-tuning branch directories, 23 scientific pretraining runs, seven legacy verification runs and 29 shared base-evaluation references). The checkpoint table has 18,419 step records: 86 actual model-weight checkpoints, 18,300 fine-tuning evaluation snapshots, 29 shared base-evaluation references and four model-related-only routing steps. Seven incomplete archived branch directories are retained and marked. Both scientific long pilots are complete.

| Owner / fine-tuning stage | Completed branches | Incomplete |
|---|---:|---:|
| Harry, clean original pilot | 468 | 0 |
| Harry, clean main | 234 | 0 |
| Harry, clean long StarCoderData | 240 | 0 |
| Harry, clean long WikiText | 315 | 0 |
| Harry, replay-bug original archive | 126 | 0 |
| Harry, replay-bug long archive | 1 | 4 |
| Harry, incompatible contiguous archive | 54 | 3 |
| Nazar, pre-fix seed-42 archive | 288 | 0 |
| Nazar, clean unbranched WikiText main | 24 | 0 |

## Root directories

| Alias | Absolute ORCD path |
|---|---|
| `HARRY_SCRATCH` | `/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch` |
| `HARRY_POOL` | `/orcd/pool/006/hgo/nanochat-optimizer-mismatch/results` |
| `NAZAR_FT` | `/orcd/scratch/orcd/010/ndrugov/finetune_pilot` |
| `NAZAR_LEGACY` | `/orcd/scratch/orcd/010/ndrugov/nanochat-optimizer-mismatch` |

`HARRY_POOL` holds copied small result files and fine-tuning evaluation arrays when present. It is **not** guaranteed to hold model weights. The actual path columns in the registry identify what is present. Scratch and pool are not backups.

## Scientific pretraining model weights (Harry)

The six main bases were pretrained at E=8 on 3B tokens; their final model checkpoint is at step 11,445. Each row below is `HARRY_SCRATCH/base_checkpoints/<run name>/` followed by the model filename. These are the pinned bases used in every scientific fine-tuning run.

| Run name | Model checkpoints saved | Final weights |
|---|---:|---|
| `e8-main-adamw-s42` | 8 | `model_011445.pt` |
| `e8-main-adamw-s43` | 8 | `model_011445.pt` |
| `e8-main-adamw-s44` | 8 | `model_011445.pt` |
| `e8-main-muon-s42` | 8 | `model_011445.pt` |
| `e8-main-muon-s43` | 8 | `model_011445.pt` |
| `e8-main-muon-s44` | 8 | `model_011445.pt` |

All base runs use the same sequence of checkpoint steps: 1,526, 3,052, 4,578, 6,104, 7,630, 9,156, 10,682, 11,445. E.g. the final AdamW seed-42 model is:

```text
/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/base_checkpoints/e8-main-adamw-s42/model_011445.pt
```

The other scientific pretraining runs use the same directory naming convention:

| Stage | Run names | Saved model filename at final step |
|---|---|---|
| AdamW 0.5B LR sweep | `e8-sweep-adamw-lr0.00025-s42`, `e8-sweep-adamw-lr0.0005-s42`, `e8-sweep-adamw-lr0.001-s42`, `e8-sweep-adamw-lr0.002-s42`, `e8-sweep-adamw-lr0.004-s42` | `model_001907.pt` |
| Muon 0.5B LR sweep | `e8-sweep-muon-lr0.0005-s42`, `e8-sweep-muon-lr0.001-s42`, `e8-sweep-muon-lr0.002-s42`, `e8-sweep-muon-lr0.004-s42`, `e8-sweep-muon-lr0.008-s42` | `model_001907.pt` |
| AdamW 3B transfer check | `e8-xfer-adamw-lr0.000326-s42`, `e8-xfer-adamw-lr0.000651-s42`, `e8-xfer-adamw-lr0.0013-s42`, `e8-xfer-adamw-lr0.0026-s42`, `e8-xfer-adamw-lr0.00521-s42` | `model_011445.pt` |
| Muon 3B transfer check | `e8-xfer-muon-lr0.000444-s42`, `e8-xfer-muon-lr0.00177-s42` | `model_011445.pt` |

The superseded AdamW seed-42 base at LR 6.51e-4 (`e8-xfer-adamw-lr0.000651-s42`) retains eight model checkpoints and is a valid transfer-check run, **not** a fine-tuning base.

## Scientific fine-tuning runs and evaluation checkpoints

All modern fine-tuning cells initialize from a final base model above. Their directory convention is:

```text
<stage root>/<base run>/<dataset>/<ft optimizer>-lr<LR tag>-s<FT seed>-T<length>/
```

An unbranched standalone run may omit `-T<length>`; the exact run name and length are in the registry, never inferred solely from a rounded LR tag. Every completed length branch contains `config.json`, `result.json`, `metrics.jsonl`, `metrics.parquet`, and `ckpt_<step:06d>.npz` per measured step. **These NPZ files are evaluation checkpoints, not fine-tuned model weights.** They contain document loss sums, route flip counts, and the D0 route assignments needed for replay/analysis. Fine-tuned weights are not saved by this driver. Shared base evaluation artifacts live beside the fine-tuning cells at `<stage root>/<base run>/<dataset>/base/`.

| Account | Stage root | Scientific status |
|---|---|---|
| Harry | `HARRY_SCRATCH/finetune_pilot` | Clean seed-42 LR sweeps and selected original pilot lengths |
| Harry | `HARRY_SCRATCH/finetune_main` | Clean seed-43/44 selected-LR main grid (both datasets, excluding Nazar's standalone 720-step WikiText runs) |
| Harry | `HARRY_SCRATCH/finetune_pilot_long` | Clean seed-42 StarCoderData extension, including repeat short branches |
| Harry | `HARRY_SCRATCH/finetune_pilot_long_wikitext103` | Clean seed-42 WikiText extension; snapshot includes finalized and partial current runs |
| Harry | `HARRY_SCRATCH/finetune_pilot_v1_replaybug` | Archived pre-fix results; longer-than-60 branches invalid, 60-step branches unaffected but superseded |
| Harry | `HARRY_SCRATCH/finetune_pilot_long_v1_replaybug` | Archived pre-fix long-pilot attempts |
| Harry | `HARRY_SCRATCH/finetune_pilot_v0_contiguous_eval` | Archived, incompatible evaluation packing; never mix with best-fit results |
| Nazar | `NAZAR_FT` | Mixed: seed-42 pre-fix archived cells; seed-43/44 standalone 720-step WikiText cells are valid main results |

Example clean branch from Harry (AdamW base, WikiText, AdamW fine-tuning, FT seed 1, 720 steps):

```text
/orcd/scratch/orcd/014/hgo/public/nanochat-optimizer-mismatch/finetune_pilot/e8-main-adamw-s42/wikitext103/adamw-lr0.000655-s1-T720/ckpt_000720.npz
```

Use the registry's owner/stage/run-directory fields, not just the basename, to disambiguate clean, rerun and archived experiments. Nazar's analysis staging symlinks are aliases, not distinct experiments, and are excluded from the inventory.

## Nazar's earlier verification models

The following actual model-weight directories also exist and are tracked separately as **verification only**, not scientific E=8 results:

| Root relative to `NAZAR_LEGACY` | Run names | Model steps |
|---|---|---|
| `base_checkpoints/` | `smoke-adamw-moe-01`, `smoke-muon-moe-01` | 2, 4 |
| `base_checkpoints/` | `verify-moe-stats-29395802` | 1, 2 |
| `chatsft_checkpoints/` | `smoke-adamw-moe-01-step000004-sft-adamw`, `smoke-adamw-moe-01-step000004-sft-muon`, `smoke-muon-moe-01-step000004-sft-adamw`, `smoke-muon-moe-01-step000004-sft-muon` | 1, 2 |

Nazar's public study project directory contains the frozen data and D0 but no additional scientific pretraining model checkpoints were found there.

Inventory scope: the documented Harry and Nazar project roots above are readable from hgo. ORCD denies listing of Nazar's entire scratch home, so additional private or undisclosed output locations cannot be discovered from Harry's account. Ask Nazar to provide any new shared root and add it to the inventory if needed; do not assume that an inaccessible path is empty.

## Maintenance

Refresh the registry after significant experiment batches or whenever run locations change. Use the repository's read-only inventory generator on a host which mounts both ORCD scratch roots, then copy the generated CSVs into `results/checkpoint_registry/` and commit alongside the milestone log. Never assume an absent checkpoint exists from a naming pattern; never delete or move checkpoints just to tidy the registry.

On ORCD, use the project Python environment:

```bash
.venv/bin/python -m scripts.build_checkpoint_registry --out ~/jobs/checkpoint-registry
```

Then copy the two CSVs to the local tracked `results/checkpoint_registry/`. The generator records pool mirrors separately, excludes symlink directory aliases, and never loads model weights or the large checkpoint metadata. Its seven tests and independent review passed. If any required root is absent or inaccessible, the generator reports a warning; resolve or explicitly document it before treating the snapshot as complete.
