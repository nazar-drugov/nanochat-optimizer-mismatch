# Storage paths on the cluster

This map describes the code at commit `0ed4cd4` when launched from
`/home/ndrugov/nanochat-optimizer-mismatch` as user `ndrugov`. Paths below are
**defaults or patterns**, not a claim that the files have already been created.
`/orcd/home/002/ndrugov/nanochat-optimizer-mismatch` is another spelling of
the same checkout on this cluster.

## The two roots

| Purpose | Full path | How it is chosen |
| --- | --- | --- |
| Source code, scripts, documentation, `pyproject.toml`, `uv.lock` | `/home/ndrugov/nanochat-optimizer-mismatch/` | Git checkout; `pwd` in the repo |
| Project Python environment | `/home/ndrugov/nanochat-optimizer-mismatch/.venv/` | `uv sync` in the repo |
| Persistent nanochat data and results | `/home/ndrugov/.cache/nanochat/` | `NANOCHAT_BASE_DIR` if set; otherwise `$HOME/.cache/nanochat` |

Set `NANOCHAT_BASE_DIR` to an **absolute path** before downloading data or
launching jobs if another filesystem is preferred. The data and checkpoint
paths below then use that directory as their root. The training code does not
put data or checkpoints inside the Git checkout by default.

The home cache path is inherited from upstream nanochat; it was not selected
for this cluster study. ORCD's [filesystem guidance](https://orcd-docs.mit.edu/filesystems-file-transfer/filesystems/)
sets a 200 GB personal home quota and 1 TB each for personal pool and scratch.
`df -h /home/ndrugov` shows capacity of the **shared home filesystem**, not
the personal quota. Check your current usage with `cat /home/ndrugov/orcd/.quota`;
that report is refreshed periodically.

### Recommended placement for this study

| Purpose | Proposed full path | Reason |
| --- | --- | --- |
| Code, `.venv`, and small irreplaceable results | `/home/ndrugov/nanochat-optimizer-mismatch/` | Home is backed up. |
| Canonical downloaded dataset | `/home/ndrugov/orcd/pool/nanochat-optimizer-mismatch/base_data_fineweb_edu_100b/` | Pool holds large datasets between jobs. |
| Dataset copy read by active training | `/home/ndrugov/orcd/scratch/public/nanochat-optimizer-mismatch/base_data_fineweb_edu_100b/` | Scratch is intended for active job I/O. |
| Active checkpoints, D0, W&B files, and logs | `/home/ndrugov/orcd/scratch/public/nanochat-optimizer-mismatch/` | Keep large, frequently written job outputs on scratch. |
| GPT-2 tokenizer cache | `/home/ndrugov/.cache/nanochat/tiktoken_cache/` | Already populated; this small upstream cache is shared independently of the study's pool and scratch roots. |

Pool and scratch are **not backed up**. Copy the final checkpoints, selected
run records, and analysis outputs that must survive a storage loss to home or
another backed-up location. Decide what to retain before a large sweep; home
cannot hold an unlimited number of checkpoints.

The code has one `NANOCHAT_BASE_DIR` for several outputs. The study launcher
`runs/pretrain_study.sh` applies this layout without changing the Python
defaults: `download` writes to pool, `stage` copies shards to scratch,
`heldout` writes D0 to scratch, and `train` reads scratch data and writes
checkpoints, W&B files, and logs there. Download and stage are **one-time
dataset preparation**, repeated only when the dataset changes or the scratch
copy needs rebuilding. `train` never invokes either command.

## Pretraining inputs

| Item | Default full path | Notes |
| --- | --- | --- |
| FineWeb-Edu-100B training shards | `/home/ndrugov/.cache/nanochat/base_data_fineweb_edu_100b/shard_00000.parquet`, `shard_00001.parquet`, ... | `python -m nanochat.dataset --dataset fineweb-edu-100b -n N` downloads the first `N` training shards. |
| FineWeb-Edu-100B validation shard | `/home/ndrugov/.cache/nanochat/base_data_fineweb_edu_100b/shard_01822.parquet` | Downloaded even when `-n` is small. Direct `base_train.py` runs without `--val-data-dir` evaluate this last shard. |
| Alternative ClimbMix shards | `/home/ndrugov/.cache/nanochat/base_data_climbmix/shard_*.parquet` | Only when `--dataset climbmix` is selected; its last shard is `shard_06542.parquet`. |
| GPT-2 tokenizer cache | `/home/ndrugov/.cache/nanochat/tiktoken_cache/` | Actual cached filenames are chosen by tiktoken. `TIKTOKEN_CACHE_DIR` overrides this directory. |
| Nanochat-trained tokenizer | `/home/ndrugov/.cache/nanochat/tokenizer/tokenizer.pkl` and `token_bytes.pt` | Used with `--tokenizer nanochat`; current pretraining protocol chooses `gpt2`. |
| Deduplicated held-out D0 | `/home/ndrugov/orcd/scratch/public/nanochat-optimizer-mismatch/heldout/fineweb-edu-100b/d0.parquet` | Created by `scripts.make_heldout`; the study launcher passes its directory as `--val-data-dir`. |
| D0 construction report | `/home/ndrugov/.cache/nanochat/heldout/fineweb-edu-100b/d0_report.json` | Includes the scanned training shards and deduplication results. |

`--data-dir /absolute/path` makes pretraining read that directory instead of
the registered dataset directory. The last alphabetically sorted `.parquet`
file in that directory is validation; all earlier files are training.
`scripts.make_heldout` has its own `--data-dir` and `--output-dir` options.
Changing one command's directory does not automatically change the other.
`--val-data-dir /absolute/path` makes training validate on the sole Parquet
file in that directory. The study launcher sets it to D0 on scratch, verifies
the D0 manifest, and defaults to 1,048,576 evaluation tokens per evaluation.
Before every `train` start and resume, the launcher also hashes every training
shard against `data_manifests/fineweb_edu_100_v1/shards.sha256` (about 30
seconds) and stops if any shard is missing, extra, or different.

## Pretraining checkpoints and run records

For a run named `moe-e8-muon-s42`, the directory is
`/home/ndrugov/.cache/nanochat/base_checkpoints/moe-e8-muon-s42/`.
`--run moe-e8-muon-s42` sets that name. Without `--run`, the code generates
a longer name from the architecture, optimizer, seed, token budget, and other
settings. `--save-every=-1` still saves the final checkpoint.

| Item | Full path for the example run | Notes |
| --- | --- | --- |
| Resolved run settings | `/home/ndrugov/.cache/nanochat/base_checkpoints/moe-e8-muon-s42/config.json` | Includes Git state, hardware, model, optimizer, data, and tokenizer details. |
| Resume settings | `/home/ndrugov/.cache/nanochat/base_checkpoints/moe-e8-muon-s42/config_resume_001000.json` | Written on a resume from step 1000. |
| Model weights | `/home/ndrugov/.cache/nanochat/base_checkpoints/moe-e8-muon-s42/model_001000.pt` | One model file per saved step. |
| Training metadata | `/home/ndrugov/.cache/nanochat/base_checkpoints/moe-e8-muon-s42/meta_001000.json` | Step, configuration, data cursor and fingerprints, training shard names and sizes, tokenizer file fingerprints, validation metric, and more. A resume compares the shard list and tokenizer with the files it finds. |
| Optimizer state | `/home/ndrugov/.cache/nanochat/base_checkpoints/moe-e8-muon-s42/optim_001000_rank0.pt` | One file **per rank**; a two-GPU run also writes `optim_001000_rank1.pt`. |
| MoE routing probe | `/home/ndrugov/.cache/nanochat/base_checkpoints/moe-e8-muon-s42/routing_001000.pt` | Only for MoE checkpoints. |

The number in each filename is the **optimizer step**, zero-padded to six
digits. Checkpoints and run records go under `NANOCHAT_BASE_DIR`; there is
no separate checkpoint-directory flag in `base_train.py`.

## Logs and metrics

| Item | Full path or behavior | How to control it |
| --- | --- | --- |
| Console output and errors | Terminal by default; **no file is created automatically** | Redirect the launch command to an absolute `.log` path or set the scheduler's output path. |
| Per-step JSONL metrics | No file by default | Pass `--train-stats-jsonl=/absolute/path/run.jsonl`; rank 0 appends records and creates parent directories. |
| Weights & Biases local files | `/home/ndrugov/nanochat-optimizer-mismatch/wandb/` when launched from the repo with the installed W&B defaults | Set `WANDB_DIR` to an absolute parent directory to move W&B's local `wandb/` folder. Pretraining calls `wandb.init(project="nanochat")` regardless of `--run`. |
| Weights & Biases remote metrics | Project `nanochat`, if W&B is configured for online use | The run name matches the checkpoint directory name. |
| Existing pytest capture | `/home/ndrugov/nanochat-optimizer-mismatch/pytest.log` | Created by the shell redirection used earlier; unrelated to training. |
| Existing verification logs | `/home/ndrugov/nanochat-optimizer-mismatch/verification_runs/` | Files from earlier checks; not an automatic output location for new jobs. |

Use the dedicated launcher from the repository root:

```bash
cd /home/ndrugov/nanochat-optimizer-mismatch
bash runs/pretrain_study.sh paths

# One-time data preparation, not repeated for each run:
bash runs/pretrain_study.sh download 100  # on a node with internet access
bash runs/pretrain_study.sh stage
# On an allocated CPU job, build D0 if needed:
bash runs/pretrain_study.sh heldout --workers 32

# For each pretraining cell, preview the command without starting a GPU job:
RUN_NAME=moe-e8-muon-s42 bash runs/pretrain_study.sh dry-run \
  --depth=10 --ffn-type=moe --n-experts=8 --optimizer-name=muon \
  --train-tokens=3000238080 --save-every=1000
```

For an actual GPU job, use `train` in place of `dry-run` after choosing the
cell's learning rate and the rest of the protocol. The launcher uses the
existing `.venv`; it does not install packages or allocate a GPU. Its
`RUN_NAME` sets the checkpoint directory and both log filenames.

With that example run name, the console log is
`/home/ndrugov/orcd/scratch/public/nanochat-optimizer-mismatch/logs/moe-e8-muon-s42.log`,
the JSONL metrics are
`/home/ndrugov/orcd/scratch/public/nanochat-optimizer-mismatch/logs/moe-e8-muon-s42.jsonl`,
and W&B's local files are under
`/home/ndrugov/orcd/scratch/public/nanochat-optimizer-mismatch/wandb/`.

## Other stages and caches

| Item | Default full path |
| --- | --- |
| SFT checkpoints | `/home/ndrugov/.cache/nanochat/chatsft_checkpoints/<SOURCE_TAG>-step<STEP>-sft-<OPTIMIZER>/` |
| RL checkpoints | `/home/ndrugov/.cache/nanochat/chatrl_checkpoints/<MODEL_TAG>/` |
| Downloaded task datasets | `/home/ndrugov/.cache/nanochat/task_data/<REPO_SLUG>/<SUBSET>/<SPLIT>/` |
| CORE evaluation bundle | `/home/ndrugov/.cache/nanochat/eval_bundle.zip` and `eval_bundle/` |
| Base CORE evaluation CSV | `/home/ndrugov/.cache/nanochat/base_eval/base_model_<STEP>.csv` |

These later stages are not yet the launch path for the optimizer-mismatch
pretraining study. In particular, the current protocol says GPT-2 checkpoint
support in `chat_sft.py` and `base_eval.py` still needs work.

## Check what actually exists

```bash
cd /home/ndrugov/nanochat-optimizer-mismatch
printf 'Repo: %s\n' "$PWD"
printf 'Base: %s\n' "${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
find "${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}" -maxdepth 4 -type f 2>/dev/null | sort | head -100
```

If the final command prints nothing, the default base directory may not have
been populated yet. Check `printenv NANOCHAT_BASE_DIR` and the launch script
for an override before concluding the data is missing.
