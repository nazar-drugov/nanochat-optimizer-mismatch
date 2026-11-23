#!/usr/bin/env bash
: <<'GUIDE'
PRETRAINING STUDY: START HERE

Run this script from the repository directory. Data preparation (`download`,
`stage`, `heldout`) is separate and normally done once per dataset version.
A training run reads the existing FineWeb-Edu files on scratch; `train` never
downloads them again.

Before the learning-rate sweep, check D0 on an allocated two-GPU node with a
one-step run. Use a fresh name (or change the suffix if you repeat the test):

  RUN_NAME=d0-check-s42 bash runs/pretrain_study.sh train \
      --depth=10 --ffn-type=moe --n-experts=8 \
      --optimizer-name=muon --muon-matrix-lr=0.004 \
      --train-tokens=262144 --warmup-steps=0 --eval-every=1 \
      --eval-tokens=131072 --core-metric-every=-1 --sample-every=-1 --seed=42

This one-step test evaluates D0 at steps 0 and 1, writes a checkpoint, and
logs to <scratch>/logs/d0-check-s42.log. Check for two finite "Validation bpb"
lines, then inspect <scratch>/base_checkpoints/d0-check-s42/config.json for
data.val_data_file ending in /d0.parquet and its val_data_sha256. This test
does not select a learning rate.

To start a run, first get an allocated GPU node. In that node's shell:

  cd ~/nanochat-optimizer-mismatch
  RUN_NAME=moe-e8-muon-s42 bash runs/pretrain_study.sh dry-run \
      --depth=10 --ffn-type=moe --n-experts=8 \
      --optimizer-name=muon --muon-matrix-lr=0.004 \
      --train-tokens=3000238080 --eval-tokens=1048576 \
      --core-metric-every=-1 --sample-every=-1 --save-every=1000 --seed=42

`dry-run` prints the command and paths without starting training. Review it,
then run the same command with `train` instead of `dry-run`. Give each new run
a unique RUN_NAME. To use AdamW, replace the Muon options with
`--optimizer-name=adamw --adamw-matrix-lr=<chosen_rate>`. Change the expert
count or token budget with their flags. This example uses a provisional learning
rate. The launcher evaluates D0 from scratch and records its SHA-256 in every
run. Finish the short GPU validation test before the learning-rate sweep.

WHAT THE EXAMPLE RUN USES (including defaults from scripts/base_train.py):

  Data/model:     FineWeb-Edu training shards on scratch (currently 100), GPT-2
                  tokenizer, depth 10 (width 640), 8-expert top-1 MoE in every block;
                  sequence length 2048. Two GPU processes by default.
  Optimizer:      plain Muon for matrices and routers, matrix/router peak LR
                  0.004; AdamW for embeddings (LR 0.3) and output head (LR 0.008).
                  Auxiliary balance coefficient 0.01, router z-loss coefficient 0.001,
                  matrix weight decay 0.1, gradient clipping norm 1.0.
  Training:       3,000,238,080 tokens = 11,445 updates at 262,144 tokens per
                  update. Per-GPU microbatch is 32 sequences; on two GPUs,
                  two microbatches are accumulated per update. Linear LR warmup
                  over 130M tokens, then constant, then linear decay over the
                  final 20% of updates. Seed 42.
  Validation:     Bits per byte every 250 updates on D0 (`d0.parquet`), using
                  1,048,576 model input tokens per evaluation. The same fixed
                  subset is revisited each time. CORE benchmark and text
                  sampling are disabled in this example.
  Outputs:        Checkpoints and resolved config.json:
                    <scratch>/base_checkpoints/<RUN_NAME>/
                  Console log and metrics JSONL:
                  <scratch>/logs/<RUN_NAME>.log and <RUN_NAME>.jsonl
                  W&B files under <scratch>.
                  `<scratch>` is printed by `bash runs/pretrain_study.sh paths`.

Only `--depth`, `--ffn-type`, `--n-experts`, `--optimizer-name`,
`--muon-matrix-lr`, `--train-tokens`, `--eval-tokens`,
`--core-metric-every`, `--sample-every`, `--save-every`, and `--seed` are
explicit in the example. The other values above are current Python defaults,
not settings imposed by this launcher. `dry-run` prints the launch command but
does not expand those defaults or verify the data. Before every `train` start
or resume, the launcher checks that the shard files match the frozen inventory
in `data_manifests/` and that every shard's SHA-256 matches (about 30 seconds),
and it stops if anything differs. After a run starts, inspect its config.json for
the exact resolved values. See `PRETRAINING_PROTOCOL.md` for study decisions and
`.venv/bin/python -m scripts.base_train --help` for all available flags.
GUIDE

# Launcher for this study. The first argument selects ONE action below:
#   paths    Print locations only.
#   download Fetch FineWeb-Edu shards from Hugging Face into pool.
#   stage    Copy already-downloaded shards from pool to scratch.
#   heldout  Build D0 from the shards on scratch (CPU-intensive, run once).
#   dry-run  Print a training command without starting it.
#   train    Start scripts.base_train using the shards already on scratch.
# Running "train" never calls "download" or "stage" and never fetches the
# FineWeb-Edu shards. Run this file from any directory with the repo's .venv.

# Exit on a failed command (-e), reject unset variables (-u), and report
# failures inside pipelines such as `torchrun | tee` (pipefail).
set -euo pipefail

# Resolve this script's location so relative commands work even when the caller
# is in a different directory. These are the default study paths; the two
# NANOCHAT_*_BASE environment variables allow explicit overrides.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POOL_BASE="${NANOCHAT_POOL_BASE:-$HOME/orcd/pool/nanochat-optimizer-mismatch}"
SCRATCH_BASE="${NANOCHAT_SCRATCH_BASE:-$HOME/orcd/scratch/public/nanochat-optimizer-mismatch}"
# The dataset has the same filename on pool and scratch. Pool is the source
# copy; only the scratch copy is passed to training and D0 construction.
DATA_DIR="$SCRATCH_BASE/base_data_fineweb_edu_100b"
POOL_DATA_DIR="$POOL_BASE/base_data_fineweb_edu_100b"
HELDOUT_DIR="$SCRATCH_BASE/heldout/fineweb-edu-100b"
# Frozen file inventory and SHA-256 fingerprints of the shards and D0. Only
# automated tests override this.
MANIFEST_DIR="${NANOCHAT_DATA_MANIFEST_DIR:-$REPO_DIR/data_manifests/fineweb_edu_100_v1}"
# Use executables from this checkout's virtual environment, not a system Python.
PYTHON="$REPO_DIR/.venv/bin/python"
TORCHRUN="$REPO_DIR/.venv/bin/torchrun"
# Two GPU processes by default, one per GPU. Override for another allocation.
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

usage() {
    cat <<'EOF'
Usage:
  bash runs/pretrain_study.sh paths
  bash runs/pretrain_study.sh download [TRAIN_SHARDS]
  bash runs/pretrain_study.sh stage
  bash runs/pretrain_study.sh heldout [make_heldout arguments...]
  RUN_NAME=name bash runs/pretrain_study.sh dry-run [base_train arguments...]
  RUN_NAME=name bash runs/pretrain_study.sh train [base_train arguments...]

Default download: 100 training shards plus the validation shard.
Download and stage are dataset preparation steps, not part of train.
Repeat them only when adding/changing shards or rebuilding the scratch copy.
Train inside an allocated GPU job. Pass --train-tokens and the chosen cell settings.
NANOCHAT_POOL_BASE and NANOCHAT_SCRATCH_BASE may override the two storage roots.
EOF
}

need_python() {
    # A missing .venv would otherwise produce a less useful shell error.
    if [[ ! -x "$PYTHON" ]]; then
        echo "Missing $PYTHON; prepare the repository virtual environment first." >&2
        exit 1
    fi
}

# The first word is the action. `shift` leaves its remaining arguments in "$@"
# so they can be passed to the selected Python program.
mode="${1:-}"
if [[ -z "$mode" ]]; then
    usage
    exit 2
fi
shift

case "$mode" in
    paths)
        # Display the resolved paths; this mode reads or writes no dataset files.
        printf 'Repository: %s\n' "$REPO_DIR"
        printf 'Canonical dataset: %s\n' "$POOL_DATA_DIR"
        printf 'Active dataset: %s\n' "$DATA_DIR"
        printf 'Active outputs: %s\n' "$SCRATCH_BASE"
        printf 'Held-out D0: %s/d0.parquet\n' "$HELDOUT_DIR"
        printf 'Tokenizer cache: %s\n' "${TIKTOKEN_CACHE_DIR:-$HOME/.cache/nanochat/tiktoken_cache}"
        ;;
    download)
        # This is the ONLY mode that asks nanochat.dataset to fetch FineWeb-Edu.
        # `-n 100` means shards 00000..00099 plus shard 01822 (validation).
        # Existing filenames are skipped by nanochat.dataset; this mode does
        # not copy files to scratch, build D0, or launch training.
        need_python
        if [[ $# -gt 1 || ! "${1:-100}" =~ ^[1-9][0-9]*$ ]]; then
            echo "Usage: bash runs/pretrain_study.sh download [TRAIN_SHARDS]" >&2
            exit 2
        fi
        # nanochat.dataset writes under NANOCHAT_BASE_DIR. Set that variable
        # for this one command so the dataset lands in pool rather than home.
        mkdir -p "$POOL_BASE"
        cd "$REPO_DIR"
        NANOCHAT_BASE_DIR="$POOL_BASE" "$PYTHON" -m nanochat.dataset \
            --dataset fineweb-edu-100b -n "${1:-100}"
        ;;
    stage)
        # Copy the pool dataset to scratch for active jobs. This reads local
        # cluster storage; it makes no request to Hugging Face.
        if [[ $# -ne 0 ]]; then
            usage >&2
            exit 2
        fi
        if [[ ! -f "$POOL_DATA_DIR/shard_01822.parquet" || ! -f "$POOL_DATA_DIR/shard_00000.parquet" ]]; then
            echo "Download the FineWeb-Edu training and validation shards to $POOL_DATA_DIR first." >&2
            exit 1
        fi
        mkdir -p "$DATA_DIR"
        # Extra Parquet files could silently become training shards because the
        # data loader treats every file except the last as training data.
        for staged_shard in "$DATA_DIR"/*.parquet; do
            [[ -e "$staged_shard" ]] || continue
            if [[ ! -f "$POOL_DATA_DIR/$(basename "$staged_shard")" ]]; then
                echo "Unexpected shard $staged_shard; remove or move it before staging." >&2
                exit 1
            fi
        done
        # -a preserves file metadata; --checksum compares file contents when
        # deciding whether an existing scratch file needs to be recopied.
        # There is no progress display, so this command may be quiet for a while.
        rsync -a --checksum "$POOL_DATA_DIR/" "$DATA_DIR/"
        printf 'Staged dataset at %s\n' "$DATA_DIR"
        ;;
    heldout)
        # Scan every staged training shard for near-duplicates of candidate
        # documents from the validation shard. The Python program writes D0
        # and d0_report.json under HELDOUT_DIR. Re-running it rebuilds D0.
        # It does not train a model or download the dataset.
        need_python
        if [[ ! -f "$DATA_DIR/shard_01822.parquet" || ! -f "$DATA_DIR/shard_00000.parquet" ]]; then
            echo "Stage the dataset to $DATA_DIR before building D0." >&2
            exit 1
        fi
        # Keep D0 output on scratch and reuse the small GPT-2 tokenizer cache
        # in home. If that cache is absent, tiktoken may need network access.
        export NANOCHAT_BASE_DIR="$SCRATCH_BASE"
        export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HOME/.cache/nanochat/tiktoken_cache}"
        mkdir -p "$HELDOUT_DIR" "$TIKTOKEN_CACHE_DIR"
        cd "$REPO_DIR"
        "$PYTHON" -m scripts.make_heldout \
            --dataset fineweb-edu-100b --data-dir "$DATA_DIR" \
            --output-dir "$HELDOUT_DIR" --tokenizer gpt2 \
            --workers "${HELDOUT_WORKERS:-4}" "$@"
        ;;
    dry-run|train)
        # Both modes assemble the same base_train command. dry-run prints it;
        # train runs it inside an already-allocated GPU job.
        need_python
        if [[ ! -x "$TORCHRUN" ]]; then
            echo "Missing $TORCHRUN; prepare the repository virtual environment first." >&2
            exit 1
        fi
        # RUN_NAME selects the checkpoint directory and the log filenames.
        RUN_NAME="${RUN_NAME:-}"
        if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9._-]+$ || "$RUN_NAME" == "." || "$RUN_NAME" == ".." ]]; then
            echo "Set RUN_NAME to a run-specific name using letters, digits, dot, dash, or underscore." >&2
            exit 2
        fi
        if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
            echo "NPROC_PER_NODE must be a positive integer." >&2
            exit 2
        fi
        # Do not let callers override paths and names supplied by this
        # launcher. Require an explicit token budget for every training run.
        has_train_tokens=false
        has_eval_tokens=false
        is_resume=false
        for arg in "$@"; do
            case "$arg" in
                --run|--run=*|--data-dir|--data-dir=*|--val-data-dir|--val-data-dir=*|--train-stats-jsonl|--train-stats-jsonl=*)
                    echo "$arg is set by this launcher; remove it from the base_train arguments." >&2
                    exit 2
                    ;;
                --train-tokens|--train-tokens=*) has_train_tokens=true ;;
                --eval-tokens|--eval-tokens=*) has_eval_tokens=true ;;
                --resume-from-step|--resume-from-step=*) is_resume=true ;;
            esac
        done
        if [[ "$has_train_tokens" != true ]]; then
            echo "Pass an explicit --train-tokens value from the pretraining protocol." >&2
            exit 2
        fi
        # D0 has about 2M raw tokens. Use 1M model input tokens by default;
        # the data loader may discard some raw tokens when packing batches.
        # A caller can override this count for a short test.
        eval_token_args=(--eval-tokens=1048576)
        if [[ "$has_eval_tokens" == true ]]; then
            eval_token_args=()
        fi
        # base_train uses NANOCHAT_BASE_DIR for checkpoints and other outputs.
        # Its --data-dir below points specifically at the staged scratch shards.
        # W&B files, tokenizer cache, and per-run logs get explicit locations.
        export NANOCHAT_BASE_DIR="$SCRATCH_BASE"
        export WANDB_DIR="${WANDB_DIR:-$SCRATCH_BASE}"
        export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HOME/.cache/nanochat/tiktoken_cache}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
        log_file="$SCRATCH_BASE/logs/$RUN_NAME.log"
        stats_file="$SCRATCH_BASE/logs/$RUN_NAME.jsonl"
        # torchrun starts one copy of scripts.base_train per GPU. Arguments in
        # "$@" come from the caller; the final path arguments are fixed here.
        cmd=("$TORCHRUN" --standalone "--nproc_per_node=$NPROC_PER_NODE" -m scripts.base_train
             "${eval_token_args[@]}" "$@" "--run=$RUN_NAME" "--data-dir=$DATA_DIR"
             "--val-data-dir=$HELDOUT_DIR" "--train-stats-jsonl=$stats_file")
        if [[ "$mode" == dry-run ]]; then
            # Preview only: do not run torchrun or check that data exists.
            printf 'NANOCHAT_BASE_DIR=%s\nWANDB_DIR=%s\nTIKTOKEN_CACHE_DIR=%s\n' \
                "$NANOCHAT_BASE_DIR" "$WANDB_DIR" "$TIKTOKEN_CACHE_DIR"
            printf 'Console log: %s\nMetrics JSONL: %s\nCommand: ' "$log_file" "$stats_file"
            printf '%q ' "${cmd[@]}"
            printf '\n'
            exit 0
        fi
        if [[ ! -f "$DATA_DIR/shard_01822.parquet" || ! -f "$DATA_DIR/shard_00000.parquet" ]]; then
            echo "Stage the dataset to $DATA_DIR before training." >&2
            exit 1
        fi
        # Check every training shard against the frozen manifest before every
        # start and every resume. Python rejects a resume whose shard names or
        # sizes differ from the checkpoint's, but cannot see a same-size edit
        # or a different dataset version; this byte-level check can. Hashing the
        # 100 shards (about 9.5 GB) took about 26 seconds on a CPU node.
        # First, the set of Parquet files must match the manifest exactly, since
        # the data loader treats every file except the last as training data.
        # `find -L` follows symbolic links, as the Python data loader does.
        if ! diff_output="$(diff <(awk '{print $2}' "$MANIFEST_DIR/shards.sha256" | LC_ALL=C sort) \
                                 <(find -L "$DATA_DIR" -maxdepth 1 -name '*.parquet' -printf '%f\n' | LC_ALL=C sort))"; then
            echo "The Parquet files in $DATA_DIR do not match the data manifest" >&2
            echo "(< = in the manifest but missing here, > = here but not in the manifest):" >&2
            echo "$diff_output" >&2
            exit 1
        fi
        echo "Checking training shard fingerprints against $MANIFEST_DIR/shards.sha256 ..." >&2
        (cd "$DATA_DIR" && sha256sum --quiet -c "$MANIFEST_DIR/shards.sha256" >&2) || {
            echo "A training shard differs from the frozen data manifest (see the lines above)." >&2
            exit 1
        }
        if [[ ! -f "$HELDOUT_DIR/d0.parquet" || ! -f "$HELDOUT_DIR/d0_report.json" ]]; then
            echo "Build D0 at $HELDOUT_DIR before training." >&2
            exit 1
        fi
        # The checked-in manifest freezes D0 across separate runs. Python also
        # hashes it into each checkpoint and rejects a changed file on resume.
        (cd "$HELDOUT_DIR" && sha256sum --status -c "$MANIFEST_DIR/heldout.sha256") || {
            echo "D0 or its report differs from the frozen data manifest." >&2
            exit 1
        }
        # Refuse to reuse an existing run name unless this is an explicit
        # checkpoint resume, avoiding accidental mixing of two runs' outputs.
        if [[ -f "$SCRATCH_BASE/base_checkpoints/$RUN_NAME/config.json" && "$is_resume" != true ]]; then
            echo "Run $RUN_NAME already has a config.json; choose a new RUN_NAME or pass --resume-from-step." >&2
            exit 1
        fi
        mkdir -p "$SCRATCH_BASE/logs" "$WANDB_DIR" "$TIKTOKEN_CACHE_DIR"
        cd "$REPO_DIR"
        printf '\n=== %s run %s ===\n' "$(date -Is)" "$RUN_NAME" | tee -a "$log_file"
        # Show stdout/stderr in the terminal and append it to the run log.
        # The Python program performs optimizer steps, periodic validation,
        # and checkpoint writes. Validation reads D0 via --val-data-dir.
        # W&B can use the network, and enabled CORE evaluation may fetch its
        # benchmark bundle, but neither operation downloads FineWeb-Edu again.
        "${cmd[@]}" 2>&1 | tee -a "$log_file"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
