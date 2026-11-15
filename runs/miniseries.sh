#!/bin/bash

# See speedrun.sh for more comments
# Usage: ./miniseries.sh [series_name]
# Example: ./miniseries.sh jan11
# Default series name is today's date (e.g., jan11)

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Setup (skip with SKIP_SETUP=1)
if [ -z "$SKIP_SETUP" ]; then
    # uv
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate

    # Tokenizer, download 1000 shards for pretraining
    # (probably this can be reduced but it's tricky to determine the exact right number, TODO).
    python -m nanochat.dataset -n 1000
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
else
    source .venv/bin/activate
fi

# Series name: from arg, env var, or default to today's date (e.g., jan11)
SERIES_NAME="${1:-${SERIES_NAME:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}}"
# Depths to train (the "miniseries")
DEPTHS=(12 14 16 18 20 22 24 26)
# Hardware
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_TOKENS="${TRAIN_TOKENS:-125829120}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
# Logging
WANDB_RUN="${WANDB_RUN:-${SERIES_NAME}_miniseries}"

RESULTS_DIR="$NANOCHAT_BASE_DIR/${SERIES_NAME}_miniseries_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "depth,model_dim,num_iterations,tokens_trained,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=============================================="
log "${SERIES_NAME} Miniseries Training"
log "=============================================="

for d in "${DEPTHS[@]}"; do
    log "Training d=$d..."

    TAG="${SERIES_NAME}_miniseries_d${d}"
    START_TIME=$(date +%s)

    # Reduce --device-batch-size to avoid OOM at larger depths
    if [ $d -ge 28 ]; then
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=8"
    elif [ $d -ge 20 ]; then
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=16"
    else
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=32"
    fi

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --depth=$d \
        --train-tokens=$TRAIN_TOKENS \
        --total-batch-size=$TOTAL_BATCH_SIZE \
        --run="${WANDB_RUN}_d${d}" \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        $DEVICE_BATCH_SIZE_ARG \
        2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"

    END_TIME=$(date +%s)
    TRAIN_TIME=$((END_TIME - START_TIME))

    # Extract stats from log
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"
    NUM_ITERS=$(grep "Number of iterations:" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
    TOKENS_TRAINED=$TRAIN_TOKENS
    MODEL_DIM=$((d * 64))
    VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')
    CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')

    if [ -z "$CORE_SCORE" ]; then
        CORE_SCORE="0.0"
    fi

    log "  d=$d: iterations=$NUM_ITERS, tokens=$TOKENS_TRAINED, bpb=$VAL_BPB, CORE=$CORE_SCORE, time=${TRAIN_TIME}s"

    # Append to CSV
    echo "$d,$MODEL_DIM,$NUM_ITERS,$TOKENS_TRAINED,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"
done

log "=============================================="
log "${SERIES_NAME} Miniseries Complete!"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
