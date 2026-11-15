#!/bin/bash

LABEL="jan26"

TOKEN_BUDGETS=(
    125829120
    251658240
    503316480
    1006632960
)
DEPTHS=(10 12 14 16 18 20)
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN="${WANDB_RUN:-scaling_${LABEL}}"
EVAL_TOKENS=$((100 * 524288))  # ~100M tokens for final eval (default is ~10M)

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_laws_results_${LABEL}"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "token_budget,depth,model_dim,num_iterations,tokens_trained,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if a run already exists in results
run_exists() {
    local tokens=$1
    local depth=$2
    grep -q "^${tokens},${depth}," "$RESULTS_FILE" 2>/dev/null
}

# =============================================================================
# Main Loop
# =============================================================================

for tokens in "${TOKEN_BUDGETS[@]}"; do
    log "=============================================="
    log "Token budget: $tokens tokens"
    log "=============================================="

    for d in "${DEPTHS[@]}"; do

        # Skip if already completed
        if run_exists "$tokens" "$d"; then
            log "Skipping d=$d at $tokens tokens (already in results)"
            continue
        fi

        log "Training d=$d at $tokens tokens..."

        # Unique tag for this run
        TAG="scaling_${tokens}_d${d}"

        # Reduce --device-batch-size to avoid OOM at larger depths
        if [ $d -ge 28 ]; then
            DEVICE_BATCH_SIZE_ARG="--device-batch-size=8"
        elif [ $d -ge 20 ]; then
            DEVICE_BATCH_SIZE_ARG="--device-batch-size=16"
        else
            DEVICE_BATCH_SIZE_ARG="--device-batch-size=32"
        fi

        # Record start time
        START_TIME=$(date +%s)

        # Train the model with a fixed token and batch budget.
        # CORE eval happens once at the end (999999 ensures only final step)
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
            --depth=$d \
            --train-tokens=$tokens \
            --total-batch-size=$TOTAL_BATCH_SIZE \
            --run="${WANDB_RUN}_${TAG}" \
            --eval-tokens=$EVAL_TOKENS \
            --core-metric-every=999999 \
            --core-metric-max-per-task=-1 \
            --sample-every=-1 \
            --save-every=-1 \
            $DEVICE_BATCH_SIZE_ARG \
            2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"

        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

        # Extract training stats from the log
        LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

        NUM_ITERS=$(grep "Number of iterations:" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
        TOKENS_TRAINED=$tokens
        # Model dim
        MODEL_DIM=$((d * 64))
        # Val BPB from final eval
        VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')

        # Extract CORE score from training log (evaluated on final step)
        CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        if [ -z "$CORE_SCORE" ]; then
            log "WARNING: Could not extract CORE score for d=$d"
            CORE_SCORE="0.0"
        fi

        log "  Iters: $NUM_ITERS, Tokens: $TOKENS_TRAINED, Val BPB: $VAL_BPB, CORE: $CORE_SCORE"

        # Append to CSV
        echo "$tokens,$d,$MODEL_DIM,$NUM_ITERS,$TOKENS_TRAINED,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"
    done
done

log "=============================================="
log "Scaling Laws Sweep Complete"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
