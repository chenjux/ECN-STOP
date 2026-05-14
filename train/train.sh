#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <dataset_path> <model> <output_dir> <seed> <max_length>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/scripts/common.sh"

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/log}"
ensure_dir "$LOG_DIR"

DATASET_PATH="$1"
MODEL="$2"
OUTPUT_DIR="$3"
SEED="$4"
MAX_LENGTH="$5"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

LOGFILE="$LOG_DIR/training_log_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="$LOG_DIR/training_$(basename "$OUTPUT_DIR")_$$.pid"

cleanup() {
    rm -f "$PIDFILE"
}
trap cleanup EXIT

echo "Starting training: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Dataset: $DATASET_PATH"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOGFILE"
echo "==============================="

require_file "$DATASET_PATH"
SWIFT_BIN="$(require_modelscope_swift)"

ensure_dir "$OUTPUT_DIR"

"$SWIFT_BIN" sft \
    --model "$MODEL" \
    --train_type lora \
    --dataset "$DATASET_PATH" \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length "$MAX_LENGTH" \
    --num_train_epochs 6 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --seed "$SEED" \
    --model_name "$(basename "$MODEL")-finetune" \
    >"$LOGFILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" >"$PIDFILE"

echo "Training started in background with PID: $TRAIN_PID"
echo "Log file: $LOGFILE"
echo "Waiting for training to complete..."

set +e
wait "$TRAIN_PID"
TRAIN_EXIT_CODE=$?
set -e

if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Check log file: $LOGFILE"
    exit "$TRAIN_EXIT_CODE"
fi

echo "Training completed successfully: $(date)"
