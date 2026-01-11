#!/bin/bash

# Check if required arguments are provided
if [ $# -ne 5 ]; then
    echo "Usage: $0 <dataset_path> <model> <output_dir> <seed> <max_length>"
    exit 1
fi

# Assign arguments to variables
DATASET_PATH="$1"
MODEL="$2"
OUTPUT_DIR="$3"
SEED="$4"
MAX_LENGTH="$5"

LOGFILE="./train/log/training_log_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="training_${OUTPUT_DIR//\//_}.pid"

# Log basic information
echo "Starting training: $(date)"
echo "CUDA_VISIBLE_DEVICES=0"
echo "Dataset: $DATASET_PATH"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOGFILE"
echo "==============================="

# Check if swift command is available
if ! command -v swift &> /dev/null
then
    echo "ERROR: swift command not found. Please make sure it is installed and in PATH."
    exit 1
fi

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset file not found: $DATASET_PATH"
    exit 1
fi

# Start training in background but save PID
CUDA_VISIBLE_DEVICES=0 swift sft \
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
    --num_train_epochs 6\
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --seed "$SEED" \
    --model_name "$(basename "$MODEL")-finetune" \
> "$LOGFILE" 2>&1 &

# Save the PID
TRAIN_PID=$!
echo $TRAIN_PID > "$PIDFILE"

echo "Training started in background with PID: $TRAIN_PID"
echo "Log file: $LOGFILE"
echo "Waiting for training to complete..."

# Wait for the process to finish
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

# Clean up PID file
rm -f "$PIDFILE"

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Check log file: $LOGFILE"
    exit $TRAIN_EXIT_CODE
fi

echo "Training completed successfully: $(date)"
exit 0