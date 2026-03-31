#!/bin/bash

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" #change base model here
DATASET_DIR="./data" #change data dir here
OUTPUT_BASE_DIR="./train/model"
LOG_DIR="./train/log"
mkdir -p "$LOG_DIR"

# main log
PIPELINE_LOG="$LOG_DIR/pipeline_log_$(date +%Y_%m%d_%H%M%S).log"

echo "Starting pipeline: $(date)" | tee -a "$PIPELINE_LOG"

run_task() {
    local SEED=$1
    local MAX_LENGTH=$2
    local DATASET=$3
    local TASK_NAME=$4

    TIMESTAMP=$(date +%Y_%m%d_%H%M%S)
    OUTPUT_DIR="$OUTPUT_BASE_DIR/${TASK_NAME}_seed${SEED}_$TIMESTAMP"
    mkdir -p "$OUTPUT_DIR"

    TASK_LOG="$LOG_DIR/${TASK_NAME}_seed${SEED}_$TIMESTAMP.log"

    echo "Starting $TASK_NAME (seed $SEED) at $(date)" | tee -a "$PIPELINE_LOG" "$TASK_LOG"

    # 训练（前台执行，阻塞）
    bash ./train.sh "$DATASET" "$MODEL" "$OUTPUT_DIR" "$SEED" "$MAX_LENGTH" \
        >> "$TASK_LOG" 2>&1

    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for $TASK_NAME" | tee -a "$PIPELINE_LOG" "$TASK_LOG"
        exit 1
    fi

    echo "Training finished for $TASK_NAME at $(date)" | tee -a "$PIPELINE_LOG" "$TASK_LOG"

    # 找 checkpoint
    CHECKPOINT=$(find "$OUTPUT_DIR" -type d -name "checkpoint-*" | sort -V | tail -n 1)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found for $TASK_NAME" | tee -a "$PIPELINE_LOG" "$TASK_LOG"
        exit 1
    fi

    # 导出（前台执行，阻塞）
    bash ./export_model.sh "$CHECKPOINT" >> "$TASK_LOG" 2>&1

    if [ $? -ne 0 ]; then
        echo "ERROR: Export failed for $TASK_NAME" | tee -a "$PIPELINE_LOG" "$TASK_LOG"
        exit 1
    fi

    echo "$TASK_NAME finished at $(date)" | tee -a "$PIPELINE_LOG" "$TASK_LOG"
}

DATASETS=(
    "$DATASET_DIR/ecn_self_distill_qwen.jsonl"
)

MAX_LENGTHS=(
"8192"
)

SEEDS=("1")   # can add multiple seeds like ("1" "2" "3")



for i in ${!DATASETS[@]}; do
    for seed in ${SEEDS[@]}; do
        DATASET="${DATASETS[i]}"
        MAX_LENGTH="${MAX_LENGTHS[i]}"
        DATASET_NAME=$(basename "$DATASET" .jsonl)
        TASK_NAME="${DATASET_NAME}_${MAX_LENGTH}_${seed}_${MODEL}"
        run_task "$seed" "$MAX_LENGTH" "$DATASET" "$TASK_NAME"
    done
done

echo "All tasks completed: $(date)" | tee -a "$PIPELINE_LOG"
