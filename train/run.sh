#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/scripts/common.sh"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/data}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$SCRIPT_DIR/model}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/log}"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-$SCRIPT_DIR/export_model.sh}"
DATASET_LIST_CSV="${DATASETS:-$DATASET_DIR/ecn_self_distill_qwen.jsonl}"
MAX_LENGTH_LIST_CSV="${MAX_LENGTHS:-8192}"
SEED_LIST_CSV="${SEEDS:-1}"
MANIFEST_FILE="${MANIFEST_FILE:-}"

ensure_dir "$LOG_DIR"
ensure_dir "$OUTPUT_BASE_DIR"

PIPELINE_LOG="$LOG_DIR/pipeline_log_$(date +%Y_%m%d_%H%M%S).log"

log() {
    local message="$1"
    shift || true
    echo "$message" | tee -a "$PIPELINE_LOG" "$@"
}

latest_checkpoint() {
    local output_dir="$1"
    local checkpoint=""
    local latest_step=-1

    while IFS= read -r candidate; do
        local name step
        name="$(basename "$candidate")"
        step="${name#checkpoint-}"

        if [[ "$step" =~ ^[0-9]+$ ]] && (( step > latest_step )); then
            latest_step="$step"
            checkpoint="$candidate"
        fi
    done < <(find "$output_dir" -type d -name "checkpoint-*" 2>/dev/null)

    printf '%s\n' "$checkpoint"
}

run_task() {
    local seed="$1"
    local max_length="$2"
    local dataset="$3"
    local task_name="$4"

    local timestamp output_dir task_log checkpoint
    local merged_dir
    timestamp="$(date +%Y_%m%d_%H%M%S)"
    output_dir="$OUTPUT_BASE_DIR/${task_name}_$timestamp"
    task_log="$LOG_DIR/${task_name}_$timestamp.log"

    require_file "$dataset"
    ensure_dir "$output_dir"

    log "Starting $task_name (seed $seed) at $(date)" "$task_log"

    if ! bash "$SCRIPT_DIR/train.sh" "$dataset" "$MODEL" "$output_dir" "$seed" "$max_length" \
        >>"$task_log" 2>&1; then
        log "ERROR: Training failed for $task_name" "$task_log"
        exit 1
    fi

    log "Training finished for $task_name at $(date)" "$task_log"

    checkpoint="$(latest_checkpoint "$output_dir")"
    if [[ -z "$checkpoint" ]]; then
        log "ERROR: No checkpoint found for $task_name" "$task_log"
        exit 1
    fi

    if [[ -f "$EXPORT_SCRIPT" ]]; then
        merged_dir="${checkpoint}-merged"
        if ! bash "$EXPORT_SCRIPT" "$checkpoint" "$merged_dir" >>"$task_log" 2>&1; then
            log "ERROR: Export failed for $task_name" "$task_log"
            exit 1
        fi
        if [[ -n "$MANIFEST_FILE" ]]; then
            ensure_dir "$(dirname "$MANIFEST_FILE")"
            printf '%s\t%s\n' "$merged_dir" "$task_name" >>"$MANIFEST_FILE"
        fi
    else
        log "WARNING: Export script not found, skipping export: $EXPORT_SCRIPT" "$task_log"
    fi

    log "$task_name finished at $(date)" "$task_log"
}

IFS=',' read -r -a DATASET_LIST <<<"$DATASET_LIST_CSV"
IFS=',' read -r -a MAX_LENGTH_LIST <<<"$MAX_LENGTH_LIST_CSV"
IFS=',' read -r -a SEED_LIST <<<"$SEED_LIST_CSV"

if [[ "${#DATASET_LIST[@]}" -ne "${#MAX_LENGTH_LIST[@]}" ]]; then
    die "DATASETS and MAX_LENGTHS must have the same number of comma-separated items."
fi

MODEL_LABEL="${MODEL_LABEL:-$(safe_name "$(basename "$MODEL")")}"

log "Starting pipeline: $(date)"

for i in "${!DATASET_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        dataset="${DATASET_LIST[$i]}"
        max_length="${MAX_LENGTH_LIST[$i]}"
        dataset_name="$(basename "$dataset" .jsonl)"
        task_name="${dataset_name}_${max_length}_seed${seed}_${MODEL_LABEL}"

        run_task "$seed" "$max_length" "$dataset" "$task_name"
    done
done

log "All tasks completed: $(date)"
