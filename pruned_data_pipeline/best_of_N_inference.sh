#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/scripts/common.sh"

MODEL_ID=""
DATASET_FILE=""
OUTPUT_DIR="./best_of_n_results"
SEEDS=(1 2 3 4)

BACKEND="vllm"
TEMPERATURE=0.7
REPETITION_PENALTY=1
TOP_P=1
MAX_NEW_TOKENS=8192
TIMEOUT_HOURS=4

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

usage() {
    local exit_code="${1:-1}"

    echo "Usage: $0 -m <model_id_or_path> -d <dataset_file> [-o <output_dir>] [-s <seed1,seed2,...>] [-h]"
    echo "  -m  Model ID or local path. Required."
    echo "  -d  Dataset JSONL file. Required."
    echo "  -o  Output directory. Default: ./best_of_n_results"
    echo "  -s  Comma-separated seeds. Default: 1,2,3,4"
    echo "  -h  Show this help message."
    exit "$exit_code"
}

while getopts "m:d:o:s:h" opt; do
    case "$opt" in
        m) MODEL_ID="$OPTARG" ;;
        d) DATASET_FILE="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        s) IFS=',' read -r -a SEEDS <<<"$OPTARG" ;;
        h) usage 0 ;;
        *) usage 1 ;;
    esac
done

if [[ -z "$MODEL_ID" || -z "$DATASET_FILE" ]]; then
    echo "ERROR: -m and -d are required."
    usage 1
fi

require_file "$DATASET_FILE"
SWIFT_BIN="$(require_modelscope_swift)"

ensure_dir "$OUTPUT_DIR"
dataset_name="$(basename "$DATASET_FILE" .jsonl)"
round_count="${#SEEDS[@]}"

echo "=========================================="
echo "Starting Best-of-$round_count inference"
echo "Model: $MODEL_ID"
echo "Dataset: $DATASET_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Seeds: ${SEEDS[*]}"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

for current_seed in "${SEEDS[@]}"; do
    result_file="$OUTPUT_DIR/${dataset_name}_seed${current_seed}.jsonl"
    log_file="$OUTPUT_DIR/${dataset_name}_seed${current_seed}.log"

    echo
    echo ">>> Running seed $current_seed"
    echo ">>> Saving results to: $result_file"

    if [[ -f "$result_file" ]]; then
        echo "Result file already exists, skipping seed $current_seed."
        continue
    fi

    run_with_timeout_hours "$TIMEOUT_HOURS" "$SWIFT_BIN" infer \
        --model "$MODEL_ID" \
        --val_dataset "$DATASET_FILE" \
        --infer_backend "$BACKEND" \
        --temperature "$TEMPERATURE" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --top_p "$TOP_P" \
        --seed "$current_seed" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --result_path "$result_file" \
        >"$log_file" 2>&1

    exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo "Seed $current_seed completed successfully."
    elif [[ $exit_code -eq 124 ]]; then
        echo "Seed $current_seed exceeded $TIMEOUT_HOURS hours."
    else
        echo "Seed $current_seed failed (exit code: $exit_code). See log: $log_file"
        echo "--------- Last 15 lines of log ---------"
        tail -n 15 "$log_file"
    fi
done

echo
echo "=========================================="
echo "All $round_count rounds completed."
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
