#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/scripts/common.sh"

MODEL_ID=""
DATASET_FILE=""
OUTPUT_DIR="./results"

BACKEND="vllm"
TEMPERATURE=0.7
REPETITION_PENALTY=1
TOP_P=1
SEED=1
MAX_NEW_TOKENS=16384
TIMEOUT_HOURS=2

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

usage() {
    local exit_code="${1:-1}"

    echo "Usage: $0 -m <model_id_or_path> -d <dataset_file> [-o <output_dir>] [-h]"
    echo "  -m  Model ID or local path."
    echo "  -d  Dataset JSONL file."
    echo "  -o  Output directory. Default: ./results"
    echo "  -h  Show this help message."
    exit "$exit_code"
}

while getopts "m:d:o:h" opt; do
    case "$opt" in
        m) MODEL_ID="$OPTARG" ;;
        d) DATASET_FILE="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
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
result_file="$OUTPUT_DIR/${dataset_name}_result.jsonl"
log_file="$OUTPUT_DIR/${dataset_name}.log"

echo "---------------- RUNNING ----------------"
echo "Model: $MODEL_ID"
echo "Dataset: $DATASET_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Output Dir: $OUTPUT_DIR"
echo "Result: $result_file"
echo "Log: $log_file"
echo "------------------------------------------"

run_with_timeout_hours "$TIMEOUT_HOURS" "$SWIFT_BIN" infer \
    --model "$MODEL_ID" \
    --val_dataset "$DATASET_FILE" \
    --infer_backend "$BACKEND" \
    --temperature "$TEMPERATURE" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --top_p "$TOP_P" \
    --seed "$SEED" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --result_path "$result_file" \
    >"$log_file" 2>&1

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo "SUCCESS"
elif [[ $exit_code -eq 124 ]]; then
    echo "TIMEOUT"
else
    echo "FAILED (exit code: $exit_code)"
    echo "--------- LOG TAIL ---------"
    tail -n 15 "$log_file"
fi

echo "Logs saved at: $log_file"
echo "Results saved at: $result_file"
echo "------------------------------------------"
exit "$exit_code"
