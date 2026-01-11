#!/usr/bin/env bash
# ============================================================
# This is a simple example script for single-seed inference, which can be adapted for self-stilled data generation or answer checking.
# Example: ./infer_single.sh -m "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" -d "path/to/dataset.jsonl" -o "path/to/output_dir"
# ============================================================

# 
MODEL_ID=""               #   Model ID/dir "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASET_FILE=""           #   dataset
OUTPUT_DIR="./results"    # 

# Hyperparameters
BACKEND="vllm"
TEMPERATURE=0.7
REPETITION_PENALTY=1
TOP_P=1
SEED=1
MAX_NEW_TOKENS=16384
TIMEOUT_HOURS=2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

# ==================== cmd ====================
usage() {
    echo "method: $0 -m <model_id_or_path> -d <dataset_file> [-o <output_dir>] [-h]"
    echo "  -m: Model ID（ModelScope/HF"
    echo "  -d: Data jsonl file"
    echo "  -o: output dir (default: ./results）"
    echo "  -h: help"
    exit 1
}

while getopts "m:d:o:h" opt; do
    case $opt in
        m) MODEL_ID="$OPTARG" ;;
        d) DATASET_FILE="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        h) usage ;;
        ?) usage ;;
    esac
done

if [ -z "$MODEL_ID" ] || [ -z "$DATASET_FILE" ]; then
    echo "Error: -m and -d parameters"
    usage
fi

# ==================== filename ====================
mkdir -p "$OUTPUT_DIR"
dataset_name=$(basename "$DATASET_FILE" .jsonl)
result_file="$OUTPUT_DIR/${dataset_name}_result.jsonl"
log_file="$OUTPUT_DIR/${dataset_name}.log"

echo "---------------- RUNNING ----------------"
echo "Model: $MODEL_ID"
echo "Dataset: $DATASET_FILE"
echo "Output Dir: $OUTPUT_DIR"
echo "Result: $result_file"
echo "Log: $log_file"
echo "------------------------------------------"

# ==================== inference ====================
timeout ${TIMEOUT_HOURS}h swift infer \
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

# ==================== Results ====================
if [ $exit_code -eq 0 ]; then
    echo "✓ SUCCESS"
elif [ $exit_code -eq 124 ]; then
    echo "⏰ TIMEOUT"
else
    echo "✗ FAILED (exit code: $exit_code)"
    echo "--------- LOG TAIL ---------"
    tail -n 15 "$log_file"
fi

echo "Logs saved at: $log_file"
echo "Results saved at: $result_file"
echo "------------------------------------------"