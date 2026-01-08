#!/usr/bin/env bash
# ============================================================
# Best-of-N Inference Script (Runs multiple rounds with different seeds)
# ============================================================

# Default values (can be overridden via command-line arguments)
MODEL_ID=""                  # Model ID (ModelScope/HF) or local path
DATASET_FILE=""              # Path to the dataset .jsonl file
OUTPUT_DIR="./best_of_n_results"  # Default output directory
SEEDS=(1 2 3 4)                 # Default seeds for Best-of-N (can be customized)
N=${#SEEDS[@]}               # Automatically compute N from seed list

# Fixed inference parameters
BACKEND="vllm"
TEMPERATURE=0.7              # Keep 0.7 for diversity
REPETITION_PENALTY=1
TOP_P=1
MAX_NEW_TOKENS=8192          # Important: prevent truncation
TIMEOUT_HOURS=4              # Increased timeout for safety

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

# ==================== Parse command-line arguments ====================
usage() {
    echo "Usage: $0 -m <model_id_or_path> -d <dataset_file> [-o <output_dir>] [-s <seed1,seed2,...>] [-h]"
    echo ""
    echo "  -m  Model ID (e.g., 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B') or local path (required)"
    echo "  -d  Dataset JSONL file path (required)"
    echo "  -o  Output directory (optional, default: ./best_of_n_results)"
    echo "  -s  Comma-separated list of seeds (optional, default: 2,3,4)"
    echo "  -h  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -m \"deepseek-ai/DeepSeek-R1-Distill-Llama-8B\" -d \"data/self_distill.jsonl\" -o \"./results/llama\" -s 1,2,3,4"
    exit 1
}

# Parse options
while getopts "m:d:o:s:h" opt; do
    case $opt in
        m) MODEL_ID="$OPTARG" ;;
        d) DATASET_FILE="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        s) IFS=',' read -ra SEEDS <<< "$OPTARG" ;;
        h) usage ;;
        ?) usage ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_ID" ] || [ -z "$DATASET_FILE" ]; then
    echo "Error: -m and -d are required."
    usage
fi

# Update N based on actual seed count
N=${#SEEDS[@]}

# ==================== Setup paths ====================
mkdir -p "$OUTPUT_DIR"
dataset_name=$(basename "$DATASET_FILE" .jsonl)

echo "=========================================="
echo "Starting Best-of-$N Inference Task"
echo "Model: $MODEL_ID"
echo "Dataset: $DATASET_FILE"
echo "Seeds: ${SEEDS[*]}"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="

# ==================== Run inference for each seed ====================
for CURRENT_SEED in "${SEEDS[@]}"; do
    result_file="$OUTPUT_DIR/${dataset_name}_seed${CURRENT_SEED}.jsonl"
    log_file="$OUTPUT_DIR/${dataset_name}_seed${CURRENT_SEED}.log"

    echo ""
    echo ">>> [Round Start] Running with SEED: $CURRENT_SEED"
    echo ">>> Saving results to: $result_file"

    # Skip if result file already exists (avoid re-running)
    if [ -f "$result_file" ]; then
        echo "⚠️  Result file already exists, skipping this seed..."
        continue
    fi

    timeout ${TIMEOUT_HOURS}h swift infer \
        --model "$MODEL_ID" \
        --val_dataset "$DATASET_FILE" \
        --infer_backend "$BACKEND" \
        --temperature "$TEMPERATURE" \
        --repetition_penalty "$REPETITION_PENALTY" \
        --top_p "$TOP_P" \
        --seed "$CURRENT_SEED" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --result_path "$result_file" \
        >"$log_file" 2>&1

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ [Success] Seed $CURRENT_SEED completed successfully."
    elif [ $exit_code -eq 124 ]; then
        echo "⏰ [Timeout] Seed $CURRENT_SEED exceeded $TIMEOUT_HOURS hours."
    else
        echo "✗ [Failed] Seed $CURRENT_SEED failed (exit code: $exit_code). See log: $log_file"
        echo "--------- Last 15 lines of log ---------"
        tail -n 15 "$log_file"
    fi

    # Optional: Clear GPU cache between runs (uncomment if needed)
    # python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
done

echo ""
echo "=========================================="
echo "🎉 All $N rounds completed!"
echo "Results are ready for Best-of-N selection/filtering."
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="