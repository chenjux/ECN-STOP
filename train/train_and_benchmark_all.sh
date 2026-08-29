#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/scripts/common.sh"

MODEL_SPECS="${MODEL_SPECS:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
DATASETS="${DATASETS:-$ROOT_DIR/data/ecn_self_distill_qwen.jsonl}"
MAX_LENGTHS="${MAX_LENGTHS:-8192}"
SEEDS="${SEEDS:-1}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$SCRIPT_DIR/model}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/log}"

BENCHMARK_DATASETS="${BENCHMARK_DATASETS:-${EVAL_DATASETS:-math_500}}"
BENCHMARK_BASE_PORT="${BENCHMARK_BASE_PORT:-${EVAL_BASE_PORT:-8801}}"
BENCHMARK_GPU_ID="${BENCHMARK_GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
BENCHMARK_BATCH_SIZE="${BENCHMARK_BATCH_SIZE:-32}"
BENCHMARK_MAX_TOKENS="${BENCHMARK_MAX_TOKENS:-8192}"
BENCHMARK_SERVER_TIMEOUT="${BENCHMARK_SERVER_TIMEOUT:-300}"
BENCHMARK_COOLDOWN_SECONDS="${BENCHMARK_COOLDOWN_SECONDS:-15}"
BENCHMARK_LOG_DIR="${BENCHMARK_LOG_DIR:-$ROOT_DIR/vllm_logs}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"

timestamp="$(date +%Y_%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/train_benchmark_$timestamp}"
MANIFEST_FILE="${MANIFEST_FILE:-$RUN_DIR/benchmark_models.tsv}"

usage() {
    cat <<'EOF'
Usage: train/train_and_benchmark_all.sh

Environment:
  MODEL_SPECS                 Comma-separated models to train.
                              Each item may be MODEL_PATH or MODEL_PATH=ALIAS.
  DATASETS                    Comma-separated training JSONL files.
                              Applied to every model in MODEL_SPECS.
  MAX_LENGTHS                 Comma-separated max lengths, same count as DATASETS.
  SEEDS                       Comma-separated seeds.
  OUTPUT_BASE_DIR             Training output directory.
  LOG_DIR                     Training log directory.
  SWIFT_CLI                   ModelScope Swift executable if not "swift".
  CUDA_VISIBLE_DEVICES        GPU(s) used for training.

Benchmark environment:
  RUN_BENCHMARK               1 to run benchmark after training, 0 to only train/export.
  BENCHMARK_DATASETS          EvalScope datasets, comma-separated. Default: math_500.
  BENCHMARK_BASE_PORT         First vLLM port. Default: 8801.
  BENCHMARK_GPU_ID            GPU id used by benchmark vLLM. Default: CUDA_VISIBLE_DEVICES or 0.
  BENCHMARK_BATCH_SIZE        Eval batch size. Default: 32.
  BENCHMARK_MAX_TOKENS        Generation max tokens. Default: 8192.
  BENCHMARK_SERVER_TIMEOUT    vLLM readiness timeout seconds. Default: 300.
  BENCHMARK_COOLDOWN_SECONDS  Cooldown between model benchmarks. Default: 15.

Example:
  MODEL_SPECS="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B=qwen7b,deepseek-ai/DeepSeek-R1-Distill-Llama-8B=llama8b" \
  DATASETS="./data/ecn_self_distill_qwen.jsonl" \
  MAX_LENGTHS="8192" \
  SEEDS="1,2,3" \
  BENCHMARK_DATASETS="math_500,gsm8k,aime24" \
  ./train/train_and_benchmark_all.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

model_alias() {
    local model_spec="$1"
    local alias

    if [[ "$model_spec" == *"="* ]]; then
        alias="${model_spec#*=}"
    else
        model_path="$model_spec"
        alias="$(basename "$model_path")"
    fi

    safe_name "$alias"
}

model_path() {
    local model_spec="$1"
    if [[ "$model_spec" == *"="* ]]; then
        printf '%s\n' "${model_spec%%=*}"
    else
        printf '%s\n' "$model_spec"
    fi
}

ensure_dir "$RUN_DIR"
ensure_dir "$OUTPUT_BASE_DIR"
ensure_dir "$LOG_DIR"
: >"$MANIFEST_FILE"

IFS=',' read -r -a MODEL_LIST <<<"$MODEL_SPECS"

if [[ "${#MODEL_LIST[@]}" -eq 0 ]]; then
    die "MODEL_SPECS must contain at least one model."
fi

echo "Run directory: $RUN_DIR"
echo "Benchmark manifest: $MANIFEST_FILE"
echo "Models: ${MODEL_LIST[*]}"
echo "Datasets: $DATASETS"
echo "Max lengths: $MAX_LENGTHS"
echo "Seeds: $SEEDS"

for spec in "${MODEL_LIST[@]}"; do
    model="$(model_path "$spec")"
    alias="$(model_alias "$spec")"
    model_output_dir="$OUTPUT_BASE_DIR/$alias"

    echo
    echo "================================================================"
    echo "Training model: $model"
    echo "Alias: $alias"
    echo "Output base: $model_output_dir"
    echo "================================================================"

    MODEL="$model" \
    MODEL_LABEL="$alias" \
    DATASETS="$DATASETS" \
    MAX_LENGTHS="$MAX_LENGTHS" \
    SEEDS="$SEEDS" \
    OUTPUT_BASE_DIR="$model_output_dir" \
    LOG_DIR="$LOG_DIR" \
    MANIFEST_FILE="$MANIFEST_FILE" \
    bash "$SCRIPT_DIR/run.sh"
done

if [[ ! -s "$MANIFEST_FILE" ]]; then
    die "No exported models were written to benchmark manifest: $MANIFEST_FILE"
fi

echo
echo "Exported models:"
cat "$MANIFEST_FILE"

if [[ "$RUN_BENCHMARK" != "1" ]]; then
    echo "RUN_BENCHMARK=$RUN_BENCHMARK, skipping benchmark."
    exit 0
fi

benchmark_args=()
benchmark_model_count=0
while IFS=$'\t' read -r merged_model_path model_name; do
    [[ -n "$merged_model_path" && -n "$model_name" ]] || continue
    require_dir "$merged_model_path"
    benchmark_args+=(--model "${merged_model_path}=${model_name}")
    benchmark_model_count=$((benchmark_model_count + 1))
done <"$MANIFEST_FILE"

if [[ "${#benchmark_args[@]}" -eq 0 ]]; then
    die "No valid benchmark model specs found in $MANIFEST_FILE"
fi

echo
echo "================================================================"
echo "Running benchmark for $benchmark_model_count model specs"
echo "Datasets: $BENCHMARK_DATASETS"
echo "================================================================"

python3 "$ROOT_DIR/eval/auto_eval.py" \
    "${benchmark_args[@]}" \
    --gpu-id "$BENCHMARK_GPU_ID" \
    --base-port "$BENCHMARK_BASE_PORT" \
    --datasets "$BENCHMARK_DATASETS" \
    --eval-batch-size "$BENCHMARK_BATCH_SIZE" \
    --max-tokens "$BENCHMARK_MAX_TOKENS" \
    --server-timeout "$BENCHMARK_SERVER_TIMEOUT" \
    --cooldown-seconds "$BENCHMARK_COOLDOWN_SECONDS" \
    --log-dir "$BENCHMARK_LOG_DIR"

echo "Train/export/benchmark run completed."
