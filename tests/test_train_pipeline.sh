#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$TEST_DIR"' EXIT

mkdir -p "$TEST_DIR/bin" "$TEST_DIR/data" "$TEST_DIR/output" "$TEST_DIR/log"
touch "$TEST_DIR/data/ecn_self_distill_qwen.jsonl"

cat >"$TEST_DIR/bin/swift" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

command_name="$1"
shift

case "$command_name" in
    sft)
        output_dir=""
        while [ "$#" -gt 0 ]; do
            if [ "$1" = "--output_dir" ]; then
                output_dir="$2"
                break
            fi
            shift
        done
        test -n "$output_dir"
        mkdir -p "$output_dir/checkpoint-100"
        touch "$output_dir/checkpoint-100/adapter_config.json"
        ;;
    export)
        adapters=""
        while [ "$#" -gt 0 ]; do
            if [ "$1" = "--adapters" ]; then
                adapters="$2"
                break
            fi
            shift
        done
        test -n "$adapters"
        mkdir -p "${adapters}-merged"
        ;;
    *)
        echo "Unexpected swift command: $command_name" >&2
        exit 1
        ;;
esac
EOF
chmod +x "$TEST_DIR/bin/swift"

(
    cd "$TEST_DIR"
    PATH="$TEST_DIR/bin:$PATH" \
    MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    DATASET_DIR="$TEST_DIR/data" \
    OUTPUT_BASE_DIR="$TEST_DIR/output" \
    LOG_DIR="$TEST_DIR/log" \
        bash "$ROOT_DIR/train/run.sh"
)

checkpoint="$(find "$TEST_DIR/output" -type d -name checkpoint-100 -print -quit)"
test -n "$checkpoint"
test -d "${checkpoint}-merged"

if find "$TEST_DIR/output" "$TEST_DIR/log" -type d -name deepseek-ai | grep -q .; then
    echo "Model ID created an unintended nested directory" >&2
    exit 1
fi

echo "Training pipeline smoke test passed."
