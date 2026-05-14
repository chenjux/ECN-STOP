#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <checkpoint_dir> [output_dir]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$ROOT_DIR/scripts/common.sh"

CHECKPOINT_DIR="$1"
OUTPUT_DIR="${2:-}"

require_dir "$CHECKPOINT_DIR"
SWIFT_BIN="$(require_modelscope_swift)"

cmd=(
    "$SWIFT_BIN" export
    --adapters "$CHECKPOINT_DIR"
    --merge_lora true
)

if [[ -n "$OUTPUT_DIR" ]]; then
    ensure_dir "$OUTPUT_DIR"
    cmd+=(--output_dir "$OUTPUT_DIR")
fi

echo "Exporting merged LoRA model from: $CHECKPOINT_DIR"
"${cmd[@]}"
