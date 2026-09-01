#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <checkpoint_dir>"
    exit 1
fi

CHECKPOINT_DIR="$1"

if ! command -v swift >/dev/null 2>&1; then
    echo "ERROR: swift command not found. Please make sure it is installed and in PATH."
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

swift export \
    --adapters "$CHECKPOINT_DIR" \
    --merge_lora true
