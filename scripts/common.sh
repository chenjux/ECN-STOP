#!/usr/bin/env bash

die() {
    echo "ERROR: $*" >&2
    exit 1
}

repo_root() {
    git rev-parse --show-toplevel 2>/dev/null || {
        local script_dir
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        cd "$script_dir/.." && pwd
    }
}

safe_name() {
    printf '%s' "$1" | tr -c '[:alnum:]._-' '_'
}

require_file() {
    local path="$1"
    [[ -f "$path" ]] || die "File not found: $path"
}

require_dir() {
    local path="$1"
    [[ -d "$path" ]] || die "Directory not found: $path"
}

ensure_dir() {
    mkdir -p "$1"
}

require_modelscope_swift() {
    local cli="${SWIFT_CLI:-swift}"

    command -v "$cli" >/dev/null 2>&1 || die "Swift CLI not found: $cli"

    if ! "$cli" sft --help >/dev/null 2>&1; then
        die "ModelScope Swift CLI is not available through '$cli'. Activate the ms-swift environment or set SWIFT_CLI to the correct executable."
    fi

    printf '%s\n' "$cli"
}

run_with_timeout_hours() {
    local hours="$1"
    shift

    local duration="${hours}h"
    if command -v timeout >/dev/null 2>&1; then
        timeout "$duration" "$@"
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$duration" "$@"
    else
        echo "WARNING: timeout/gtimeout not found; running without a timeout." >&2
        "$@"
    fi
}
