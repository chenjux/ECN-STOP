#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

python3 -m py_compile \
    eval/auto_eval.py \
    eval/report.py \
    pipeline/*.py

bash -n \
    scripts/common.sh \
    scripts/check.sh \
    train/run.sh \
    train/train.sh \
    train/export_model.sh \
    train/train_and_benchmark_all.sh \
    pruned_data_pipeline/inference.sh \
    pruned_data_pipeline/best_of_N_inference.sh

python3 eval/auto_eval.py --help >/dev/null
python3 eval/report.py --help >/dev/null
python3 -m pipeline.prepare_prompts --help >/dev/null
python3 -m pipeline.build_judge_prompts --help >/dev/null
python3 -m pipeline.select_good_traces --help >/dev/null
python3 -m pipeline.segment_traces --help >/dev/null
python3 -m pipeline.build_annotation_batches --help >/dev/null
python3 -m pipeline.collect_annotations --help >/dev/null
python3 -m pipeline.prune_ecn --help >/dev/null
python3 -m pipeline.validate_artifacts --help >/dev/null
bash pruned_data_pipeline/inference.sh -h >/dev/null
bash pruned_data_pipeline/best_of_N_inference.sh -h >/dev/null
bash train/train_and_benchmark_all.sh -h >/dev/null

echo "Static checks passed."
