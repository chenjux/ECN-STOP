# Pipeline

This repository keeps the expensive STOP data-construction experiments in notebooks and the repeatable runtime steps in shell/Python entrypoints.

Run pipeline commands from the repository root with `python3 -m pipeline.<stage>`.

## 1. Environment

```bash
conda create -n stop python=3.10 -y
conda activate stop
pip install -r requirements.txt
```

If your machine also has Apple's Swift compiler, make sure the ModelScope Swift CLI is first on `PATH`, or set:

```bash
export SWIFT_CLI=/path/to/modelscope-swift
```

## 2. Prepare Self-Distill Prompts

```bash
python3 -m pipeline.prepare_prompts \
  --input-csv ./pruned_data_pipeline/data/prm12k.csv \
  --output-dir ./pruned_data_pipeline/data
```

This writes chunked prompt files like:

```text
./pruned_data_pipeline/data/self_distill_questions_dataset_1.jsonl
```

## 3. Self-Distill Sampling

```bash
./pruned_data_pipeline/best_of_N_inference.sh \
  -m "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  -d "./pruned_data_pipeline/data/self_distill_questions_dataset_1.jsonl" \
  -o "./best_of_n_results/qwen" \
  -s 1,2,3,4
```

## 4. Build Answer-Check Prompts

```bash
python3 -m pipeline.build_judge_prompts \
  --seed-output 1=./best_of_n_results/qwen/self_distill_questions_dataset_1_seed1.jsonl \
  --seed-output 2=./best_of_n_results/qwen/self_distill_questions_dataset_1_seed2.jsonl \
  --seed-output 3=./best_of_n_results/qwen/self_distill_questions_dataset_1_seed3.jsonl \
  --seed-output 4=./best_of_n_results/qwen/self_distill_questions_dataset_1_seed4.jsonl \
  --output-dir ./pruned_data_pipeline/data/judge_prompts
```

Run those judge prompt files through the inference script or batch API, then collect the answer-check result files.

## 5. Select Good Traces

```bash
python3 -m pipeline.select_good_traces \
  --seed-output 1=./best_of_n_results/qwen/self_distill_questions_dataset_1_seed1.jsonl \
  --seed-output 2=./best_of_n_results/qwen/self_distill_questions_dataset_1_seed2.jsonl \
  --seed-output 3=./best_of_n_results/qwen/self_distill_questions_dataset_1_seed3.jsonl \
  --seed-output 4=./best_of_n_results/qwen/self_distill_questions_dataset_1_seed4.jsonl \
  --check-result 1=./pruned_data_pipeline/data/self_distill_seed1_check_result.jsonl \
  --check-result 2=./pruned_data_pipeline/data/self_distill_seed2_check_result.jsonl \
  --check-result 3=./pruned_data_pipeline/data/self_distill_seed3_check_result.jsonl \
  --check-result 4=./pruned_data_pipeline/data/self_distill_seed4_check_result.jsonl \
  --output ./pruned_data_pipeline/data/self_distill_best_of_N_seed.jsonl
```

The default selection policy is `median_correct`, matching the notebook logic.

## 6. Segment Traces

```bash
python3 -m pipeline.segment_traces \
  --input ./pruned_data_pipeline/data/self_distill_best_of_N_seed.jsonl \
  --output ./pruned_data_pipeline/data/self_distill_best_of_N_seed_think_heuristic_segmentation.jsonl
```

## 7. Structured Annotation

Run the annotation and ECN pruning notebooks:

```bash
python3 -m pipeline.build_annotation_batches \
  --task taxonomy \
  --input ./pruned_data_pipeline/data/self_distill_best_of_N_seed_think_heuristic_segmentation.jsonl \
  --output-dir ./pruned_data_pipeline/data/batch_requests_taxonomy
```

Submit the generated request files through `pruned_data_pipeline/google_batch_api.ipynb`, then collect results:

```bash
python3 -m pipeline.collect_annotations \
  --kind taxonomy \
  --input-dir ./pruned_data_pipeline/data/batch_requests_taxonomy \
  --join ./pruned_data_pipeline/data/self_distill_best_of_N_seed_think_heuristic_segmentation.jsonl \
  --drop-missing \
  --output ./pruned_data_pipeline/data/self_filter_all_taxonomy.jsonl
```

Build conclusion annotation requests:

```bash
python3 -m pipeline.build_annotation_batches \
  --task conclusion \
  --input ./pruned_data_pipeline/data/self_filter_all_taxonomy.jsonl \
  --output-dir ./pruned_data_pipeline/data/conclusion_batches
```

Collect conclusion results:

```bash
python3 -m pipeline.collect_annotations \
  --kind conclusion \
  --input-dir ./pruned_data_pipeline/data/conclusion_batches \
  --drop-missing \
  --output ./pruned_data_pipeline/data/self_filter_all_conclusions.jsonl
```

## 8. ECN Pruning

```bash
python3 -m pipeline.prune_ecn \
  --selected ./pruned_data_pipeline/data/self_distill_best_of_N_seed.jsonl \
  --segmented ./pruned_data_pipeline/data/self_distill_best_of_N_seed_think_heuristic_segmentation.jsonl \
  --conclusions ./pruned_data_pipeline/data/self_filter_all_conclusions.jsonl \
  --output ./data/ecn_self_distill_qwen.jsonl \
  --debug-output ./pruned_data_pipeline/data/ecn_pruning_debug.jsonl
```

Validate artifacts:

```bash
python3 -m pipeline.validate_artifacts \
  --artifact selected=./pruned_data_pipeline/data/self_distill_best_of_N_seed.jsonl \
  --artifact segmented=./pruned_data_pipeline/data/self_distill_best_of_N_seed_think_heuristic_segmentation.jsonl \
  --artifact taxonomy=./pruned_data_pipeline/data/self_filter_all_taxonomy.jsonl \
  --artifact conclusion=./pruned_data_pipeline/data/self_filter_all_conclusions.jsonl \
  --artifact training=./data/ecn_self_distill_qwen.jsonl
```

The final training output is a Swift-compatible JSONL file:

```text
./data/ecn_self_distill_qwen.jsonl
```

## 9. Train And Export

```bash
./train/run.sh
```

Useful overrides:

```bash
DATASETS="./data/ecn_self_distill_qwen.jsonl" \
MAX_LENGTHS="8192" \
SEEDS="1" \
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
./train/run.sh
```

`train/run.sh` finds the latest `checkpoint-*` under each training output and calls `train/export_model.sh` to merge the LoRA adapter.

## 10. Evaluate

```bash
python3 ./eval/auto_eval.py \
  --model "./train/model/example/checkpoint-100-merged=stop_qwen" \
  --datasets math_500,gsm8k,aime24
```

Summarize EvalScope outputs:

```bash
python3 ./eval/report.py ./outputs --output-csv ./outputs/summary.csv
```

## 11. Static Checks

```bash
./scripts/check.sh
```
