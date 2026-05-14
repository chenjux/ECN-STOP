# Pipeline

This repository keeps the expensive STOP data-construction steps in notebooks and the repeatable runtime steps in shell/Python entrypoints.

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

## 2. Self-Distill Sampling

```bash
./pruned_data_pipeline/best_of_N_inference.sh \
  -m "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  -d "./data/self_distill_prompts.jsonl" \
  -o "./best_of_n_results/qwen" \
  -s 1,2,3,4
```

Use `pruned_data_pipeline/prm12k_self_distill.ipynb` to select correct candidates and write `./pruned_data_pipeline/data/self_distill_best_of_N_seed.jsonl`.

## 3. Structured Pruning

Run the annotation and ECN pruning notebooks:

```text
pruned_data_pipeline/google_batch_api.ipynb
pruned_data_pipeline/prm12k_tree_pruning.ipynb
```

The expected training output is a Swift-compatible JSONL file such as:

```text
./data/ecn_self_distill_qwen.jsonl
```

## 4. Train And Export

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

## 5. Evaluate

```bash
python3 ./eval/auto_eval.py \
  --model "./train/model/example/checkpoint-100-merged=stop_qwen" \
  --datasets math_500,gsm8k,aime24
```

Summarize EvalScope outputs:

```bash
python3 ./eval/report.py ./outputs --output-csv ./outputs/summary.csv
```

## 6. Static Checks

```bash
./scripts/check.sh
```
