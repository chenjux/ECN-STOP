# FCCA-Pruning

Official implementation of FCCA-Pruning:  
Structured Chain-of-Thought Pruning for Efficient Reasoning in Large Language Models

Paper: https://arxiv.org/abs/XXXX.XXXXX  
ACL 2026 (under review)

## Introduction

FCCA-Pruning is a structured chain-of-thought pruning method that retains only the minimal prefix up to the first correct conclusion, removing all redundant post-solution reasoning steps. This significantly reduces inference costs, mitigates overthinking behavior, and preserves semantic coherence and task performance.

## Key Features

- Minimal prefix structured pruning (FCCA)
- Built on ModelScope + Swift ecosystem
- Standardized evaluation with EvalScope
- Reproducible pruning and evaluation pipelines

## Repository Structure

.
├── pruned_data_pipeline/       # Core FCCA pruning logic
├── training/                   # Fine-tuning & self-distillation code
├── eval/                       # EvalScope-based evaluation scripts
├── data/                       # Data processing/generation scripts (no raw data)
├── scripts/                    # Ready-to-run experiment scripts
├── figs/                       # Figures for the paper
├── outputs/                    # Experiment outputs (git ignore recommended)
└── README.md

## Quick Start

1. Environment Setup (Recommended)

conda create -n fcca python=3.10 -y
conda activate fcca

pip install modelscope "modelscope-swift[llm]" evalscope vllm transformers


2. Run FCCA Pruning (Example)

python scripts/run_fcca_pruning.py \
  --model qwen-7b \
  --dataset math500 \
  --max_length 4096 \
  --output_dir outputs/fcca_qwen_math500_4096 \
  --seed 42


3. Evaluate the Pruned Model

python eval/run_evalscope.py \
  --model outputs/fcca_qwen_math500_4096 \
  --datasets math500 gsm8k aime2024 \
  --batch_size 32


## Main Experimental Settings

Base Models: Qwen-7B, Qwen-14B
Datasets: Math500, GSM8K, AIME 2024
Context Lengths: 2048 / 4096 / 8192
Random Seeds: 10 different seeds
Decoding Strategy: top-p (default)


## Evaluation Benchmarks

Math500      - Main result table
GSM8K        - Standard 8-shot CoT
AIME 2024    - Extremely challenging competition problems, evaluation only

Important: Due to licensing and policy constraints, we do not distribute self-distilled or pruned reasoning traces.
However, you can regenerate them using scripts in pruned_data_pipeline.


## Citation

@article{xu2026fcca,
  title     = {FCCA-Pruning: Structured Chain-of-Thought Pruning for Efficient Reasoning in Large Language Models},
  author    = {Xu, Chenjun and ...},
  journal   = {arXiv preprint arXiv:XXXX.XXXXX},
  year      = {2026}
}


## Notes & Known Limitations

- Currently tested primarily on Qwen series; support for other models coming soon
- Long contexts (≥8192) require high GPU memory
- Pruning effectiveness varies by model size and task
- AIME 2024 is extremely difficult—even strong models have low accuracy

Happy experimenting!