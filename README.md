# From Traces to Trees: Structured On-Policy Pruning of Long-Form Reasoning in Reasoning Language  Models

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

```
Main
.
├── pruned_data_pipeline/ 
│   ├── data/
│   │   ├── batch_requests_taxonomy
│   │   ├── batch_requests_conclusion
│   │   └── prm12k.csv                #Raw data start from here
│   ├──  prm12k_self_distill.ipynb    #FCCA  Step 1
│   ├──  google_batch_api.ipynb       #FCCA  Step 2
│   └──  prm12k_tree_pruning.ipynb    #FCCA  Step 3
├── inference/                        #Inference for self-distill response generation and answer checking
│   ├── inference.py                    
│   └── best_of_N_inference.py
├── training/
│   ├── export_model.sh                 #Merge Adaptor
│   ├── run.sh                          #Use this script to run. 
│   └── train.sh                        #Training hyperparameters
├── eval/
│   └── auto_eval.py
├── outputs/
├── figs/
└── README.md
```
## Quick Start

1. Environment Setup (Recommended)

```
conda create -n fcca python=3.10 -y
conda activate fcca
pip install modelscope "modelscope-swift[llm]" evalscope vllm transformers
```

2. Train FCCA Pruning model
```
./train/run.sh
```

3. Evaluate the Pruned Model
```
python ./eval/auto_evalscope.py
```

## Main Experimental Settings
```
Base Models: DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-llama-8B
Datasets: PRM 12k (1000 selected)
Training Lengths:  4096 / 8192
Random Seeds: 1
Decoding Strategy: top-p (0.95)
```

## Evaluation Benchmarks

Math500       
GSM8K         
AIME 2024     

## Citation

@article{xu2026fcca,
  title     = {FCCA-Pruning: Structured Chain-of-Thought Pruning for Efficient Reasoning in Large Language Models},
  author    = {Xu, Chenjun and ...},
  journal   = {arXiv preprint arXiv:XXXX.XXXXX},
  year      = {2026}
}


## Notes & Known Limitations

- Long contexts (≥8192) require high GPU memory
- Pruning effectiveness varies by model size and task

Happy experimenting!