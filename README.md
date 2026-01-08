# FCCA-Pruning

# Project Overview

This repository contains code and utilities for training, fine-tuning, and evaluating large language models using the **ModelScope ecosystem**, including **Swift** for efficient fine-tuning and **EvalScope** for standardized evaluation.

---

## Requirements

- Python >= 3.10  
- CUDA-compatible GPU (recommended)  
- `pip` >= 23.0  

---

## Installation

We recommend using a virtual environment.

```bash
conda create -n llm-env python=3.10 -y
conda activate llm-env

## Install ModelScope
pip install modelscope

## Install Swift (ModelScope-Swift) & Evalscope & vllm
pip install "modelscope-swift[llm]"

pip install evalscope
pip install vllm

