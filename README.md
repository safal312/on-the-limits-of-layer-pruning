# On the Limits of Layer Pruning for Generative Reasoning in LLMs

## Abstract
Recent works have shown that layer pruning can compress large language models (LLMs) while retaining strong performance on classification benchmarks with little or no finetuning. However, existing pruning techniques often suffer severe degradation on generative reasoning tasks. Through a systematic study across multiple model families, we find that tasks requiring multi-step reasoning are particularly sensitive to depth reduction. Beyond surface-level text degeneration, we observe degradation of critical algorithmic capabilities, including arithmetic computation for mathematical reasoning and balanced parenthesis generation for code synthesis. Under realistic post-training constraints, without access to pretraining-scale data or compute, we evaluate a simple mitigation strategy based on supervised finetuning with Self-Generated Responses. This approach achieves strong recovery on classification tasks, retaining up to 90\% of baseline performance, and yields substantial gains of up to 20--30 percentage points on generative benchmarks compared to prior post-pruning techniques. Crucially, despite these gains, recovery for generative reasoning remains fundamentally limited relative to classification tasks and is viable primarily at lower pruning ratios. Overall, we characterize the practical limits of layer pruning for generative reasoning and provide guidance on when depth reduction can be applied effectively under constrained post-training regimes.

---

## Overview

This repository contains the implementation for **On the Limits of Layer Pruning for Generative Reasoning in LLMs**, a framework for exploring and pushing the limits of layer pruning for generative reasoning in Large Language Models (LLMs).

## Project Structure

```text
.
├── layer_pruner/        # Code for (iterative) layer pruning
│   ├── script.py        # Main entry point for the (iterative) ablation algorithm
│   ├── merge_model.py   # Script to remove specific layers from a model
│   └── utils/           # Helper utilities for config generation and evaluation
├── train/               # Code for training
│   └── sft_trainer.py   # SFT/LoRA/QLoRA trainer using HF TRL
├── eval/                # Code for evaluation
│   ├── classif_eval/    # Classification-based tasks
│   └── gen_eval/        # Generation-based tasks (GSM8K, etc.)
├── self_generate/       # Code for response generation
└── analysis/            # Jupyter notebooks for results visualization
```

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd limit-of-layer-pruning
   ```

2. Create the environment from the provided YAML files:
   ```bash
   # For the main pruning and evaluation logic
   conda env create -f layer_pruner/merge_eval_env.yaml
   
   # For training
   conda env create -f train/trl_env.yaml

   # For analysis with nnsight
   conda env create -f analysis/nnsight_env.yaml
   ```

### Core Dependencies

This project leverages several key open-source libraries:
- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**: Used for standardized evaluation of pruned models.
- **[mergekit](https://github.com/arcee-ai/mergekit)**: Used for the underlying layer removal and model merging logic.
- **[nnsight](https://nnsight.net/)**: Used for deep analysis and visualization of model internals.


## Citation

If you find this work useful in your research, please cite:

```bibtex

```
