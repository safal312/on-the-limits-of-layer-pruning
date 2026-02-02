#!/bin/bash

# shellcheck disable=SC2206
## SBATCH -p compute
#SBATCH -p cs
#SBATCH -A condo_cs_ross
##SBATCH -q nvidia-xxl
#SBATCH --gres=gpu:1
#SBATCH --job-name=merge_lora
#SBATCH --output=out_merge_lora.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=01:00:00

# Base: The pruned llama model
# BASE_MODEL="/scratch/ss13750/nnsight/out_models/meta-llama_Llama-3_1-8B-Instruct__24-25-26-27-28-29-30-31"
BASE_MODEL="/scratch/ss13750/nnsight/out_models/mistralai_Mistral-7B-Instruct-v0_3__21-22-23-24-25-26-27-28"

# Adapter: The LoRA checkpoint from current folder
# ADAPTER_PATH="/scratch/ss13750/rl/outputs/6reverse_lora_self_dolci_llama/checkpoint-9000"
# ADAPTER_PATH="/scratch/ss13750/rl/outputs/8reverse_lora_alpaca_llama/checkpoint-5823"
# ADAPTER_PATH="/scratch/ss13750/rl/outputs/8reverse_lora_self_alpaca_llama/checkpoint-5823"
# ADAPTER_PATH="/scratch/ss13750/rl/outputs/8reverse_lora_dolci_llama/checkpoint-11250"
# ADAPTER_PATH="/scratch/ss13750/rl/outputs/8reverse_lora_self_dolci_llama/checkpoint-11250"
ADAPTER_PATH="/scratch/ss13750/rl/outputs/8cossim_lora_self_dolci_mistral/checkpoint-11250"

# Output: A new directory for the standalone model
OUTPUT_PATH="/scratch/ss13750/nnsight/out_models/merged_mistral_8cossim_self_dolci_mistral_checkpoint11250"

python merge_lora.py \
    --base_model "$BASE_MODEL" \
    --adapter "$ADAPTER_PATH" \
    --output "$OUTPUT_PATH"
