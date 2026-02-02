#!/bin/bash

# shellcheck disable=SC2206
## SBATCH -p compute
#SBATCH -p nvidia
##SBATCH -q nvidia-xxl
#SBATCH --gres=gpu:1
##SBATCH --constraint=80g
#SBATCH --job-name=new8iter_dolci_mistral   
#SBATCH --output=new8iter_dolci_mistral.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --tasks-per-node=1
#SBATCH --time=1-00:00:00

# export VLLM_USE_V1="0"
export HF_ALLOW_CODE_EVAL="1"


model_name="/scratch/id/nnsight/lm-eval-scripts/merged_models/merged_mistral_new8iter_dolci_mistral_checkpoint11250"

# Adapter path (Leave empty to run base model only)
adapter_path=""

GPUS_per_model=1

if [ -z "$adapter_path" ]; then
    # Run Base Model with vLLM (Faster)
    echo "Running Base Model with vLLM..."
    lm_eval --model vllm \
        --model_args "pretrained=${model_name},tensor_parallel_size=${GPUS_per_model},dtype=auto,gpu_memory_utilization=0.85" \
        --tasks mmlu,arc_challenge,arc_easy,hellaswag,winogrande,piqa,openbookqa \
        --num_fewshot 5 \
        --confirm_run_unsafe_code \
        --gen_kwargs "temperature=0.9,top_p=0.95,do_sample=True" \
        --limit 0.2 \
        --output_path ./outputs/
else
    # Run Adapter with HF Backend (Required for LoRA in lm-eval)
    echo "Running Adapter with vllm Backend..."
    lm_eval --model vllm \
        --model_args "pretrained=${model_name},lora_local_path=${adapter_path},max_lora_rank=64,tensor_parallel_size=${GPUS_per_model},dtype=auto,gpu_memory_utilization=0.85,enable_prefix_caching=False,enable_chunked_prefill=False" \
        --tasks mmlu,arc_challenge,arc_easy,hellaswag,winogrande,piqa,openbookqa \
        --num_fewshot 5 \
        --confirm_run_unsafe_code \
        --gen_kwargs "temperature=0.9,top_p=0.95,do_sample=True" \
        --limit 0.2 \
        --output_path ./outputs/
fi
