#!/bin/bash

# shellcheck disable=SC2206
#SBATCH -p compute 
#SBATCH --job-name=self_bleu
#SBATCH --output=mistral_self_bleu.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --tasks-per-node=1
#SBATCH --time=04:00:00

# Example usage: Replace the path below with your target JSON file
# python calc_self_bleu.py outputs/Qwen_Qwen2_5-7B-Instruct__gsm8k_multi____iterate/baseline.json

# Or loop through a directory of results:
for f in /scratch/ss13750/nnsight/outputs/mistralai_Mistral-7B-Instruct-v0_3__gsm8k_multi____iterate/*.json; do
    echo "Processing $f"
    python calc_self_bleu.py "$f"
done

# python calc_self_bleu.py "/scratch/ss13750/nnsight/outputs/Qwen_Qwen2_5-7B-Instruct__gsm8k_multi____iterate/results_27.json"
