#!/bin/bash
#SBATCH -p nvidia
#SBATCH --job-name=qwen_inference
#SBATCH --output=/scratch/id/nnsight/openthoughts/inference.out
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Variables
MODEL="Qwen/Qwen2.5-7B-Instruct"
# INPUT_CSV="/scratch/id/nnsight/openthoughts/dolci_data.csv"
# OUTPUT_CSV="/scratch/id/nnsight/openthoughts/dolci_generated_responses.csv"

INPUT_CSV="/scratch/id/nnsight/openthoughts/alpaca_data.csv"
OUTPUT_CSV="/scratch/id/nnsight/openthoughts/alpaca_qwen_generated_responses.csv"

python -u /scratch/id/nnsight/openthoughts/generate_responses.py \
    --model_path "$MODEL" \
    --input_file "$INPUT_CSV" \
    --output_file "$OUTPUT_CSV" \
    --tensor_parallel_size 1 \
    --max_tokens 8192 \
    --batch_size 1000
