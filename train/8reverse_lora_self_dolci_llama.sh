#!/bin/bash

# shellcheck disable=SC2206
#SBATCH -p nvidia
#SBATCH -q nvidia-xxl
#SBATCH --job-name=8reverse_lora_self_dolci_llama
#SBATCH --output=8reverse_lora_self_dolci_llama.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --mem-per-cpu=64GB
#SBATCH --tasks-per-node=1
#SBATCH --time=1-00:00:00

srun bash -c 'echo "$(hostname):"; nvidia-smi --query-gpu=name,index --format=csv,noheader; echo'

export GPUS_PER_NODE=1

# Get the list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Count unique nodes
NUM_NODES=${#NODELIST[@]}

# Use the count in your commands
echo "Job allocated ${NUM_NODES} unique nodes: ${NODELIST[*]}"

# CSV_FILE="/scratch/ss13750/nnsight/openthoughts/dolci_data.csv"
# CSV_FILE="/scratch/ss13750/nnsight/openthoughts/dolci_generated_responses.csv"
CSV_FILE="/scratch/ss13750/nnsight/openthoughts/dolci_llama_generated_responses.csv"
# CSV_FILE="/scratch/ss13750/anubhav/train_responses_complete_20251027_221538.csv"

# Calculate processes (currently 1 GPU per node in SBATCH)

export NUM_PROCESSES=$(($NUM_NODES * $GPUS_PER_NODE))
# /scratch/ss13750/nnsight/out_models/Qwen_Qwen2_5-7B-Instruct__1-5-7-11-12-20-25

# Run training with accelerate launch
srun --nodes=$NUM_NODES --ntasks=$NUM_NODES accelerate launch \
     --config_file ./zero3.yaml \
     --num_processes $NUM_PROCESSES \
     --num_machines $NUM_NODES \
     --main_process_ip ${NODELIST[0]} \
     --machine_rank $SLURM_PROCID \
     --rdzv_backend c10d \
     sft_trainer.py \
        --model_name /scratch/ss13750/nnsight/out_models/meta-llama_Llama-3_1-8B-Instruct__24-25-26-27-28-29-30-31-32 \
        --dataset_type dolci \
        --csv_file_path  "$CSV_FILE" \
        --use_completion_only True \
        --use_lora False \
        --lora_r 16 \
        --lora_alpha 32 \
        --use_qlora True \
        --bnb_4bit_compute_dtype bfloat16 \
        --bnb_4bit_quant_type nf4 \
        --add_chat_template False \
        --wandb_project lora_sft_csv_ablate \
        --output_dir ./outputs/8reverse_lora_self_dolci_llama \
        --run_name 8reverse_lora_self_dolci_llama \
        --random_weights False \
        --resume_from_checkpoint False \
        --learning_rate 2e-4 \
        --adam_beta1 0.9 \
        --adam_beta2 0.99 \
        --weight_decay 0.0001 \
        --warmup_steps 50 \
        --lr_scheduler_type constant_with_warmup \
        --logging_steps 10 \
        --bf16 True \
        --bf16_full_eval True \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --gradient_checkpointing True \
        --gradient_checkpointing_kwargs '{"use_reentrant": true}' \
        --num_train_epochs 1 \
        --save_strategy steps \
        --save_steps 3000 \
        --max_grad_norm 1.0 \
        --report_to wandb \
        --max_seq_length 8192 \
        --max_steps -1 \
        --eval_strategy epoch \
        --eval_on_start False \
        --eval_accumulation_steps 2 \
        --average_tokens_across_devices False \
        --seed 42

wait