import re
import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from transformers import AutoModel, AutoConfig, OlmoModel, OlmoConfig
from trl.trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from datasets import load_dataset
from math_verify import verify, parse

from dataclasses import dataclass, field, asdict
from typing import Optional

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from dataset_loader import load_math
import pandas as pd


def sft_data_loader(config, tokenizer, dataset):
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        df = ds.to_pandas()
        df = df.sample(n=1100, random_state=42)

        if config.add_chat_template:
            def map_template(row):
                question = row['question']
                answer = row['answer']
                messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False  # Changed to False since we have the complete answer
                )
                return text  # Added return statement

            df['text'] = df.apply(map_template, axis=1)  # Changed to apply with axis=1
        else:
            df['text'] = df['question'] + "\n" + df['answer']

        # Randomly sample 500 rows for the test set
        test_set = df.sample(n=100, random_state=42)

        # Remaining rows for the training set
        train_set = df.drop(test_set.index)
        train_set = train_set.sample(n=1000, random_state=42)
        print("Len of training set: ", len(train_set))
        # train_set = df

        # Convert to Hugging Face Dataset format
        test_dataset = Dataset.from_pandas(test_set.reset_index(drop=True))
        train_dataset = Dataset.from_pandas(train_set.reset_index(drop=True))

        return train_dataset, test_dataset
    
    elif dataset == "triviaqa":
        ds = load_dataset("trivia_qa", "rc.nocontext")["train"]
        df = ds.to_pandas()
        df = df.sample(n=1100, random_state=42)

        if config.add_chat_template:
            def map_template(row):
                question = row['question']
                answer = row['answer']['value']
                messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False  # Changed to False since we have the complete answer
                )
                return text  # Added return statement

            df['text'] = df.apply(map_template, axis=1)  # Changed to apply with axis=1
        else:
            df['text'] = df['question'] + "\n" + df['answer']

        # Randomly sample 500 rows for the test set
        test_set = df.sample(n=100, random_state=42)

        # Remaining rows for the training set
        train_set = df.drop(test_set.index)
        train_set = train_set.sample(n=1000, random_state=42)
        print("Len of training set: ", len(train_set))
        # train_set = df

        # Convert to Hugging Face Dataset format
        test_dataset = Dataset.from_pandas(test_set.reset_index(drop=True))
        train_dataset = Dataset.from_pandas(train_set.reset_index(drop=True))

        return train_dataset, test_dataset


def csv_data_loader(csv_file_path, test_split_ratio=0.1, random_state=42):
    """
    Load data from CSV file with 'prompt' and 'generated_text' columns.
    Combines them into a single 'text' column for training.
    
    Args:
        csv_file_path: Path to CSV file
        test_split_ratio: Ratio of test split (default: 0.1)
        random_state: Random state for reproducibility
    
    Returns:
        train_dataset, test_dataset: Hugging Face Dataset objects
    """
    logging.info(f"Loading CSV data from: {csv_file_path}")
    
    # Load CSV
    df = pd.read_csv(csv_file_path)
    
    # Check required columns exist
    if 'prompt' not in df.columns or 'generated_text' not in df.columns:
        raise ValueError(f"CSV must have 'prompt' and 'generated_text' columns. Found: {df.columns.tolist()}")
    
    # Combine prompt and generated_text
    df['text'] = df['prompt'] + df['generated_text']
    
    logging.info(f"Total samples in CSV: {len(df)}")
    
    # Split into train and test
    if test_split_ratio > 0:
        test_size = max(1, int(len(df) * test_split_ratio))
        test_set = df.sample(n=test_size, random_state=random_state)
        train_set = df.drop(test_set.index)
    else:
        train_set = df
        test_set = df.sample(n=min(100, len(df)), random_state=random_state)  # Small test set for validation
    
    logging.info(f"Training samples: {len(train_set)}")
    logging.info(f"Test samples: {len(test_set)}")
    
    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_set[['text']].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_set[['text']].reset_index(drop=True))
    
    return train_dataset, test_dataset


def dolci_data_loader(csv_file_path, tokenizer, model_path=None, test_split_ratio=0.1, random_state=42):
    """
    Load Dolci data from CSV, formatting with chat template.
    Expects 'question' and 'response' columns.
    """
    logging.info(f"Loading Dolci CSV data from: {csv_file_path}")
    
    # Load CSV
    # handle large fields if necessary, though pandas usually handles it ok.
    df = pd.read_csv(csv_file_path)
    
    # Check required columns exist
    if 'question' not in df.columns or 'response' not in df.columns:
        # Fallback for 'prompt'/'generated_text' if user mixed files, but expected is question/response
        if 'prompt' in df.columns and 'generated_text' in df.columns:
             logging.warning("Columns 'question'/'response' not found, using 'prompt'/'generated_text' mapping.")
             df['question'] = df['prompt']
             df['response'] = df['generated_text']
        else:
             raise ValueError(f"CSV must have 'question' and 'response' columns. Found: {df.columns.tolist()}")
    
    # Apply Chat Template
    def apply_template(row):

        if tokenizer.chat_template is not None:
            if "gemma" in model_path:
                messages = [
                    {"role": "user", "content": str(row['question'])},
                    {"role": "model", "content": str(row['response'])}
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": str(row['question'])},
                    {"role": "assistant", "content": str(row['response'])}
                ]
            
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            # Fallback for base models (e.g. Llama 2 base)
            prompt = ""
            # if system_prompt:
            #     prompt += f"{system_prompt}\n\n"
            prompt += f"You are a helpful assistant. Solve the following query.\n\nQuestion: {row['question']}\n\nAnswer: {str(row['response'])}"
            return prompt
        

    logging.info("Applying chat template to dataset...")
    df['text'] = df.apply(apply_template, axis=1)
    
    logging.info(f"Total samples in CSV: {len(df)}")
    
    # Split into train and test
    if test_split_ratio > 0:
        test_size = max(1, int(len(df) * test_split_ratio))
        test_set = df.sample(n=test_size, random_state=random_state)
        train_set = df.drop(test_set.index)
    else:
        train_set = df
        test_set = df.sample(n=min(100, len(df)), random_state=random_state)
    
    logging.info(f"Training samples: {len(train_set)}")
    logging.info(f"Test samples: {len(test_set)}")
    
    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_set[['text']].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_set[['text']].reset_index(drop=True))
    
    return train_dataset, test_dataset


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen3-4B")
    random_weights: bool = field(default=False)
    add_chat_template: bool = field(default=False)
    csv_file_path: Optional[str] = field(default=None)  # NEW: Path to CSV file
    dataset_type: str = field(default="triviaqa")  # NEW: Dataset type - gsm8k, triviaqa, or csv
    use_completion_only: bool = field(default=False)  # NEW: Use DataCollatorForCompletionOnlyLM
    # block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="test")
    wandb_entity: Optional[str] = field(default="ss13750-new-york-university-abu-dhabi")
    # train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    # dagger: bool = field(default=False)
    # LoRA parameters
    use_lora: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="all-linear")
    
    # QLoRA parameters
    use_qlora: bool = field(default=False)
    bnb_4bit_compute_dtype: str = field(default="bfloat16")
    bnb_4bit_quant_type: str = field(default="nf4")
    bnb_4bit_use_double_quant: bool = field(default=True)
    
    wandb_project: Optional[str] = field(default="test")
    wandb_entity: Optional[str] = field(default="ss13750-new-york-university-abu-dhabi")

    def __post_init__(self):
        # os.environ['HF_HOME'] = "/home/ss13750/.cache/huggingface"
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():        
    # parsing input
    parser = HfArgumentParser((TrainingConfig, SFTConfig))
    config, training_args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(training_args)}
    logging.info(f"Training config: {log_config}")


    model_path = config.model_name

    if not config.random_weights:
        logging.info("Initializing Pretrained Model...")
        model_name = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2")
    else:
        logging.info("Initializing Random Weights Model...")
        # Load the configuration but initialize with random weights
        config = AutoConfig.from_pretrained(model_path)

        # Fix the embedding configuration issues
        if hasattr(config, 'pad_token_id'):
            # Set pad_token_id to None to avoid the padding index issue
            config.pad_token_id = None

        # Ensure vocab_size is reasonable
        if not hasattr(config, 'vocab_size') or config.vocab_size <= 1:
            config.vocab_size = 50280  # OLMo's typical vocab size

        # Create model with fixed config
        with torch.no_grad():
            model_name = AutoModelForCausalLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side='left'

    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # data handling
    # Load dataset based on config
    if config.dataset_type == "csv":
        if config.csv_file_path is None:
            raise ValueError("csv_file_path must be provided when dataset_type='csv'")
        logging.info(f"Loading CSV dataset from: {config.csv_file_path}")
        train_dataset, test_dataset = csv_data_loader(config.csv_file_path)
    elif config.dataset_type == "gsm8k":
        logging.info("Loading GSM8K dataset")
        train_dataset, test_dataset = sft_data_loader(config, tokenizer, "gsm8k")
    elif config.dataset_type == "triviaqa":
        logging.info("Loading TriviaQA dataset")
        train_dataset, test_dataset = sft_data_loader(config, tokenizer, "triviaqa")
    elif config.dataset_type == "dolci":
        if config.csv_file_path is None:
            raise ValueError("csv_file_path must be provided when dataset_type='dolci'")
        logging.info(f"Loading Dolci dataset from: {config.csv_file_path}")
        train_dataset, test_dataset = dolci_data_loader(config.csv_file_path, tokenizer, model_path)
    else:
        raise ValueError(f"Unknown dataset_type: {config.dataset_type}. Choose from: csv, gsm8k, triviaqa, dolci")
    
    # Setup data collator for completion-only training if requested
    collator = None
    if config.use_completion_only:
        response_template_ids = None
        logging.info("Using DataCollatorForCompletionOnlyLM - loss only on assistant responses")
        # Auto-detect template based on model name
        if "Llama-3" in config.model_name:
            response_template = "<|start_header_id|>assistant<|end_header_id|>"
            logging.info("Detected Llama-3 model, using Llama-3 response template")
        elif "Qwen" in config.model_name:
            response_template = "<|im_start|>assistant"
            logging.info("Detected Qwen model, using ChatML response template")
        elif "Mistral" in config.model_name:
            response_template = "[/INST]"
            logging.info("Detected Mistral model, using ChatML response template")
        elif "gemma" in config.model_name:
            response_template = "<start_of_turn>model"
            logging.info("Detected Gemma model, using ChatML response template")
        else:
            # Fallback to Qwen/ChatML default or warn user
            response_template = None 
            response_template_ids = [13, 22550, 29901]
            logging.warning("Model type not detected from name, defaulting to Qwen/ChatML template: Answer: \\n")

        if response_template is not None: 
            response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        else:
            raise ValueError("Response template not found for model type: {}".format(config.model_name))
        logging.info(f"Response template: {response_template}")
        logging.info(f"Response template token IDs: {response_template_ids}")
        
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer,
        )
    else:
        logging.info("Using standard collator - loss on full sequence")

    # Setup QLoRA quantization if requested
    if config.use_qlora:
        logging.info(f"Using QLoRA with 4-bit quantization (type={config.bnb_4bit_quant_type}, double_quant={config.bnb_4bit_use_double_quant})")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
        
        # Load model with quantization
        logging.info(f"Loading model {config.model_name} with 4-bit quantization...")
        model_name = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        )
        
        # Force use_lora=True for QLoRA
        config.use_lora = True
        # Use higher rank for QLoRA if not explicitly set
        if config.lora_r == 16:  # default value
            config.lora_r = 64
            config.lora_alpha = 16
            logging.info(f"Auto-adjusted LoRA rank to {config.lora_r} for QLoRA")
    
    # Setup LoRA if requested (or forced by QLoRA)
    peft_config = None
    if config.use_lora:
        logging.info(f"Using LoRA with r={config.lora_r}, alpha={config.lora_alpha}, target={config.lora_target_modules}")
        
        target_modules = config.lora_target_modules
        if target_modules == "all-linear":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "," in target_modules:
            target_modules = [m.strip() for m in target_modules.split(",")]

        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    trainer = SFTTrainer(
        model=model_name,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator,
        peft_config=peft_config,
    )

    if training_args.resume_from_checkpoint == 'False':
        logging.info("Starting from base model...")
        trainer.train()
    elif training_args.resume_from_checkpoint == 'True':
        logging.info("Starting from last checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        logging.info(f"Starting from last checkpoint: {training_args.resume_from_checkpoint}...")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


# # ## Get a sample from your dataset
# sample = train_dataset[0]

# # Process it with your collator
# batch = collator([tokenizer.encode(sample['text'], truncation=True)])

# # Examine the labels
# input_ids = batch["input_ids"][0]
# labels = batch["labels"][0]

# # Print them side by side for comparison
# for i, (input_id, label) in enumerate(zip(input_ids, labels)):
#     token = tokenizer.decode([input_id])
#     print(f"Position {i}: Token '{token}' - Input ID: {input_id}, Label: {label}")

if __name__ == "__main__":
    train()


