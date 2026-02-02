import os
import copy
import json
import shutil
import yaml
import importlib

from utils.merge_yaml_generator import generate_file
from utils.merge import merge_model, validate_config
from utils.evaluate import eval_math
from utils.prompts import QWEN_INSTRUCT

from transformers import AutoConfig, AutoModel

# --- Inputs ---
# Load configuration from YAML file
def load_config(config_path="merge_config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load inputs from YAML
input_config = load_config()

print(input_config)

layers_to_remove = input_config['layers_to_remove']
layers_to_remove = sorted(layers_to_remove)
model_name = input_config['model_name']

# model = AutoModel.from_pretrained(model_name)

config = AutoConfig.from_pretrained(model_name)
total_layers = config.num_hidden_layers


def merge(model_name, layers_to_remove, output_path="./merged"):
    generate_file(model_name, layers_to_remove)
    merge_model(output_path=output_path)
    validate_config(output_path=output_path)
    print("After ablation:", AutoConfig.from_pretrained(output_path).num_hidden_layers, "layers")

out_dir = "./out_models/" + model_name.replace("/", "_").replace(".", "_") + "__" + "-".join([str(i) for i in layers_to_remove])

merge(model_name, layers_to_remove, output_path=out_dir)

print("""
ALL DONE!!!

""")


