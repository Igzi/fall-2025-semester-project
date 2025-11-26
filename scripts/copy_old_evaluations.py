import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel, get_peft_model, LoraConfig
import json
import argparse
import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.instructor_retrieval import perform_search, initialize_index
from datasets import load_dataset
from utils.prompter import Prompter
from utils.instructor_retrieval import perform_search, get_embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Generate outputs with LoRA adapters")
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()

args = parse_args()
device = args.device

correct_count = 0
model_size='7b'
config_path = './config/config2.json'
data_path = './dataset/config_large_flat.json'
res_path = './performance_large/outputs_hf_large/hf_adapter_outputs'
results = []  # Initialize a list to store question and response data

original_model_names = []
tasks = []
with open(config_path, 'r') as file:
    lora_configs = json.load(file)
    for model in lora_configs:
        original_model_names.append(f"Styxxxx/llama2_7b_lora-{model['model_name']}")
        tasks.append(model['model_name'])

# Load the dataset
if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    dataset = load_dataset("json", data_files=data_path)
else:
    dataset = load_dataset(data_path)

with open('./scripts/llama2_7b_adapters.json', 'r') as file:
    lora_adapters = json.load(file)

model_names = []
model_ranks = []
original_adapter_ids = []
original_adapter_names = []
for i, model in enumerate(lora_adapters):
    model_names.append(model['model_id'])
    model_ranks.append(model['rank'])
    if model['model_id'] in original_model_names:
        original_adapter_ids.append(i)
        original_adapter_names.append(model['model_id'])

print(original_model_names)
print(original_adapter_names)

for adapter_name, adapter_id in zip(original_adapter_names, original_adapter_ids):
    old_model_id = original_adapter_names.index(adapter_name)
    # copy performance_large/outputs/model_{old_model_id}.json to performance_large/outputs/outputs_hf_large/hf_adapter_outputs_{adapter_id}.json
    src_path = f'./performance_large/outputs/model_{old_model_id}.json'
    dst_path = f'./performance_large/outputs_hf_large/hf_adapter_outputs_{adapter_id}.json'
    print(f'Copying from {src_path} to {dst_path}')
    os.system(f'cp {src_path} {dst_path}')
