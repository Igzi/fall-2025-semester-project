import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import get_peft_model
import json
import numpy as np
from utils.instructor_retrieval import initialize_index, perform_search_by_group
from utils.prompter import Prompter
import random
import os

prompter = Prompter("alpaca")

def load_base_model(model_name_or_path='meta-llama/Llama-2-7b-hf'):
    """
    Load the base model and tokenizer from a given model path.
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    base_model = LlamaForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16
    )
    base_model.bfloat16()
    return base_model, tokenizer

def init_vector_db(config_path='./config/config2.json'):
    """
    Initialize the vector database with configurations from the specified JSON file.
    """
    model_names = []
    with open(config_path, 'r') as file:
        lora_configs = json.load(file)

    initialize_index(lora_configs)

def load_peft_model(lora_module_list, base_model, lora_cfg):
    """
    Load and configure PEFT (Parameter-Efficient Fine-Tuning) adapters onto the base model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_lists = []
    peft_model = get_peft_model(base_model, lora_cfg)
    for i, lora_model in enumerate(lora_module_list):
        peft_model.load_adapter(lora_model, f"adapter{i}")
        lora_lists.append(f"adapter{i}")

    peft_model.set_adapter(lora_lists)
    peft_model = peft_model.to(device)
    peft_model.eval()

    return peft_model

def get_fused_peft_model(base_model, train_data, exclude_list, lora_cfg):
    """
    Load multiple LoRA adapters and fuse them into a single model.
    """
    init_vector_db(config_path='./config/config2.json')

    random.seed(42)
    sample_size = min(20, len(train_data))  # Handle case where train_data has less than 20 items
    random_samples = random.sample(list(train_data), sample_size)
    inputs = [data['inputs'] for data in random_samples]

    module_list, _ = perform_search_by_group(inputs, k=3, exclude_list=exclude_list)
    peft_model = load_peft_model(module_list, base_model, lora_cfg)

    adapters = [f"adapter{i}" for i in range(len(module_list))]
    weights = torch.ones(len(module_list)) / len(module_list)
    adapter_name = "fusion_adapter"
    peft_model.add_weighted_adapter(adapters, weights, adapter_name, combination_type='linear')
    peft_model.set_adapter("fusion_adapter")

    return peft_model


