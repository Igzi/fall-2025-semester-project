import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel
import json
import numpy as np
from utils.instructor_retrieval import perform_search, initialize_index
from datasets import load_dataset
from utils.prompter import Prompter
from utils.merge_adapters import load_adapter_to_memory
from utils.merge_adapters import merge_adapters_fusion
import os

# Prompter is a utility class to create a prompt for a given input
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

# Global dictionary to store adapters in memory
adapters = {}

# Load the base model
model_size='7b'
model_path = f"meta-llama/Llama-2-{model_size}-hf"
base_model, tokenizer = load_base_model(model_path)
base_model.eval()

# Load adapter into memory
adapter_name = "Styxxxx/llama2_7b_lora-aeslc"
adapter_info = load_adapter_to_memory(adapters, adapter_name)

# Print what's stored in memory
print(f"\nAdapters cached: {list(adapters.keys())}")
print(f"Adapter info keys: {list(adapter_info.keys())}")

# You can also load multiple adapters
additional_adapters = [
    "Styxxxx/llama2_7b_lora-common_gen",
    "Styxxxx/llama2_7b_lora-dart"
]

for adapter in additional_adapters:
    load_adapter_to_memory(adapters, adapter)

# Access any adapter from cache
adapter = adapters.get("Styxxxx/llama2_7b_lora-aeslc")

peft_model = merge_adapters_fusion(additional_adapters, adapters)