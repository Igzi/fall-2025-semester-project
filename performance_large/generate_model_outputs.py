import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel, get_peft_model, LoraConfig
import json
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

# Prompter is a utility class to create a prompt for a given input
prompter = Prompter("alpaca")

def load_base_model(model_name_or_path='meta-llama/Llama-2-7b-hf'):
    """
    Load the base model and tokenizer from a given model path.
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    base_model = LlamaForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16
    )
    base_model.bfloat16()
    return base_model, tokenizer

def init_vector_db(config_path):
    """
    Initialize the vector database with configurations from the specified JSON file.
    """
    model_names = []
    with open(config_path, 'r') as file:
        lora_configs = json.load(file)

    initialize_index(lora_configs)

def load_peft_model(lora_module_list, base_model):
    """
    Load and configure PEFT (Parameter-Efficient Fine-Tuning) adapters onto the base model.
    """
    lora_lists = []
    for i, lora_model in enumerate(lora_module_list):
        if i == 0:
            peft_model = PeftModel.from_pretrained(base_model, lora_model, f"adapter{i}")
        else:
            peft_model.load_adapter(lora_model, f"adapter{i}")
        lora_lists.append(f"adapter{i}")

    peft_model.set_adapter(lora_lists)
    peft_model = peft_model.to(device)
    peft_model.eval()
    return peft_model

correct_count = 0
model_size='13b'
batch_size = 1
config_path = './config/config_large.json'
data_path = './dataset/config_large_flat.json'
res_path = './performance_large/outputs_13b/model_'
results = []  # Initialize a list to store question and response data
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Initialize vector database for retrieval
init_vector_db('./config/config_large.json')

def generate_and_tokenize_prompt(data_point):
    """
    Generate the full prompt for a given data point and return it.
    """
    full_prompt = prompter.generate_prompt(
        data_point["inputs"],
        "",
        "",
    )
    return {"full_prompt": full_prompt}

# Load the dataset
if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    dataset = load_dataset("json", data_files=data_path)
else:
    dataset = load_dataset(data_path)

# Prepare the dataset with full prompts
eval_data = dataset["train"].map(generate_and_tokenize_prompt)

model_path = f"meta-llama/Llama-2-{model_size}-hf"
base_model, tokenizer = load_base_model(model_path)
base_model.eval()

with open(config_path, 'r') as file:
    lora_configs = json.load(file)

models = lora_configs
model_names = []

# Compute average embeddings for each model
for model in models:
    model_name = f"Styxxxx/llama2_{model_size}_lora-{model['model_name']}"

    model_names.append(model_name)

peft_model = load_peft_model(model_names, base_model)
peft_model = peft_model.to(device)
peft_model.eval()

with torch.no_grad():
    for model_id in range(48):
        print(f"Evaluating model {model_names[model_id]} ({model_id})")
        results = []
        with tqdm(total=len(dataset["train"]), desc="Evaluating", unit="item") as pbar:
            for i in range(0, len(eval_data["full_prompt"]), batch_size):
                input_text = eval_data["inputs"][i : i + batch_size]

                # If out-of-domain filtering is required, specify exclusion list
                exclude_list = None

                mapping_matrix_tensor = torch.zeros((batch_size, len(model_names)), dtype=torch.bfloat16).to(device)
                mapping_matrix_tensor[:, model_id] = 1.0
                input_text = eval_data["full_prompt"][i : i + batch_size]

                # Tokenize the input text
                inputs = tokenizer(
                    input_text,
                    max_length=512,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                outputs = peft_model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=50,
                    temperature=0.001,
                    merging_type='mixture',
                    lora_mapping=mapping_matrix_tensor
                )

                # Process and store results
                for j, (output, expected_answer) in enumerate(zip(outputs, eval_data["targets"][i : i + batch_size])):
                    generated_answer = tokenizer.decode(output, skip_special_tokens=True)
                    generated_answer = generated_answer.strip().split('### Response:\n')[-1]

                    sample = {
                        'inputs': eval_data["inputs"][i+j],
                        'targets': eval_data["targets"][i+j],
                        'metric': eval_data["metric"][i+j],
                        'domain': eval_data["domain"][i+j],
                        'model_name': eval_data["model_name"][i+j],
                        'predicted_answer': generated_answer
                    }
                    results.append(sample)
                
                pbar.update(len(input_text))
        
        # Save the results to a JSON file
        os.makedirs(os.path.dirname(f"{res_path}"), exist_ok=True)
        with open(f"{res_path}_test.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
