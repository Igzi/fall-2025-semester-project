import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

# Prompter is a utility class to create a prompt for a given input
prompter = Prompter("alpaca")

def load_base_model(model_name_or_path='meta-llama/Llama-3.1-8B'):
    """
    Load the base model and tokenizer from a given model path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    base_model.bfloat16()
    return base_model, tokenizer

def load_peft_model(lora_module_list, base_model):
    """
    Load and configure PEFT (Parameter-Efficient Fine-Tuning) adapters onto the base model.
    """
    lora_lists = []
    for i, lora_model in enumerate(lora_module_list):
        print(i, lora_model)
        if i == 0:
            peft_model = PeftModel.from_pretrained(base_model, lora_model, f"adapter{i}")
        else:
            peft_model.load_adapter(lora_model, f"adapter{i}")
        lora_lists.append(f"adapter{i}")

    peft_model.set_adapter(lora_lists)
    peft_model = peft_model.to(device)
    peft_model.eval()
    return peft_model

def parse_args():
    parser = argparse.ArgumentParser(description="Generate outputs with LoRA adapters")
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B", type=str)
    parser.add_argument("--adapter_file", default="./scripts/llama3_1_8b_adapters_rank64.json", type=str)
    parser.add_argument("--performance_file", default="./performance_large/model_hf_performance.npy", type=str)
    parser.add_argument("--output_prefix", default="./performance_large/outputs_hf_large_llama3/hf_adapter_outputs", type=str)
    return parser.parse_args()

args = parse_args()
device = args.device

correct_count = 0
config_path = './config/config2.json'
data_path = './dataset/config2_flat.json'
res_path = args.output_prefix
results = []  # Initialize a list to store question and response data

original_model_names = []
tasks = []
with open(config_path, 'r') as file:
    lora_configs = json.load(file)
    for model in lora_configs:
        original_model_names.append(f"Styxxxx/llama2_7b_lora-{model['model_name']}")
        tasks.append(model['model_name'])

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

model_path = args.base_model
base_model, tokenizer = load_base_model(model_path)
base_model.eval()

with open(args.adapter_file, 'r') as file:
    lora_adapters = json.load(file)

model_names = []
model_ranks = []
original_adapter_ids = []
for i, model in enumerate(lora_adapters):
    model_names.append(model['model_id'])
    model_ranks.append(model['rank'])
    if model['model_id'] in original_model_names:
        original_adapter_ids.append(i)

model_id = args.model_id

# Load the model performance matrix
# model_hf_performance = np.load(args.performance_file, allow_pickle=True)
# model_hf_performance[model_hf_performance == None] = 0.0
# model_hf_performance = model_hf_performance.astype(np.float32)
# for original_model_id in original_adapter_ids:
#     model_hf_performance[:, original_model_id] = -1.0

# selected_adapters = [[]]*model_hf_performance.shape[0]
# model_id_selected = False
# for i in range(model_hf_performance.shape[0]):
#     selected_adapters[i] = list(np.argsort(-model_hf_performance[i])[:50])
#     if model_id in selected_adapters[i]:
#         model_id_selected = True

# if not model_id_selected:
if model_id >= len(model_names) or model_id < 0:
    #print(f"Model ID {model_id} is not selected in any of the tasks. Exiting gracefully.")
    print(f"Model ID {model_id} is out of range. Exiting gracefully.")
    # Free up any GPU memory that might be allocated
    torch.cuda.empty_cache()
    sys.exit(0)

if model_names[model_id].startswith("igzi"):
    print(f"Model ID {model_id} is an original model. Exiting gracefully.")
    torch.cuda.empty_cache()
    sys.exit(0)

peft_model = load_peft_model([model_names[model_id]], base_model)
peft_model = peft_model.to(device)
peft_model.eval()

results = []
with tqdm(total=len(dataset["train"]), desc="Evaluating", unit="item") as pbar:
    for pos in range(0, len(eval_data["full_prompt"]), 10):
        task_id = tasks.index(eval_data["model_name"][pos])
        if eval_data["model_name"][pos] != "arc_easy":
            pbar.update(10)
            continue
        # if model_id not in selected_adapters[task_id]:
        #     pbar.update(10)
        #     # Skip the next 10 samples since they are in the same task
        #     continue

        for i in range(pos, min(pos+10, len(eval_data["full_prompt"]))):
            mapping_matrix_tensor = torch.ones((1,1), device=device)
            input_text = [eval_data["full_prompt"][i]]

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

            targets = [eval_data["targets"][i]]

            # Process and store results
            for j, (output, expected_answer) in enumerate(zip(outputs, targets)):
                generated_answer = tokenizer.decode(output, skip_special_tokens=True)
                generated_answer = generated_answer.strip().split('### Response:\n')[-1]

                sample = {
                    'inputs': eval_data["inputs"][i],
                    'targets': eval_data["targets"][i],
                    'metric': eval_data["metric"][i],
                    'domain': eval_data["domain"][i],
                    'model_name': eval_data["model_name"][i],
                    'model': model_names[model_id],
                    'predicted_answer': generated_answer
                }
                results.append(sample)
            
            pbar.update(len(input_text))

# Save the results to a JSON file
os.makedirs(os.path.dirname(f"{res_path}"), exist_ok=True)
with open(f"{res_path}_{model_id}.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
