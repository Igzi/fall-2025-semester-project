import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel, get_peft_model, LoraConfig
import json
import numpy as np
from utils.instructor_retrieval import perform_search, initialize_index
from datasets import load_dataset
from utils.prompter import Prompter
from utils.instructor_retrieval import perform_search, get_embeddings
import os
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

def load_peft_model(lora_module_list, base_model):
    """
    Load and configure PEFT (Parameter-Efficient Fine-Tuning) adapters onto the base model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

class BilinearFusionScorer(nn.Module):
    """
    Learns Wi, Wr and returns softmax weights over K adapters given a batch of inputs.
    """
    def __init__(self, d_in: int, d_a: int, d_proj: int, A_init: torch.Tensor, top_k: int, temperature: float = 1.0):
        super().__init__()
        self.Wi = nn.Linear(d_in, d_proj, bias=False)   # I -> d_proj
        self.Wr = nn.Linear(d_a, d_proj, bias=False)    # A -> d_proj
        self.register_buffer("A", A_init.clone())       # (K, d_a)
        self.top_k = top_k
        self.tau = temperature

    @torch.no_grad()
    def set_adapter_embeddings(self, A_new: torch.Tensor):
        self.A = A_new.clone().to(self.A.device)

    def forward(self, I: torch.Tensor):
        """
        I: (B, d_in) input embeddings
        Returns:
          probs: (B, K) softmax weights per sample
          logits: (B, K)
        """
        proj_I = self.Wi(I)                 # (B, d_proj)
        proj_A = self.Wr(self.A)            # (K, d_proj)
        logits = I @ self.A.t()        # (B, K)

        if self.top_k is not None and 0 < self.top_k < logits.size(-1):
            # Build boolean mask for top-k indices per row s
            topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            
            mask.scatter_(1, topk_idx, True)
            masked_logits = logits.masked_fill(~mask, float('-inf'))
            # masked_logits = torch.full_like(logits, float('-inf'))
            # masked_logits.masked_fill_(mask, 1.0)
        else:
            masked_logits = logits
        
        probs = F.softmax(masked_logits / self.tau, dim=-1)
        return probs, logits

def eval_datasets(
    data_path, 
    res_path, 
    config_path="config/config2.json", 
    eval_type="fusion", 
    lora_num=3, 
    batch_size=1, 
    ood=False, 
    best_selection=False, 
    model_size='7b'
):
    """
    Evaluate the model on given datasets.

    Parameters:
    - data_path: Path to the evaluation dataset.
    - res_path: Path to save the evaluation results.
    - config_path: Path to configuration file for vector DB initialization.
    - eval_type: The merging type for LoRA adapters (e.g., 'fusion').
    - lora_num: Number of LoRA adapters to be retrieved.
    - batch_size: Batch size for evaluation.
    - ood: Flag indicating if out-of-domain exclusion should be applied.
    - best_selection: If True, use the most appropriate LoRA for each input.
    - model_size: Model size of Llama-2.
    """
    correct_count = 0
    results = []  # Initialize a list to store question and response data
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path = "./training/MoE_for_adapter_fusion/checkpoints/scorer_final.config.json"
    ckpt_path = "./training/MoE_for_adapter_fusion/checkpoints/scorer_final.pt"
     # Load scorer
    with open(cfg_path) as f:
        cfg = json.load(f)

    scorer = BilinearFusionScorer(
        d_in=cfg["d_in"],
        d_a=cfg["d_a"],
        d_proj=cfg["d_proj"],
        A_init=torch.zeros(cfg["K"], cfg["d_a"], dtype=torch.bfloat16),  # placeholder, will be loaded
        top_k=cfg["top_k"],
        temperature=cfg["temperature"],
    ).to(device)
    if cfg["dtype"] == "bfloat16":
        scorer.bfloat16()

    state = torch.load(ckpt_path, map_location="cpu")
    scorer.load_state_dict(state)
    scorer.to(device).eval()

    # Initialize vector database for retrieval
    init_vector_db(config_path)

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

    model_path = f"meta-llama/Llama-2-7b-hf"
    base_model, tokenizer = load_base_model(model_path)

    with open(config_path, 'r') as file:
        lora_configs = json.load(file)

    models = lora_configs
    model_names = []

    # Compute average embeddings for each model
    for model in models:
        model_name = f"Styxxxx/llama2_7b_lora-{model['model_name']}"

        model_names.append(model_name)

    
    peft_model = load_peft_model(model_names, base_model)
    peft_model = peft_model.to(device)
    peft_model.eval()

    with torch.no_grad():
        with tqdm(total=len(dataset["train"]), desc="Evaluating", unit="item") as pbar:
            for i in range(0, len(eval_data["full_prompt"]), batch_size):
                input_text = eval_data["inputs"][i : i + batch_size]
                task_names = eval_data["task"][i : i + batch_size]

                if eval_data["domain"][i] != "struct to text":
                    continue

                # If out-of-domain filtering is required, specify exclusion list
                exclude_list = None
                if ood:
                    if model_size == '7b':
                        exclude_list = [f"Styxxxx/llama2_7b_lora-{task}" for task in task_names]
                    else:
                        exclude_list = [f"Styxxxx/llama2_13b_lora-{task}" for task in task_names]

                # Perform retrieval to get top-k LoRA modules
                I_batch = get_embeddings([input_text[0]])
                I_batch = torch.tensor(I_batch, dtype=torch.bfloat16).to(device) 

                # If best_selection is True, re-map module_list and mapping_matrix for a more constrained set
                if best_selection:
                    if model_size == '7b':
                        exclude_list = [f"Styxxxx/llama2_7b_lora-{task}" for task in task_names]
                    else:
                        exclude_list = [f"Styxxxx/llama2_13b_lora-{task}" for task in task_names]

                    unique_items = list(set(exclude_list))
                    module_list = unique_items

                mapping_matrix_tensor, _ = scorer(I_batch)
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
                    merging_type='fusion',
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
                        'task': eval_data["task"][i+j],
                        'predicted_answer': generated_answer
                    }
                    results.append(sample)

                pbar.update(len(input_text))

    # Save the results to a JSON file
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import fire
    fire.Fire(eval_datasets)
