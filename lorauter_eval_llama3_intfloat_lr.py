from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel, get_peft_model, LoraConfig
import json
import numpy as np
from utils.instructor_retrieval import initialize_index
from datasets import load_dataset
from utils.prompter import Prompter
from utils.instructor_retrieval import get_embeddings
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Prompter is a utility class to create a prompt for a given input
prompter = Prompter("alpaca")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_results_matrix(path: str = "./adapter_evaluation/model_performance_13b.npy"):
    if not os.path.exists(path):
        # Not an error: just return None so callers can fall back to defaults
        print(f"[info] results matrix not found at: {path}")
        return None
    try:
        arr = np.load(path, allow_pickle=True)
        # Coerce to ndarray for consistent handling (may be an object array)
        arr = np.array(arr, dtype=np.float32)
        print(f"[info] loaded results matrix from {path} with shape {arr.shape}")
        return arr
    except Exception as e:
        print(f"[warning] failed to load results matrix {path}: {e}")
        return None

def load_base_model(model_name_or_path='meta-llama/Llama-3.1-8B'):
    """
    Load the base model and tokenizer from a given model path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16
    )
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
    lora_lists = []
    for i, lora_model in enumerate(lora_module_list):
        if i == 0:
            peft_model = PeftModel.from_pretrained(base_model, lora_model, f"adapter{i}")
        else:
            peft_model.load_adapter(lora_model, f"adapter{i}")
        lora_lists.append(f"adapter{i}")

    peft_model.set_adapter(lora_lists)
    peft_model = peft_model.to(torch.bfloat16)
    peft_model.eval()
    # Force all LoRA weights to bfloat16
    for n, p in peft_model.named_parameters():
        if "lora" in n:
            try:
                p.data = p.data.to(torch.bfloat16)
            except Exception:
                pass
    return peft_model

class LoRAuter(nn.Module):
    """
    Learns Wi, Wr and returns softmax weights over K adapters given a batch of inputs.
    """
    def __init__(self, A_init: torch.Tensor, results_matrix: np.ndarray, top_k: int, temperature: float = 1.0):
        super().__init__()
        self.register_buffer("A", A_init.clone())       # (K, d_a)
        self.results_matrix = results_matrix
        self.top_k = top_k
        self.tau = temperature

    @torch.no_grad()
    def set_adapter_embeddings(self, A_new: torch.Tensor):
        self.A = A_new.clone().to(self.A.device)

    def get_selected_adapter(self, old_idx: int, exclude_idx: int = None):
        return old_idx  # --- IGNORE --- For testing, just return the original index without re-selection --- IGNORE ---
        old_row = self.results_matrix[old_idx]
        mask_array = np.ones(self.results_matrix.shape[1], dtype=bool)
        if exclude_idx is not None:
            mask_array[exclude_idx] = False
        
        row = old_row*mask_array
        maxpos = np.flatnonzero(row==row.max())
        sel = int(maxpos[-1])
        return sel


    def forward(self, I: torch.Tensor, exclude_idx: int = None):
        """
        I: (B, d_in) input embeddings
        Returns:
          probs: (B, K) softmax weights per sample
          logits: (B, K)
        """
        I_norm = I / (I.norm(dim=-1, keepdim=True) + 1e-8)  # (B, d_in)
        A_norm = self.A / (self.A.norm(dim=-1, keepdim=True) + 1e-8)  # (K, d_a)
        logits = I_norm @ A_norm.t()  # (B, K) - cosine similarity in [-1, 1]

        if exclude_idx is not None:
            logits[:,exclude_idx] = -1.0

        if self.top_k is not None and 0 < self.top_k < logits.size(-1):
            # Build boolean mask for top-k indices per row s
            _, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, topk_idx, True)
            probs = torch.zeros_like(logits)
            probs[mask] = 1.0 / self.top_k
        else:
            # If top_k is not set, set all to uniform
            probs = torch.full_like(logits, 1.0 / logits.size(-1))
        return probs, None

def eval_datasets(
    data_path, 
    res_path, 
    config_path="config/config2.json", 
    lora_num=3, 
    batch_size=1, 
    ood=False, 
    best_selection=False, 
    model_size='7b',
    eval_type='mixture',
    val_embeddings_file: str = '/home/pavlovic/embeddings/validation_embeddings_e5_large.npy',
    embedding_model_name: str = 'intfloat/e5-large-v2',
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

    # Expose loaded matrix as a module-level variable; callers can check for None.
    results_matrix = load_results_matrix(f"./adapter_evaluation/model_performance_8b.npy")

    cfg_path = "./adapter_evaluation/models/base_model.config.json"
    ckpt_path = "./adapter_evaluation/models/base_model.pt"
     # Load scorer
    with open(cfg_path) as f:
        cfg = json.load(f)

        # If a validation embeddings file is provided, load and convert to torch tensor
    if val_embeddings_file is not None:
        val_embeddings_np = np.load(val_embeddings_file)
        val_embeddings = torch.from_numpy(val_embeddings_np.astype(np.float32)).to(device)
        try:
            val_embeddings = val_embeddings.to(dtype=torch.bfloat16 if cfg.get('dtype') == 'bfloat16' else torch.float32)
        except Exception:
            pass
    else:
        val_embeddings = torch.zeros(cfg['K'], 768, dtype=torch.float32).to(device)


    scorer = LoRAuter(
        A_init=torch.zeros(cfg["K"], 768, dtype=torch.bfloat16),  # placeholder, will be loaded
        results_matrix=results_matrix,
        top_k=lora_num,
        temperature=cfg["temperature"],
    ).to(device)
    if cfg["dtype"] == "bfloat16":
        scorer.bfloat16()

    scorer = LoRAuter(
        A_init=val_embeddings,
        results_matrix=results_matrix,
        top_k=lora_num,
        temperature=cfg["temperature"],
    ).to(device)
    if cfg["dtype"] == "bfloat16":
        scorer.bfloat16()

    state = torch.load(ckpt_path, map_location="cpu")
    state.pop('results_matrix', None)  # Remove if exists
    state.pop('top_k', None)  # Remove if exists
    state.pop('A', None)  # Remove if exists
    scorer.load_state_dict(state, strict=False)
    scorer.to(device).eval()

    # Initialize vector database for retrieval
    init_vector_db(config_path)

    embedding_model = SentenceTransformer(embedding_model_name)

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

    model_path = f"meta-llama/Llama-3.1-8B"
    base_model, tokenizer = load_base_model(model_path)
    base_model.eval()

    with open(config_path, 'r') as file:
        lora_configs = json.load(file)

    models = lora_configs
    model_names = []

    # Compute average embeddings for each model
    for model in models:
        model_name = f"igzi/lora-{model['model_name']}"

        model_names.append(model_name)

    peft_model = load_peft_model(model_names, base_model)
    peft_model = peft_model.to(device)
    peft_model.eval()
    # Cast all LoRA parameters to bfloat16 as requested
    for n, p in peft_model.named_parameters():
        if "lora" in n:
            try:
                p.data = p.data.to(torch.bfloat16)
            except Exception:
                pass
    model_dtype = next(peft_model.parameters()).dtype

    with torch.no_grad():
        with tqdm(total=50, desc="Evaluating", unit="item") as pbar:
            for i in range(1050, 1100, batch_size):
                input_text = eval_data["inputs"][i : i + batch_size]
                task_names = eval_data["task"][i : i + batch_size]

                # if eval_data["domain"][i] != "struct to text":
                #     continue

                # If out-of-domain filtering is required, specify exclusion list
                exclude_list = None
                if ood:
                    if model_size == '7b':
                        exclude_list = [f"igzi/lora-{task}" for task in task_names]
                    else:
                        exclude_list = [f"igzi/lora-{task}" for task in task_names]

                # Perform retrieval to get top-k LoRA modules
                I_batch = embedding_model.encode(input_text, convert_to_numpy=True)
                I_batch = torch.tensor(I_batch, dtype=torch.bfloat16).to(device) 

                # If best_selection is True, re-map module_list and mapping_matrix for a more constrained set
                if best_selection:
                    if model_size == '7b':
                        exclude_list = [f"igzi/lora-{task}" for task in task_names]
                    else:
                        exclude_list = [f"igzi/lora-{task}" for task in task_names]

                    unique_items = list(set(exclude_list))
                    module_list = unique_items

                if ood:
                    mapping_matrix_tensor, _ = scorer(I_batch, exclude_idx=model_names.index(f"igzi/lora-{task_names[0]}"))
                else:
                    mapping_matrix_tensor, _ = scorer(I_batch)
                # mapping_matrix_tensor = torch.ones_like(mapping_matrix_tensor)  # --- IGNORE --- Use uniform weights instead of scorer output for testing --- IGNORE ---
                input_text = eval_data["full_prompt"][i : i + batch_size]

                if ood:
                    mapping_matrix_tensor[0, model_names.index(f"igzi/lora-{task_names[0]}")] = 0.0
                    mapping_matrix_tensor = mapping_matrix_tensor / mapping_matrix_tensor.sum(-1, keepdim=True)

                # Tokenize the input text
                inputs = tokenizer(
                    input_text,
                    max_length=512,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                # Keep routing weights dtype/device aligned with model compute to avoid einsum dtype errors.
                mapping_matrix_tensor = mapping_matrix_tensor.to(
                    device=inputs["input_ids"].device,
                    dtype=model_dtype,
                )

                #print(mapping_matrix_tensor)

                outputs = peft_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=50,
                    temperature=0.001,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                    eos_token_id=tokenizer.eos_token_id,
                    merging_type=eval_type,
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
