import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel
import json
import numpy as np
import random
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.instructor_retrieval import get_embeddings
from datasets import load_dataset
from datasets import Dataset, load_dataset
from InstructorEmbedding import INSTRUCTOR
from typing import Union
import os.path as osp



class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("../templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


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

def read_dataset(path: str) -> Dataset:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        ds = load_dataset("json", data_files=path, split="train")
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ds = Dataset.from_list(data)
    else:
        raise ValueError("data_path must be .jsonl or .json")
    return ds

global_model = None
instruction = "Represent the sentence for similar task retrieval: "

def get_model_embeddings(config_path):
    """
    Initialize the vector database with configurations from the specified JSON file.
    """
    with open(config_path, 'r') as file:
        lora_configs = json.load(file)

    models = lora_configs
    global global_index, global_model

    # Load the embedding model for retrieval
    global_model = INSTRUCTOR('Styxxxx/lora_retriever')

    all_model_embeddings = []
    model_names = []

    # Compute average embeddings for each model
    for model in models:
        model_name = f"Styxxxx/llama2_7b_lora-{model['model_name']}"

        model_names.append(model_name)
        model_samples = []

        # Collect sample inputs for each model
        for sample in model['sample']:
            sample_context = sample['inputs']
            model_samples.append([instruction, sample_context])

        # Compute embeddings for the model's samples and take the mean
        embeddings = get_embeddings(model_samples)
        average_embedding = np.mean(embeddings, axis=0)
        all_model_embeddings.append(average_embedding)

    # Create a FAISS index with the collected embeddings
    all_model_embeddings = np.vstack(all_model_embeddings)

    return model_names, all_model_embeddings

def get_embeddings(text_list):
    """
    Encode a list of text samples using the global embedding model.

    Parameters:
    - text_list: A list of texts to be encoded. Each element should be [instruction, text].
    """
    return global_model.encode(text_list)

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

def train_val_split(ds: Dataset, val_ratio: float, seed: int):
    if val_ratio <= 0:
        return ds, None
    n = len(ds)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    val_n = max(1, int(n * val_ratio))
    val_idxs = idxs[:val_n]
    train_idxs = idxs[val_n:]
    return ds.select(train_idxs), ds.select(val_idxs)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRAuter(nn.Module):
    """
    Learns Wi, Wr and returns softmax weights over K adapters given a batch of inputs.
    """
    def __init__(self, A_init: torch.Tensor, top_k: int, temperature: float = 1.0):
        super().__init__()
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
        logits = I @ self.A.t()        # (B, K)

        if self.top_k is not None and 0 < self.top_k < logits.size(-1):
            # Build boolean mask for top-k indices per row s
            topk_vals, topk_idx = torch.topk(I@self.A.t(), self.top_k, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            
            mask.scatter_(1, topk_idx, True)
            masked_logits = logits.masked_fill(~mask, float('-inf'))
        else:
            masked_logits = logits
        
        probs = F.softmax(masked_logits / self.tau, dim=-1)
        return probs, logits

device = "cuda" if torch.cuda.is_available() else "cpu"

model_names, model_embeddings = get_model_embeddings('../config/config2.json')

scorer = LoRAuter(
    A_init=torch.tensor(model_embeddings, dtype=torch.float32),
    top_k=5,
    temperature=0.2
).to(device).bfloat16()

save_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(save_dir, exist_ok=True)

ckpt_path = os.path.join(save_dir, "base_model.pt")
cfg_path = os.path.join(save_dir, "base_model.config.json")

# Save state_dict on CPU to avoid device issues
state_cpu = {k: v.detach().cpu() for k, v in scorer.state_dict().items()}
torch.save(state_cpu, ckpt_path)

# Save minimal config to reconstruct the module later
scorer_config = {
    "top_k": int(scorer.top_k) if scorer.top_k is not None else None,
    "temperature": float(scorer.tau),
    "K": int(scorer.A.shape[0]),
    "dtype": "bfloat16",
}
with open(cfg_path, "w") as f:
    json.dump(scorer_config, f, indent=2)

print(f"Saved scorer to: {ckpt_path}")
print(f"Saved scorer config to: {cfg_path}")