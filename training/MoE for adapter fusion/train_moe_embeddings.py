import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel
import json
import numpy as np
import random
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from utils.instructor_retrieval import perform_search, get_embeddings
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
        file_name = osp.join("../../templates", f"{template_name}.json")
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
        logits = proj_I @ proj_A.t()        # (B, K)

        if self.top_k is not None and 0 < self.top_k < logits.size(-1):
            # Build boolean mask for top-k indices per row
            topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, topk_idx, True)
            masked_logits = logits.masked_fill(~mask, float('-inf'))
        else:
            masked_logits = logits
        
        probs = F.softmax(masked_logits / self.tau, dim=-1)
        return probs, logits
    
from typing import List

def activate_fused_adapter_from_embeddings(
    peft_model,                      # a PEFT-wrapped model (e.g., PeftModelForCausalLM)
    scorer: BilinearFusionScorer,    # the bilinear scorer above (Wi, Wr)
    I_batch: torch.Tensor,           # (B, d_in) input embeddings for the current batch
    adapter_names: List[str],        # list of K adapter names already present in the model
    fusion_name: str = "fusion_adapter",
    combination_type: str = "linear" # "linear" or "average" depending on your PEFT version
):
    """
    Computes weights with the scorer and builds a single fused adapter in the model,
    then activates it via set_adapter(...).

    Note: this is a *static* fusion for the whole model for subsequent forward passes.
    """
    # 1) per-sample weights (B, K)
    probs, _ = scorer(I_batch)

    # 2) collapse to a single weight vector over experts.
    #    Two reasonable choices:
    #      - mean over the batch (default)
    #      - max or top-k then renormalize
    w = probs.mean(dim=0)                    # (K,)
    w = (w / w.sum()).tolist()

    # 3) Create a fused adapter in PEFT and activate it.
    #    add_weighted_adapter will materialize the fused delta weights across all LoRA layers.
    #    combination_type="linear" uses the exact weights; "average" divides by K internally.
    peft_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=w,
        adapter_name=fusion_name,
        combination_type=combination_type
    )
    peft_model.set_adapter(fusion_name)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_names, model_embeddings = get_model_embeddings('../../config/config2.json')

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

dataset = read_dataset('./data.jsonl')
train_ds, val_ds = train_val_split(dataset, 0.01, 42)
train_ds_prompt = train_ds.map(generate_and_tokenize_prompt)
val_ds_prompt = val_ds.map(generate_and_tokenize_prompt)

model_path = f"meta-llama/Llama-2-7b-hf"
base_model, tokenizer = load_base_model(model_path)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
base_model = base_model.to(device)
base_model.eval()

peft_model = load_peft_model(model_names, base_model)
peft_model = peft_model.to(device)
peft_model.eval()

adapter_names = [f"adapter{i}" for i in range(len(model_names))]

for param in peft_model.parameters():
    param.requires_grad = False  # freeze everything

scorer = BilinearFusionScorer(
    d_in=model_embeddings.shape[1],
    d_a=model_embeddings.shape[1],
    d_proj=256,
    A_init=torch.tensor(model_embeddings, dtype=torch.float32),
    top_k=5,
    temperature=1.0
).to(device)

grad_accum_steps = 5
opt = torch.optim.AdamW(scorer.parameters(), lr=5e-5)
opt.zero_grad()
running_loss = 0.0
eval_steps = 100
val_log = []

from contextlib import suppress

with tqdm(total=len(train_ds), desc="Training", unit="sample") as pbar:
    for i in range(len(train_ds)):
        I_batch = get_embeddings([train_ds[i]['inputs']])
        I_batch = torch.tensor(I_batch, dtype=torch.float32).to(device)

        # (Optional) remove previous fusion adapter to avoid accumulating many adapters
        with suppress(Exception):
            peft_model.disable_adapter()
            peft_model.delete_adapter("fusion_adapter")

        activate_fused_adapter_from_embeddings(
            peft_model=peft_model,
            scorer=scorer,
            I_batch=I_batch,
            adapter_names=adapter_names,
            fusion_name="fusion_adapter",
            combination_type="linear",
        )

        batch = train_ds_prompt[i]
        batch = tokenizer(
            batch["full_prompt"],
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100

        outputs = peft_model(**batch)
        loss = outputs.loss / grad_accum_steps  # scale for accumulation
        loss.backward()
        running_loss += loss.item() * grad_accum_steps
        del batch, outputs, I_batch
        torch.cuda.empty_cache()

        if (i + 1) % grad_accum_steps == 0:
            before = {n: p.detach().clone() for n,p in scorer.named_parameters()}

            opt.step()

            for n,p in scorer.named_parameters():
                diff = (p.detach() - before[n]).abs().sum().item()
                print(f"{n:25s} | abs_sum_diff={diff:.6e}")
            opt.zero_grad()
            avg_loss = running_loss / grad_accum_steps
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
            running_loss = 0.0

        if (i + 1) % eval_steps == 0 or (i + 1) == len(train_ds):
            peft_model.eval()
            val_losses = []
            with torch.no_grad():
                for j in range(len(val_ds)):
                    # Re-build fusion adapter for each val sample
                    with suppress(Exception):
                        peft_model.disable_adapter()
                        peft_model.delete_adapter("fusion_adapter")

                    I_val = get_embeddings([val_ds[j]['inputs']])
                    I_val = torch.tensor(I_val, dtype=torch.float32).to(device)

                    activate_fused_adapter_from_embeddings(
                        peft_model=peft_model,
                        scorer=scorer,
                        I_batch=I_val,
                        adapter_names=adapter_names,
                        fusion_name="fusion_adapter",
                        combination_type="linear",
                    )

                    val_item = val_ds_prompt[j]
                    val_batch = tokenizer(
                        val_item["full_prompt"],
                        padding=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(device)
                    val_batch["labels"] = val_batch["input_ids"].clone()
                    val_batch["labels"][val_batch["attention_mask"] == 0] = -100

                    val_out = peft_model(**val_batch)
                    val_losses.append(val_out.loss.item())
                    del val_batch, val_out, I_val
                    torch.cuda.empty_cache()

            mean_val_loss = float(sum(val_losses) / max(1, len(val_losses)))
            val_log.append((i + 1, mean_val_loss))
            # Keep last displayed train loss (compute a current one if not just updated)
            if 'avg_loss' not in locals():
                current_train_loss = running_loss / max(1, ((i + 1) % grad_accum_steps))
            else:
                current_train_loss = avg_loss
            pbar.set_postfix(loss=f"{current_train_loss:.4f}", val_loss=f"{mean_val_loss:.4f}")
            print(f"[Step {i+1}] Validation loss: {mean_val_loss:.4f}")
            peft_model.train()

        pbar.update(1)

