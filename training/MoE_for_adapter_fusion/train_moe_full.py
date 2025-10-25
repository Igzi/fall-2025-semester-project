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

class BilinearFusionScorer(nn.Module):
    """
    Learns Wi, Wr and returns softmax weights over K adapters given a batch of inputs.
    """
    def __init__(self, d_in: int, d_a: int, d_proj: int, A_init: torch.Tensor, top_k: int, temperature: float = 1.0):
        super().__init__()
        self.Wi = nn.Linear(d_in, d_proj, bias=False)   # I -> d_proj
        # Replace single linear with 2-layer MLP
        self.Wr = nn.Sequential(
            nn.Linear(d_a, d_proj, bias=False),        # A -> d_proj (hidden)
            nn.ReLU(),
            nn.Linear(d_proj, d_proj, bias=False),      # d_proj -> d_proj (output)
            nn.ReLU(),
            nn.Linear(d_proj, d_proj, bias=False)      # d_proj -> d_proj (output)
        )
        self.register_buffer("A", A_init.clone())       # (K, d_a)
        self.top_k = top_k
        self.tau = temperature

    @torch.no_grad()
    def set_adapter_embeddings(self, A_new: torch.Tensor):
        self.A = A_new.clone().to(self.A.device)

    def forward(self, I: torch.Tensor, train: bool = False):
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
            # Build boolean mask for top-k indices per row s
            topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            
            mask.scatter_(1, topk_idx, True)
            masked_logits = logits.masked_fill(~mask, float('-inf'))
        else:
            masked_logits = logits
        
        probs = F.softmax(masked_logits / self.tau, dim=-1)
        return probs, logits

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

def calculate_em(references, candidates):
    references = [ref.split("\n\n")[0] for ref in references]
    em_scores = [1 if cal_correct(ref, cand) else 0 for ref, cand in zip(references, candidates)]
    return np.round(np.mean(em_scores) * 100, 1) if em_scores else 0

def cal_correct(generated_answer, expected_answer):
    is_correct = generated_answer.strip().lower().replace(".", "") == expected_answer.strip().lower().replace(".", "")
    return is_correct

dataset = read_dataset('./data.jsonl')
train_ds, val_ds = train_val_split(dataset, 0.05, 42)
train_ds_prompt = train_ds.map(generate_and_tokenize_prompt)
val_ds_prompt = val_ds.map(generate_and_tokenize_prompt)

model_path = f"meta-llama/Llama-2-7b-hf"
base_model, tokenizer = load_base_model(model_path)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
base_model = base_model.to(device)
base_model.eval()
base_model.config.use_cache = False          # <-- important
base_model.gradient_checkpointing_enable()

peft_model = load_peft_model(model_names, base_model)
peft_model = peft_model.to(device)
peft_model.eval()

for param in peft_model.parameters():
    param.requires_grad = False  # freeze everything

scorers = [BilinearFusionScorer(
    d_in=4096,
    d_a=model_embeddings.shape[1],
    d_proj=128,
    A_init=torch.tensor(model_embeddings, dtype=torch.float32),
    top_k=5,
    temperature=0.2
).to(device) for _ in range(32)]

for scorer in scorers:
    scorer.bfloat16()

# save_dir = "../../training/MoE_for_adapter_fusion/checkpoints"
# scorers = []

# # Find all scorer checkpoint files
# scorer_files = [f for f in os.listdir(save_dir) if f.startswith("scorer_layer_") and f.endswith("_mixture_mlp.pt")]
# scorer_files.sort(key=lambda x: int(x.split("_")[2]))  # Sort by layer index

# for scorer_file in scorer_files:
#     # Extract layer index from filename
#     layer_idx = int(scorer_file.split("_")[2])
    
#     ckpt_path = os.path.join(save_dir, f"scorer_layer_{layer_idx}_mixture_mlp.pt")
#     cfg_path = os.path.join(save_dir, f"scorer_layer_{layer_idx}_mixture_mlp.config.json")
    
#     # Load config
#     with open(cfg_path) as f:
#         cfg = json.load(f)
    
#     # Create scorer with MLP structure
#     scorer = BilinearFusionScorer(
#         d_in=cfg["d_in"],
#         d_a=cfg["d_a"],
#         d_proj=cfg["d_proj"],
#         A_init=torch.zeros(cfg["K"], cfg["d_a"], dtype=torch.bfloat16),  # placeholder, will be loaded
#         top_k=5,
#         temperature=cfg["temperature"],
#     ).to(device)
    
#     if cfg["dtype"] == "bfloat16":
#         scorer.bfloat16()
    
#     # Load state dict
#     state = torch.load(ckpt_path, map_location="cpu")
#     scorer.load_state_dict(state)
#     scorer.to(device).eval()
    
#     scorers.append(scorer)

grad_accum_steps = 32
all_scorer_params = []
for scorer in scorers:
    all_scorer_params.extend(list(scorer.parameters()))

opt = torch.optim.AdamW(all_scorer_params, lr=1e-4)
opt.zero_grad()
running_loss = 0.0
eval_steps = 200
val_log = []

with tqdm(total=len(train_ds)//2, desc="Training", unit="sample") as pbar:
    for i in range(len(train_ds)//2):
        I_batch = get_embeddings([train_ds[i]['inputs']])
        I_batch = torch.tensor(I_batch, dtype=torch.bfloat16).to(device)

        dp = train_ds_prompt[i]
        batch = tokenizer(
            dp["full_prompt"]+dp["targets"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        prefix_tok = tokenizer(
            dp["full_prompt"],
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )

        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100
        batch["labels"][:, :prefix_tok["input_ids"].size(0)] = -100  # only compute loss on the target
        
        outputs = peft_model(**batch, merging_type='moe', scorers=scorers)
        loss = outputs.loss / grad_accum_steps  # scale for accumulation
        loss.backward()
        running_loss += loss.item() * grad_accum_steps
        del batch, outputs, I_batch
        torch.cuda.empty_cache()

        if (i + 1) % grad_accum_steps == 0:
            opt.step()
            opt.zero_grad()
            avg_loss = running_loss / grad_accum_steps
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
            running_loss = 0.0

        if i==0 or (i + 1) % eval_steps == 0 or (i + 1) == len(train_ds):
            scorer.eval()
            val_losses = []
            with torch.no_grad():
                for j in range(len(val_ds)):
                    I_val = get_embeddings([val_ds[j]['inputs']])
                    I_val = torch.tensor(I_val, dtype=torch.bfloat16).to(device)

                    if val_ds[j]["task"] not in {"story_cloze_10templates", "piqa_10templates", "copa_10templates", "hellaswag_10templates"}:
                        continue

                    val_item = val_ds_prompt[j]
                    val_batch = tokenizer(
                        val_item["full_prompt"],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(device)
                    # prefix_tok = tokenizer(
                    #     val_item["full_prompt"],
                    #     truncation=True,
                    #     max_length=512,
                    #     return_tensors="pt",
                    # )
                    # val_batch["labels"] = val_batch["input_ids"].clone()
                    # val_batch["labels"][val_batch["attention_mask"] == 0] = -100
                    # val_batch["labels"][:, :prefix_tok["input_ids"].size(0)] = -100  # only compute loss on the target
                    
                    val_out = peft_model.generate(**val_batch, max_new_tokens=50,
                    temperature=0.001, merging_type='moe', scorers=scorers)
                    references = [val_item["targets"]]
                    candidates = [tokenizer.decode(val_out[0], skip_special_tokens=True).strip().split('### Response:\n')[-1]]

                    # for j in range(val_out.size(0)):
                    #     print(tokenizer.decode(val_out[j],skip_special_tokens=True))
                    #     print(val_item["targets"])
                    val_losses.append(calculate_em(references,candidates))
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
            print(len(val_losses))
            print(f"[Step {i+1}] Validation loss: {mean_val_loss:.4f}")
            x=10/0
            scorer.train()

        pbar.update(1)

save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(save_dir, exist_ok=True)

# Save all scorers in the array
for i, scorer in enumerate(scorers):
    ckpt_path = os.path.join(save_dir, f"scorer_layer_{i}_mixture_mlp.pt")
    cfg_path = os.path.join(save_dir, f"scorer_layer_{i}_mixture_mlp.config.json")

    # Save state_dict on CPU to avoid device issues
    state_cpu = {k: v.detach().cpu() for k, v in scorer.state_dict().items()}
    torch.save(state_cpu, ckpt_path)

    # Save minimal config to reconstruct the module later
    scorer_config = {
        "layer_idx": i,
        "top_k": int(scorer.top_k) if scorer.top_k is not None else None,
        "temperature": float(scorer.tau),
        "d_in": int(scorer.Wi.weight.shape[1]),
        "d_proj": int(scorer.Wi.weight.shape[0]),
        "d_a": int(scorer.Wr[0].weight.shape[1]),  # First layer of MLP
        "K": int(scorer.A.shape[0]),
        "dtype": "bfloat16",
        "use_mlp": True,  # Flag to indicate MLP structure
    }
    with open(cfg_path, "w") as f:
        json.dump(scorer_config, f, indent=2)

    print(f"Saved scorer {i} to: {ckpt_path}")

print(f"Saved all {len(scorers)} scorers to: {save_dir}")