from huggingface_hub import list_models
import json
import os
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from tqdm import tqdm # Optional, but good for tracking progress

# 1. Define the specific filter tag from the URL
# The filter 'other=base_model:adapter:meta-llama/Llama-2-7b-hf' 
# translates directly to this tag string in the Hub API.
TARGET_TAG = "base_model:adapter:meta-llama/Llama-2-7b-hf"
INCOMPATIBLE_TARGETS = {"embed_tokens", "word_embeddings", "lm_head", "token_type_embeddings"}
EXLUDED_ADAPTERS = ['NouRed/quantized-llama2-alpaca',
                    'sravaniayyagari/new-finetuned-model',
                    'Harit10/Llama2-PII_final',
                    'Harit10/Llama2-config',
                    'vashistht/bonsai-reasoning-adapter_prune_c4_ft_wiki']

def find_adapter_configs(model_id: str) -> list[str]:
    """
    Return paths to adapter config files inside a repo (root or subfolders).
    """
    files = list_repo_files(model_id, repo_type="model")
    return [p for p in files if p.endswith("adapter_config.json") or p.endswith("adapter_config.json#")], files


def is_lora_for_llama2_7b(model_id: str) -> list[dict]:
    """
    Inspect all adapter_config.json files in the repo and return matching adapters.
    Each match contains the repo id and the adapter subpath.
    """
    if model_id in EXLUDED_ADAPTERS:
        return []
    
    if model_id.startswith("simonycl/llama-2-7b-hf"):
        return []

    matches = []
    cfg_paths, files = find_adapter_configs(model_id)
    for f in files:
        if f.endswith("added_tokens.json"):
            return []
    
    for cfg_path in cfg_paths:
        try:
            local = hf_hub_download(model_id, cfg_path, repo_type="model")
            with open(local, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue

        peft_type = (cfg.get("peft_type") or cfg.get("base_peft_type") or "").upper()
        base = (cfg.get("base_model_name_or_path") or cfg.get("base_model_name") or "")
        target_modules = cfg.get("target_modules")
        modules_to_save = cfg.get("modules_to_save")
        rank = cfg.get("r") or cfg.get("lora_r")

        if peft_type != "LORA":
            continue

        if base and "Llama-2-7b-hf" not in base:
            # Adapter targets a different base checkpoint; skip to avoid vocab/shape mismatches.
            continue

        if not isinstance(target_modules, (list, tuple)) or not target_modules:
            # Missing or malformed target modules list; skip to avoid runtime issues.
            continue

        if any(module in INCOMPATIBLE_TARGETS for module in target_modules):
            # Known incompatible targets (e.g. embed tokens, lm head) for Llama-2.
            continue

        if isinstance(modules_to_save, (list, tuple)) and any(
            module in INCOMPATIBLE_TARGETS for module in modules_to_save
        ):
            continue

        if isinstance(rank, int) and rank > 64:
            # Keep adapters bounded to manageable LoRA rank.
            continue

        adapter_name = "/".join(cfg_path.split("/")[:-1]) or "(root)"
        matches.append({
            "repo_id": model_id,
            "adapter_path": adapter_name,
            "base_model_name_or_path": base,
            "config_path": cfg_path,
            "rank": rank,
        })
    return matches

# 2. Use list_models with the 'tags' filter
print(f"Fetching models with the tag: {TARGET_TAG}...")

# Set a large limit to retrieve all results. 
# The list_models function handles pagination automatically.
try:
    # list_models returns a generator, which we convert to a list
    model_generator = list_models(
        tags=[TARGET_TAG],
        # Sort by downloads to get more relevant models first (optional)
        sort="downloads",
        direction=-1,
        # Set 'limit' to None to fetch all models that match the filter
        limit=None 
    )

    # Convert the generator to a list, using tqdm for a progress bar
    models = list(tqdm(model_generator, desc="Processing models"))
    
except Exception as e:
    print(f"An error occurred while fetching models: {e}")
    models = []

# 3. Process the results
if models:
    # Extract model repository ID and other useful information
    data = []
    for model in models:
        res = is_lora_for_llama2_7b(model.modelId)
        if res:
            data.append({
                "model_id": model.modelId,
                "downloads": model.downloads,
                "likes": model.likes,
                "rank": res[0].get("rank", None),
            })

    data = sorted(data, key=lambda x: x["rank"], reverse=True)

    print(f"Found {len(data)} LoRA adapter(s) for Llama-2-7b:")
    output_path = os.path.join(os.path.dirname(__file__), "llama2_7b_adapters.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Adapter list written to {output_path}")
    
else:
    print("No models found or an error occurred.")