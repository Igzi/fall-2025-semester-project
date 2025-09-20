from peft import PeftModel
from peft import LoraConfig, get_peft_model
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json
import torch

def load_adapter_to_memory(adapters, adapter_name):
    """
    Load adapter configuration and weights into memory dictionary.
    """
    try:
        print(f"Loading adapter {adapter_name} into memory...")
        
        # Download config file
        config_file = hf_hub_download(
            repo_id=adapter_name,
            filename="adapter_config.json"
        )
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Download weights file
        try:
            weights_file = hf_hub_download(
                repo_id=adapter_name,
                filename="adapter_model.safetensors"
            )
            weights = load_file(weights_file)
        except:
            # Fallback to .bin format
            weights_file = hf_hub_download(
                repo_id=adapter_name,
                filename="adapter_model.bin"
            )
            weights = torch.load(weights_file, map_location='cpu')
        
        # Store everything in memory
        adapters[adapter_name] = {
            'config': config,
            'weights': weights,
            'loaded': True
        }
        
        print(f"Adapter {adapter_name} loaded successfully!")
        print(f"Config: {config}")
        print(f"Number of weight tensors: {len(weights)}")
        
        return adapters[adapter_name]
        
    except Exception as e:
        print(f"Error loading adapter {adapter_name}: {e}")
        return None

def get_adapter_config(lora_adapters):
    """
    Extract LoRA configuration from the cached adapter information.
    """
    r = None
    lora_alpha = None
    target_modules = None
    lora_dropout = None
    bias = None
    task_type = None
    base_model_name_or_path = None

    for adapter in lora_adapters.values():
        if r is None:
            config = adapter['config']
            r = config['r']
            lora_alpha = config['lora_alpha']
            target_modules = config['target_modules']
            lora_dropout = config.get('lora_dropout', 0.0)
            bias = config.get('bias', 'none')
            task_type = config.get('task_type', 'CAUSAL_LM')
            base_model_name_or_path = config.get('base_model_name_or_path', None)
        else:
            assert r == adapter['config']['r'], "Inconsistent r values"
            #assert lora_alpha == adapter['config']['lora_alpha'], "Inconsistent lora
            #assert target_modules == adapter['config']['target_modules'], "Inconsistent target_modules"
            assert lora_dropout == adapter['config'].get('lora_dropout', 0.0), "Inconsistent lora_dropout"
            assert bias == adapter['config'].get('bias', 'none'), "Inconsistent bias"
            assert task_type == adapter['config'].get('task_type', 'CAUSAL_LM'), "Inconsistent task_type"
            assert base_model_name_or_path == adapter['config'].get('base_model_name_or_path', None), "Inconsistent base_model_name_or_path"

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type
    )

def merge_adapters_fusion(base_model, lora_adapters):
    """
    Merge multiple LoRA adapters into the base model.
    """
    merged_config = get_adapter_config(lora_adapters)

    # Create PEFT model
    peft_model = get_peft_model(base_model, merged_config)
    num_adapters = len(lora_adapters)

    averaged_weights = {}
    
    # Get all weight keys from the first adapter
    first_adapter = next(iter(lora_adapters.values()))
    weight_keys = list(first_adapter['weights'].keys())
    
    # Average weights across all adapters
    for key in weight_keys:
        # Collect weights for this key from all adapters
        weight_tensors = []
        for adapter in lora_adapters.values():
            if key in adapter['weights']:
                weight_tensors.append(adapter['weights'][key])
        
        # Average the weights
        if weight_tensors:
            averaged_weights[key] = torch.stack(weight_tensors).mean(dim=0)
            print(f"Averaged {len(weight_tensors)} weights for {key}")

    peft_model.load_state_dict(averaged_weights, strict=False)

    return peft_model
    