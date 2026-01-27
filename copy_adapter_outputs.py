import json
import shutil
from pathlib import Path

def load_adapter_ids(adapter_file):
    """Load model_ids from llama2_7b_adapters.json"""
    with open(adapter_file, 'r') as f:
        adapters = json.load(f)
    return [adapter['model_id'] for adapter in adapters]

def load_config_model_names(config_file):
    """Load model_names from config2.json"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return [model['model_name'] for model in config]

def main():
    # File paths
    adapter_file = "scripts/llama2_7b_adapters.json"
    config_file = "config/config2.json"
    source_dir = Path("performance_large/outputs")
    dest_dir = Path("performance_large/outputs_hf_large")
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the lists
    print("Loading adapter IDs from llama2_7b_adapters.json...")
    adapter_ids = load_adapter_ids(adapter_file)
    print(f"Found {len(adapter_ids)} adapters")
    
    print("\nLoading model names from config2.json...")
    config_model_names = load_config_model_names(config_file)
    print(f"Found {len(config_model_names)} models in config")
    
    # Create a mapping of model_name to index in adapter_ids
    # Assuming model_name format matches the last part of model_id
    # e.g., "Styxxxx/llama2_7b_lora-sst2" -> "sst2"
    adapter_name_to_index = {}
    for idx, model_id in enumerate(adapter_ids):
        # Extract the last part after '-' or '/'
        if '-' in model_id:
            name_part = model_id.split('-')[-1]
        elif '/' in model_id:
            name_part = model_id.split('/')[-1]
        else:
            name_part = model_id
        adapter_name_to_index[name_part] = idx
    
    # Also try full model_id as key
    for idx, model_id in enumerate(adapter_ids):
        adapter_name_to_index[model_id] = idx
    
    print(f"\nCreated mapping for {len(adapter_name_to_index)} adapter names")
    
    # Copy files
    copied_count = 0
    skipped_count = 0
    
    print("\nCopying files...")
    for i, model_name in enumerate(config_model_names):
        source_file = source_dir / f"model_{i}.json"
        
        # Try to find position in adapter_ids
        position = None
        
        # Try direct match
        if model_name in adapter_name_to_index:
            position = adapter_name_to_index[model_name]
        # Try with prefix (e.g., "Styxxxx/llama2_7b_lora-" + model_name)
        elif f"Styxxxx/llama2_7b_lora-{model_name}" in adapter_name_to_index:
            position = adapter_name_to_index[f"Styxxxx/llama2_7b_lora-{model_name}"]
        
        if position is not None:
            dest_file = dest_dir / f"hf_adapter_outputs_{position}.json"
            
            if source_file.exists():
                shutil.copy2(source_file, dest_file)
                print(f"✓ Copied model_{i}.json ({model_name}) -> hf_adapter_outputs_{position}.json")
                copied_count += 1
            else:
                print(f"✗ Source file not found: {source_file}")
                skipped_count += 1
        else:
            print(f"⊘ Could not find position for model: {model_name}")
            skipped_count += 1
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Files copied: {copied_count}")
    print(f"  Files skipped: {skipped_count}")
    print(f"  Total models: {len(config_model_names)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
