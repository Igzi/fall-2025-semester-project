import json
import os
from pathlib import Path

def extract_model_mapping(folder_path):
    """
    Load all JSON files in the folder and extract mapping from file_name to model value.
    
    Args:
        folder_path: Path to the folder containing JSON files
    
    Returns:
        Dictionary mapping file_name to model value
    """
    folder = Path(folder_path)
    model_mapping = {}
    
    # Iterate through all JSON files in the folder
    for file_path in sorted(folder.glob("*.json")):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract model value from the first dictionary (assuming all have the same model)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                model_value = data[0].get("model", None)
                if model_value is None:
                    model_value = data[0].get("model_name", None)
                if model_value:
                    # Filter for models starting with "Styxxxx/"
                    if model_value.startswith("Styxxxx/") or "/" not in model_value:
                        model_mapping[file_path.name] = model_value
                        print(f"✓ {file_path.name}: {model_value}")
                    else:
                        print(f"⊘ {file_path.name}: Skipped (not Styxxxx/)")
                else:
                    print(f"✗ {file_path.name}: No 'model' field found")
            else:
                print(f"✗ {file_path.name}: Invalid format (not a list of dicts)")
                
        except json.JSONDecodeError as e:
            print(f"✗ {file_path.name}: JSON decode error - {e}")
        except Exception as e:
            print(f"✗ {file_path.name}: Error - {e}")
    
    return model_mapping


if __name__ == "__main__":
    # Path to the folder
    folder_path = "performance_large/outputs_hf_large"
    
    print(f"Extracting model mapping from: {folder_path}\n")
    
    # Extract the mapping
    mapping = extract_model_mapping(folder_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Total files processed: {len(mapping)}")
    print(f"{'='*80}\n")
    
    # Save the mapping to a JSON file
    output_file = "model_mapping.json"
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Mapping saved to: {output_file}")
    
    # Print the mapping
    print("\nMapping:")
    for file_name, model in sorted(mapping.items()):
        print(f"  {file_name} -> {model}")
