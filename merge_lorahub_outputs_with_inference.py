import os
import json
from lorahub.algorithm import lorahub_inference
from lorahub.constant import LORA_MODULE_NAMES
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_weights_for_task(json_files, ntasks):
    """
    Load weights for each task from the corresponding json file.
    Returns a dict: {task_name: weights}
    """
    weights = {}
    for fname in json_files:
        with open(fname, 'r') as f:
            data = json.load(f)
            task_name = data.get('task_name', os.path.splitext(os.path.basename(fname))[0])
            weights[task_name] = data.get('module_weights')
    return weights

def main():
    input_dir = "/home/pavlovic/lorahub_new/lorahub_outputs"
    test_file = "/home/pavlovic/dataset/combined_test.json"
    model_path = "meta-llama/Llama-3.1-8B"
    output_file = "merged_outputs_with_predictions.json"
    ntasks = 48  # Set as needed

    # Gather all json files
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    weights_dict = load_weights_for_task(json_files, ntasks)

    # Load test set
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0
    model = model.cuda() if torch.cuda.is_available() else model

    results = []
    for item in test_data:
        task_name = item.get('task')
        input_text = item['inputs']
        target = item['targets']
        weights = weights_dict.get(task_name)
        if weights is None:
            print(f"No weights found for task {task_name}, skipping.")
            continue
        # Prepare prompt (if needed, adapt to your prompt template)
        prompt = input_text
        # Run inference
        pred = lorahub_inference([prompt], model, tokenizer, batch_size=1, module_weights=weights)[0]
        results.append({
            'input': input_text,
            'prediction': pred,
            'target': target,
            'task_name': task_name
        })

    with open(output_file, 'w') as out:
        json.dump(results, out, indent=2)
    print(f"Merged outputs with predictions written to {output_file}")

if __name__ == "__main__":
    main()
