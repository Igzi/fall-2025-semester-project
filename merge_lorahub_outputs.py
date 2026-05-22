import os
import json

def collect_outputs(input_dir, output_file):
    all_outputs = []
    for fname in os.listdir(input_dir):
        if fname.endswith('.json'):
            fpath = os.path.join(input_dir, fname)
            with open(fpath, 'r') as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Skipping {fname}: {e}")
                    continue
                task_name = data.get('task_name', os.path.splitext(fname)[0])
                for pred in data.get('predictions', []):
                    all_outputs.append({
                        'input': pred.get('input'),
                        'prediction': pred.get('predicted'),
                        'task_name': task_name
                    })
    with open(output_file, 'w') as out:
        json.dump(all_outputs, out, indent=2)

if __name__ == "__main__":
    input_dir = "/home/pavlovic/lorahub_new/lorahub_outputs"
    output_file = "merged_outputs.json"
    collect_outputs(input_dir, output_file)
    print(f"Merged outputs written to {output_file}")
