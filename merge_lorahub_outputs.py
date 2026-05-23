import os
import json
from datasets import load_dataset
import os.path as osp
from typing import Union

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("./templates", f"{template_name}.json")
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

prompter = Prompter("alpaca")

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

def collect_outputs(input_dir, output_file):
    task_to_metris = {}
    task_to_domain = {}
    input_to_target = {}
    task_to_targets = {}
    task_to_inputs = {}
    data_path = "./dataset/combined_test.json"
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path)
    else:
        dataset = load_dataset(data_path)

    dataset = dataset["train"].map(generate_and_tokenize_prompt)
    for item in dataset:
        task_to_metris[item['task']] = item['metric']
        task_to_domain[item['task']] = item['domain']
        if item['task'] not in task_to_targets:
            task_to_targets[item['task']] = []
            task_to_inputs[item['task']] = []
        task_to_targets[item['task']].append(item['targets'])
        task_to_inputs[item['task']].append(item['full_prompt'])
    all_outputs = []
    for fname in os.listdir(input_dir):
        if fname.endswith('13b.json'):
            continue
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
                    generated_answer = pred.get('predicted').strip().split('### Response:\n')[-1]
                    id = pred.get('id')
                    all_outputs.append({
                        'input': task_to_inputs[task_name][id],
                        'targets': task_to_targets[task_name][id],
                        'metric': task_to_metris.get(task_name),
                        'domain': task_to_domain.get(task_name),
                        'task': task_name,
                        'predicted_answer': generated_answer,
                    })
    with open(output_file, 'w') as out:
        json.dump(all_outputs, out, indent=2)

if __name__ == "__main__":
    input_dir = "/home/pavlovic/lorahub_new/lorahub_outputs_ood"
    output_file = "merged_outputs.json"
    collect_outputs(input_dir, output_file)
    print(f"Merged outputs written to {output_file}")
