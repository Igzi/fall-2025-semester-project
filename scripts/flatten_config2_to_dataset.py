#!/usr/bin/env python3
"""Flatten config/config2.json into a dataset JSON array.

Output format: a JSON array where each element is an object with keys:
- model_name
- inputs
- targets

Usage:
    python3 scripts/flatten_config2_to_dataset.py \
        --input config/config2.json \
        --output dataset/config2_flat.json

If output dir doesn't exist, it will be created.
"""
import argparse
import json
import os

domain_to_metric = {
    "struct to text": "rouge",
    "commonsense": "em",
    "sentiment": "em",
    "reading comp": "em",
    "closed_book QA": "em",
    "coreference": "em",
    "read.comp.w:commonsense": "em",
    "translation": "bleu",
    "paraphrase": "em",
    "nli": "em",
}


def flatten_config(input_path: str, output_path: str) -> int:
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out = []
    for entry in data:
        model_name = entry.get('model_name')
        domain = entry.get('domain', None)
        samples = entry.get('sample', [])
        for s in samples:
            # Only include items that have inputs (and optionally targets)
            if 'inputs' not in s:
                continue
            item = {
                'model_name': model_name,
                'domain': domain,
                'metric': domain_to_metric.get(domain, "None"),
                'inputs': s.get('inputs'),
                'targets': s.get('targets')
            }
            out.append(item)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return len(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='config/config2.json', help='Path to config JSON')
    parser.add_argument('--output', '-o', default='dataset/config2_flat.json', help='Path to write flattened dataset')
    args = parser.parse_args()

    count = flatten_config(args.input, args.output)
    print(f'Wrote {count} samples to {args.output}')
