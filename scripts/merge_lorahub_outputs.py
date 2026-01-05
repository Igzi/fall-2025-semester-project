import os
import json
import argparse

def merge_lorahub_outputs(input_folder, output_file, model_size):
    """
    Merge all LoRAHub task outputs for a specific model size into a single file.
    
    Args:
        input_folder: Path to folder containing task output files ({task_id}_{model_size}.json)
        output_file: Path to output file
        model_size: Model size to process (e.g., '7b' or '13b')
    """
    # Define task order to match reference file
    task_order = [
        "web_nlg_en", "dart", "e2e_nlg", "common_gen", "story_cloze", "piqa", "copa", 
        "hellaswag", "sst2", "yelp_polarity_reviews", "imdb_reviews", "sentiment140",
        "multirc", "squad_v2", "squad_v1", "openbookqa", "bool_q", "drop",
        "natural_questions", "arc_easy", "arc_challenge", "definite_pronoun_resolution",
        "wsc", "cosmos_qa", "record", "paws_wiki", "glue_qqp", "glue_mrpc", "cb", "wnli",
        "mnli_matched", "anli_r3", "anli_r2", "anli_r1", "mnli_mismatched", "snli",
        "qnli", "rte", "wmt16_translate_tren", "wmt16_translate_deen", "wmt16_translate_ruen",
        "wmt16_translate_fien", "wmt16_translate_roen", "wmt14_enfr", "wmt16_translate_csen",
        "stsb", "trivia_qa", "para_crawl_enes"
    ]
    
    merged_results = []
    
    # Get all files matching the pattern
    task_files = sorted([f for f in os.listdir(input_folder) 
                        if f.endswith(f'_{model_size}.json')],
                       key=lambda x: int(x.split('_')[0]))
    
    print(f"Found {len(task_files)} task files for model size {model_size}")
    
    # Store predictions by task name first
    predictions_by_task = {}
    
    for task_file in task_files:
        task_path = os.path.join(input_folder, task_file)
        
        with open(task_path, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
        
        task_name = task_data.get('task_name', 'unknown')
        predictions = task_data.get('predictions', [])

        task_name_to_domain = {
            "anli_r1": "nli",
            "anli_r2": "nli",
            "anli_r3": "nli",
            "arc_challenge": "closed_book QA",
            "arc_easy": "closed_book QA",
            "bool_q": "reading comp",
            "cb": "nli",
            "common_gen": "struct to text",
            "copa": "commonsense",
            "cosmos_qa": "read.comp.w:commonsense",
            "dart": "struct to text",
            "definite_pronoun_resolution": "coreference",
            "drop": "reading comp",
            "e2e_nlg": "struct to text",
            "glue_mrpc": "paraphrase",
            "glue_qqp": "paraphrase",
            "hellaswag": "commonsense",
            "imdb_reviews": "sentiment",
            "mnli_matched": "nli",
            "mnli_mismatched": "nli",
            "multirc": "reading comp",
            "natural_questions": "closed_book QA",
            "openbookqa": "reading comp",
            "para_crawl_enes": "translation",
            "paws_wiki": "paraphrase",
            "piqa": "commonsense",
            "qnli": "nli",
            "record": "read.comp.w:commonsense",
            "rte": "nli",
            "sentiment140": "sentiment",
            "snli": "nli",
            "squad_v1": "reading comp",
            "squad_v2": "reading comp",
            "sst2": "sentiment",
            "story_cloze": "commonsense",
            "stsb": "paraphrase",
            "trivia_qa": "closed_book QA",
            "web_nlg_en": "struct to text",
            "wmt14_enfr": "translation",
            "wmt16_translate_csen": "translation",
            "wmt16_translate_deen": "translation",
            "wmt16_translate_fien": "translation",
            "wmt16_translate_roen": "translation",
            "wmt16_translate_ruen": "translation",
            "wmt16_translate_tren": "translation",
            "wnli": "nli",
            "wsc": "coreference",
            "yelp_polarity_reviews": "sentiment"
        }

        task_to_metric = {
            "anli_r1": "em",
            "anli_r2": "em",
            "anli_r3": "em",
            "arc_challenge": "em",
            "arc_easy": "em",
            "bool_q": "em",
            "cb": "em",
            "common_gen": "rouge",
            "copa": "em",
            "cosmos_qa": "em",
            "dart": "rouge",
            "definite_pronoun_resolution": "em",
            "drop": "em",
            "e2e_nlg": "rouge",
            "glue_mrpc": "em",
            "glue_qqp": "em",
            "hellaswag": "em",
            "imdb_reviews": "em",
            "mnli_matched": "em",
            "mnli_mismatched": "em",
            "multirc": "em",
            "natural_questions": "em",
            "openbookqa": "em",
            "para_crawl_enes": "bleu",
            "paws_wiki": "em",
            "piqa": "em",
            "qnli": "em",
            "record": "em",
            "rte": "em",
            "sentiment140": "em",
            "snli": "em",
            "squad_v1": "em",
            "squad_v2": "em",
            "sst2": "em",
            "story_cloze": "em",
            "stsb": "em",
            "trivia_qa": "em",
            "web_nlg_en": "rouge",
            "wmt14_enfr": "bleu",
            "wmt16_translate_csen": "bleu",
            "wmt16_translate_deen": "bleu",
            "wmt16_translate_fien": "bleu",
            "wmt16_translate_roen": "bleu",
            "wmt16_translate_ruen": "bleu",
            "wmt16_translate_tren": "bleu",
            "wnli": "em",
            "wsc": "em",
            "yelp_polarity_reviews": "em"
        }
        
        print(f"Processing {task_file}: {task_name} with {len(predictions)} predictions")
        
        # Store predictions for this task
        task_predictions = []
        for pred in predictions:
            # Extract the relevant fields
            predicted_answer = pred.get('predicted', '')
            predicted_answer = predicted_answer.strip().split('### Response:\n')[-1]
            result_entry = {
                'inputs': pred.get('input', ''),
                'targets': pred.get('target', ''),
                'metric': task_to_metric.get(task_name, 'em'), 
                'domain': task_name_to_domain.get(task_name, 'unknown'),
                'task': task_name,
                'predicted_answer': predicted_answer
            }
            task_predictions.append(result_entry)
        
        predictions_by_task[task_name] = task_predictions
    
    # Now add predictions in the correct order
    for task_name in task_order:
        if task_name in predictions_by_task:
            merged_results.extend(predictions_by_task[task_name])
    
    # Write merged results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=4, ensure_ascii=False)
    
    print(f"\nSuccessfully merged {len(merged_results)} predictions into {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Merge LoRAHub task outputs into a single file')
    parser.add_argument('--input_folder', type=str, default='./lorahub_outputs',
                       help='Path to folder containing task output files')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output file (e.g., baselines/lorahub_7b.json)')
    parser.add_argument('--model_size', type=str, required=True, choices=['7b', '13b'],
                       help='Model size to process (7b or 13b)')
    
    args = parser.parse_args()
    
    merge_lorahub_outputs(args.input_folder, args.output_file, args.model_size)

if __name__ == '__main__':
    main()
