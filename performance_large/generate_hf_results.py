import os
import json
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np
import pandas as pd

# Function to calculate BLEU score
def calculate_bleu(references, candidates):
    scores = [sentence_bleu([ref.split()], cand.split()) for ref, cand in zip(references, candidates)]
    return np.round(np.mean(scores) * 100, 1) if scores else 0

# Function to calculate ROUGE score
def calculate_rouge(references, candidates):
    rouge = Rouge()
    scores = rouge.get_scores(candidates, references, avg=True)
    rouge_1 = np.round(scores['rouge-1']['f'] * 100, 1)
    rouge_2 = np.round(scores['rouge-2']['f'] * 100, 1)
    rouge_l = np.round(scores['rouge-l']['f'] * 100, 1)
    return rouge_1, rouge_2, rouge_l

# Function to calculate Exact Match score
def calculate_em(options, inputs, references, candidates):
    references = [ref.split("\n\n")[0] for ref in references]
    em_scores = [1 if cal_correct(option, input, ref, cand) else 0 for option, input, ref, cand in zip(options, inputs, references, candidates)]
    return np.round(np.mean(em_scores) * 100, 1) if em_scores else 0

def cal_correct(options, input, expected_answer, generated_answer):
    #options=None
    if options is None:
        gen_ans_clean = generated_answer.strip().lower().replace(".", "")
        input_clean = input.strip().lower().replace(".", "")
        
        if len(gen_ans_clean)>len(expected_answer.strip().lower().replace(".", "")) and gen_ans_clean in input_clean or input_clean in gen_ans_clean:
            return False  # Generated answer is too similar to input
        return expected_answer.strip().lower().replace(".", "") in generated_answer.strip().lower().replace(".", "")
    else:
        option_list = [opt.strip().lower().replace(".", "") for opt in options]
        gen_ans_clean = generated_answer.strip().lower().replace(".", "")
        exp_ans_clean = expected_answer.strip().lower().replace(".", "")
        assert exp_ans_clean in option_list
        for opt in option_list:
            if opt in exp_ans_clean:
                continue
            if exp_ans_clean != opt and opt in gen_ans_clean:
                return False
        return (exp_ans_clean in gen_ans_clean)

# Function to process a file
def process_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    organized_data = defaultdict(lambda: defaultdict(list))
    for entry in data:
        domain = entry['model_name']
        task = entry['model_name']
        organized_data[domain][task].append(entry)
    
    return organized_data

def get_options_from_input(input_text):
    if "OPTIONS:" in input_text:
        options = input_text.split("OPTIONS:\n-")[1]
        options = options.strip().split("\n- ")
        return options
    return None

# Function to process all files in a folder and aggregate scores by domain and metric
def process_folder(folder_path):
    domain_specific_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for file_name in os.listdir(folder_path):
        if not (file_name.startswith('model_test') or file_name.startswith('degraded_model')) and file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            domains_data = process_file(file_path)

            for domain, tasks_data in domains_data.items():
                for task, entries in tasks_data.items():
                    # if task!="squad_v2":
                    #     continue
                    # if file_name!="hf_adapter_outputs_1063.json":
                    #     continue

                    metric = entries[0]['metric']
                    inputs = [entry['inputs'] for entry in entries]
                    references = [entry['targets'] for entry in entries]
                    candidates = [entry['predicted_answer'] for entry in entries]
                    options = [get_options_from_input(entry['inputs']) for entry in entries]

                    if metric == 'bleu':
                        score = calculate_bleu(references, candidates)
                        domain_specific_metrics[domain][metric][file_name].append(score)
                    elif metric == 'rouge':
                        rouge_1, rouge_2, rouge_l = calculate_rouge(references, candidates)
                        domain_specific_metrics[domain]['rouge-avg'][file_name].append((rouge_1+rouge_2+rouge_l)/3)
                        # domain_specific_metrics[domain]['rouge-1'][file_name].append(rouge_1)
                        # domain_specific_metrics[domain]['rouge-2'][file_name].append(rouge_2)
                        # domain_specific_metrics[domain]['rouge-l'][file_name].append(rouge_l)
                    elif metric == 'em':
                        score = calculate_em(options, inputs, references, candidates)
                        domain_specific_metrics[domain][metric][file_name].append(score)
    
    return domain_specific_metrics

# Function to convert data to LaTeX format with domain and metric averages
def convert_to_latex_modified(data, folder_path):
    # Define custom ordering
    custom_order = [
        'sst2-em', 'cb-em', 'multirc-em', 'wnli-em', 'squad_v2-em', 'web_nlg_en-rouge-avg',
        'definite_pronoun_resolution-em', 'wmt16_translate_tren-bleu', 'wmt16_translate_deen-bleu',
        'squad_v1-em', 'dart-rouge-avg', 'cosmos_qa-em', 'mnli_matched-em', 'anli_r3-em',
        'e2e_nlg-rouge-avg', 'anli_r2-em', 'natural_questions-em', 'paws_wiki-em',
        'wmt16_translate_ruen-bleu', 'glue_qqp-em', 'story_cloze-em', 'openbookqa-em',
        'yelp_polarity_reviews-em', 'arc_easy-em', 'wmt16_translate_fien-bleu', 'anli_r1-em',
        'mnli_mismatched-em', 'imdb_reviews-em', 'wmt16_translate_roen-bleu', 'common_gen-rouge-avg',
        'snli-em', 'sentiment140-em', 'piqa-em', 'wmt14_enfr-bleu', 'wsc-em', 'arc_challenge-em',
        'copa-em', 'qnli-em', 'glue_mrpc-em', 'bool_q-em', 'hellaswag-em', 'wmt16_translate_csen-bleu',
        'rte-em', 'drop-em', 'record-em', 'stsb-em', 'trivia_qa-em', 'para_crawl_enes-bleu'
    ]
    
    data_list = []
    for domain, metrics in data.items():
        for metric, files in metrics.items():
            row = {'Domain-Metric': f"{domain}-{metric}"}
            for file_name in os.listdir(folder_path):
                if not (file_name.startswith('model_test') or file_name.startswith('degraded_model')) and file_name.endswith('.json'):
                    numeric_scores = [score for score in files[file_name] if isinstance(score, (int, float))]
                    average_score = np.mean(numeric_scores) if numeric_scores else 0
                    row[file_name] = "{:.1f}".format(average_score)  # Format to one decimal place
            data_list.append(row)

    # Sort data_list by custom order
    order_dict = {key: idx for idx, key in enumerate(custom_order)}
    data_list.sort(key=lambda x: order_dict.get(x['Domain-Metric'], len(custom_order)))

    df = pd.DataFrame(data_list)
    columns_ordered = ['Domain-Metric'] + [file_name for file_name in os.listdir(folder_path) if not (file_name.startswith('model_test') or file_name.startswith('degraded_model')) and file_name.endswith('.json')]
    df = df[columns_ordered]
    results = []
    for d in data_list:
        results.append([])
        for model_id in range(1700):
            results[-1].append(d.get(f'hf_adapter_outputs_{model_id}.json', None))
    
    results = np.array(results)
    # Save numeric results matrix
    np.save("./performance_large/model_hf_performance.npy", results, allow_pickle=True)

    # Prepare matrix for plotting: convert None to np.nan and ensure float dtype
    try:
        plot_matrix = np.array([[float(x) if x is not None else np.nan for x in row] for row in results])
    except Exception:
        # Fallback: coerce with numpy (handles mixed types)
        plot_matrix = results.astype(np.float64)
    
    # Row-wise min-max normalization to [0,1], ignoring NaNs
    # For each row: norm_row = (row - min_row) / (max_row - min_row)
    # If max_row == min_row (constant row), set normalized values to 0
    with np.errstate(invalid='ignore'):
        row_max = np.nanmax(plot_matrix, axis=1, keepdims=True)
        row_argmax = np.nanargmax(plot_matrix, axis=1)
    
    row_series = pd.Series(np.squeeze(row_max), name='row_max')
    argmax_series = pd.Series(row_argmax, name='best_model')
    
    combined = pd.concat([
        df['Domain-Metric'].reset_index(drop=True), 
        row_series.reset_index(drop=True),
        argmax_series.reset_index(drop=True)
    ], axis=1)
    
    # Calculate mean before formatting as strings
    mean_row_max = row_series[np.isfinite(row_series)].mean()
    combined['row_max'] = combined['row_max'].apply(lambda x: f"{x:.4f}" if np.isfinite(x) else 'NaN')
    with open('./scripts/llama2_7b_adapters.json', 'r') as file:
        lora_adapters = json.load(file)
    combined['best_model'] = combined['best_model'].apply(lambda x: lora_adapters[x]['model_id'] if x < len(lora_adapters) else 'N/A')
    
    print(combined.to_string(index=False))
    print(f"\nMean of row_max values: {mean_row_max:.4f}")

    return df.to_latex(index=False)

# Example usage
folder_path = './performance_large/outputs_hf_large'  # Replace with your actual folder path
processed_data = process_folder(folder_path)
latex_table = convert_to_latex_modified(processed_data, folder_path)
#print(latex_table)
