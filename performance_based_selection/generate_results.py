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
def calculate_em(references, candidates):
    references = [ref.split("\n\n")[0] for ref in references]
    em_scores = [1 if cal_correct(ref, cand) else 0 for ref, cand in zip(references, candidates)]
    return np.round(np.mean(em_scores) * 100, 1) if em_scores else 0

def cal_correct(generated_answer, expected_answer):
    is_correct = generated_answer.strip().lower().replace(".", "") == expected_answer.strip().lower().replace(".", "")
    return is_correct

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

# Function to process all files in a folder and aggregate scores by domain and metric
def process_folder(folder_path):
    domain_specific_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            domains_data = process_file(file_path)

            for domain, tasks_data in domains_data.items():
                for task, entries in tasks_data.items():
                    metric = entries[0]['metric']
                    references = [entry['targets'] for entry in entries]
                    candidates = [entry['predicted_answer'] for entry in entries]

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
                        score = calculate_em(references, candidates)
                        domain_specific_metrics[domain][metric][file_name].append(score)
    
    return domain_specific_metrics

# Function to convert data to LaTeX format with domain and metric averages
def convert_to_latex_modified(data, folder_path):
    data_list = []
    for domain, metrics in data.items():
        for metric, files in metrics.items():
            row = {'Domain-Metric': f"{domain}-{metric}"}
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    numeric_scores = [score for score in files[file_name] if isinstance(score, (int, float))]
                    average_score = np.mean(numeric_scores) if numeric_scores else 0
                    row[file_name] = "{:.1f}".format(average_score)  # Format to one decimal place
            data_list.append(row)

    df = pd.DataFrame(data_list)
    columns_ordered = ['Domain-Metric'] + [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.json')]
    df = df[columns_ordered]
    results = []
    for d in data_list:
        results.append([])
        for model_id in range(48):
            results[-1].append(d.get(f'model_{model_id}.json', None))
    
    results = np.array(results)
    # Save numeric results matrix
    np.save("./performance_based_selection/model_performance.npy", results, allow_pickle=True)

    # Prepare matrix for plotting: convert None to np.nan and ensure float dtype
    try:
        plot_matrix = np.array([[float(x) if x is not None else np.nan for x in row] for row in results])
    except Exception:
        # Fallback: coerce with numpy (handles mixed types)
        plot_matrix = results.astype(np.float64)

    # Row and column labels
    col_labels = [f"model_{i}" for i in range(plot_matrix.shape[1])]
    row_labels = [d.get('Domain-Metric', f"row_{i}") for i, d in enumerate(data_list)]

    # Plot heatmap using seaborn
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(max(8, plot_matrix.shape[1] * 0.2), max(6, plot_matrix.shape[0] * 0.25)))
    sns.set(font_scale=0.8)

    # Row-wise min-max normalization to [0,1], ignoring NaNs
    # For each row: norm_row = (row - min_row) / (max_row - min_row)
    # If max_row == min_row (constant row), set normalized values to 0
    with np.errstate(invalid='ignore'):
        row_min = np.nanmin(plot_matrix, axis=1, keepdims=True)
        row_max = np.nanmax(plot_matrix, axis=1, keepdims=True)
    denom = row_max - row_min
    # Avoid division by zero: where denom == 0, set denom to 1 temporarily
    denom_safe = np.where(np.isfinite(denom) & (denom != 0), denom, 1.0)
    norm_matrix = (plot_matrix - row_min) / denom_safe
    # For rows where denom was zero (constant rows), set normalized values to 0
    const_rows = (denom == 0).squeeze(axis=1)
    if np.any(const_rows):
        norm_matrix[const_rows, :] = 0.0

    # Keep NaNs where original data had NaNs
    norm_matrix[np.isnan(plot_matrix)] = np.nan

    # Plot normalized heatmap with fixed vmin/vmax 0..1
    ax = sns.heatmap(norm_matrix, xticklabels=col_labels, yticklabels=row_labels, cmap='viridis', vmin=0.0, vmax=1.0, cbar_kws={'label': 'Normalized score (0-1)'}, linewidths=0.5, linecolor='gray')
    ax.set_xlabel('Models')
    ax.set_ylabel('Domain - Metric')
    ax.set_title('Model performance heatmap')

    plt.tight_layout()
    out_png = "./performance_based_selection/model_performance_heatmap.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved heatmap to {out_png}")
    plt.show()

    return df.to_latex(index=False)

# Example usage
folder_path = './performance_based_selection/outputs'  # Replace with your actual folder path
processed_data = process_folder(folder_path)
latex_table = convert_to_latex_modified(processed_data, folder_path)
#print(latex_table)
