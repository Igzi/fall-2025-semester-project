# LoRAuter: Dynamic LoRA Adapter Selection and Fusion

**EPFL Semester Project - Fall 2025**  
*Author: Igor Pavlovic*

---

## 📋 Overview

This repository implements **LoRAuter**, a novel approach for dynamic selection and fusion of LoRA (Low-Rank Adaptation) adapters for Large Language Models (LLMs). The system intelligently routes inputs to the most appropriate task-specific adapters and performs weighted fusion to combine their strengths, achieving improved performance across diverse natural language processing tasks.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 24GB+ VRAM for 13B models)
- CUDA 11.7+ and cuDNN

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fall-2025-semester-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install custom PEFT library:**
   ```bash
   cd peft
   pip install -e .
   cd ..
   ```

### Required Models

The system uses pre-trained LoRA adapters from HuggingFace. Adapters are automatically downloaded from:
- Base models: `meta-llama/Llama-2-7b-hf` or `meta-llama/Llama-2-13b-hf`
- LoRA adapters: `Styxxxx/llama2_{model_size}_lora-{task_name}`
- Embedding model: `Styxxxx/lora_retriever`

### Required Files

Before running evaluations, ensure these files exist:
- **Model Performance Matrices**: `adapter_evaluation/model_performance_{7b|13b}.npy`
- **Base Router Model**: `adapter_evaluation/models/base_model.pt` and `base_model.config.json`
- **Datasets**: `dataset/combined_test.json`, `dataset/config_large_flat.json`, `dataset/config2_flat.json`

---

## 📊 Usage

### Complete Workflow

The system follows a multi-stage pipeline:

#### 1. **Generate Individual Model Outputs** (Optional)

Generate predictions from individual LoRA adapters for analysis:

```bash
python adapter_evaluation/generate_model_outputs.py
```

This creates output files in `outputs_{7b|13b}/` for each adapter.

### 2. **Generate Validation Set Results** (Optional)

Process individual adapter outputs to compute performance metrics and matrices:

```bash
python adapter_evaluation/generate_results.py
```

This script:
- Analyzes outputs from each adapter in `adapter_evaluation/outputs_{7b|13b}/`
- Calculates BLEU, ROUGE, and Exact Match scores for each model on validation tasks
- Generates the `model_performance_{7b|13b}.npy` performance matrices
- Creates comparison tables showing per-adapter performance across domains
- Used by the router to learn adapter selection strategies

#### 3. **Run Evaluation**

Evaluate the LoRAuter system on test datasets.

**Standard In-Distribution Evaluation:**
```bash
python lorauter_eval.py \
    --data_path dataset/combined_test.json \
    --res_path results/my_results.json \
    --config_path config/config_large.json \
    --lora_num 3 \
    --model_size 7b \
    --eval_type mixture
```

**Out-of-Distribution (OOD) Evaluation:**
```bash
python lorauter_eval.py \
    --data_path dataset/combined_test.json \
    --res_path results/my_results_ood.json \
    --config_path config/config_large.json \
    --lora_num 3 \
    --model_size 7b \
    --eval_type mixture \
    --ood True
```

**Parameters:**
- `data_path`: Path to test dataset JSON file
- `res_path`: Output path for evaluation results
- `config_path`: Configuration file listing available LoRA adapters
- `lora_num`: Number of top-k adapters to select and fuse (default: 3)
- `batch_size`: Batch size for inference (default: 1)
- `model_size`: LLaMA model size - `7b` or `13b`
- `eval_type`: Merging strategy - `mixture` (weighted fusion) or `selection` (best adapter only)
- `ood`: Enable out-of-distribution evaluation (excludes certain adapters)
- `best_selection`: Use performance-based selection instead of retrieval

#### 4. **Analyze Results**

Summarize and compare evaluation results across different methods:

```bash
python summarize_results.py
```

This script:
- Processes all JSON result files in the `results/` directory
- Calculates BLEU, ROUGE, and Exact Match scores by domain and task
- Generates comparison tables and statistics
- Outputs formatted results for analysis

---

## 📁 Repository Structure

```
├── lorauter_eval.py              # Main evaluation script
├── summarize_results.py          # Results analysis and reporting
├── adapter_evaluation/           # Adapter evaluation utilities
│   ├── create_base_router.py    # Create the task embeddings and the base router model
│   ├── generate_model_outputs.py # Generate individual adapter outputs
│   ├── generate_results.py      # Process outputs and compute metrics
│   ├── model_performance_*.npy  # Pre-computed performance matrices
│   └── models/                  # Router models
├── baselines/                    # Baseline method results
├── config/                       # LoRA adapter configurations
├── dataset/                      # Test and validation datasets
├── scripts/                      # Utility scripts
│   ├── compute_embeddings.py    # Pre-compute dataset embeddings
│   └── flatten_config2_to_dataset.py  # Convert config to dataset format
├── templates/                    # Prompt templates (Alpaca)
├── results/                      # Evaluation results (JSON)
└── peft/                        # Modified PEFT library
```

## 📄 License

This project includes a modified version of the PEFT library. See `peft/LICENSE` for details.

---

## 🤝 Acknowledgments

- **EPFL** for providing computational resources
- **HuggingFace** for the Transformers and PEFT libraries
- **Meta AI** for Llama-2 base models
- Baseline implementations: ARROW, LoraHub, SPECTR

---

## 📧 Contact

For questions or collaborations:
- **Author**: Igor Pavlovic
- **Email**: igor.pavlovic@epfl.ch
- **Institution**: École Polytechnique Fédérale de Lausanne (EPFL)
---