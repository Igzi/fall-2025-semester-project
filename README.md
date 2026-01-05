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

---

## 📊 Usage

### Run Evaluation

**Standard Fusion Evaluation:**
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

### 5. Analyze Results

Summarize and compare results:

```bash
python summarize_results.py
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