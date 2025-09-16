import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel
import json
import numpy as np
from utils.instructor_retrieval import perform_search, initialize_index
from datasets import load_dataset
from utils.prompter import Prompter
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
import os

def get_embeddings(text_list):
    """
    Encode a list of text samples using the global embedding model.

    Parameters:
    - text_list: A list of texts to be encoded. Each element should be [instruction, text].
    """
    return global_model.encode(text_list)

config_path='./config/config2.json'
instruction = "Represent the sentence for similar task retrieval: "

with open(config_path, 'r') as file:
    models = json.load(file)

global_model = INSTRUCTOR('Styxxxx/lora_retriever')

all_model_embeddings = []
model_names = []

# Compute average embeddings for each model
for model in models:
    model_name = f"Styxxxx/llama2_7b_lora-{model['model_name']}"

    model_names.append(model_name)
    model_samples = []

    # Collect sample inputs for each model
    for sample in model['sample']:
        sample_context = sample['inputs']
        model_samples.append([instruction, sample_context])

    # Compute embeddings for the model's samples and take the mean
    embeddings = get_embeddings(model_samples)
    all_model_embeddings.append([embeddings])

# Create a FAISS index with the collected embeddings
all_model_embeddings = np.vstack(all_model_embeddings)

np.save('results/lora_retriever_embeddings.npy', all_model_embeddings)

# Generate and save embeddings using sentence-transformers/all-mpnet-base-v2
mpnet_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
all_mpnet_embeddings = []

for model in models:
    model_samples = []
    for sample in model['sample']:
        sample_context = sample['inputs']
        model_samples.append(sample_context)
    mpnet_embeddings = mpnet_model.encode(model_samples)
    all_mpnet_embeddings.append(mpnet_embeddings)

all_mpnet_embeddings = np.vstack(all_mpnet_embeddings)
np.save('results/mpnet_embeddings.npy', all_mpnet_embeddings)


