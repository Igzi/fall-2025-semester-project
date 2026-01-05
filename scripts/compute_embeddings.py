#!/usr/bin/env python3
"""
Compute embeddings for validation dataset from config_large_flat.json
and save them to embeddings/validation_embeddings.npy
"""

import json
import numpy as np
from InstructorEmbedding import INSTRUCTOR
from tqdm import tqdm
import os

def main():
    # Configuration
    config_path = 'dataset/config_large_flat.json'
    output_path = 'embeddings/validation_embeddings.npy'
    instruction = "Represent the sentence for similar task retrieval: "
    
    print("Loading INSTRUCTOR model...")
    model = INSTRUCTOR('Styxxxx/lora_retriever')
    
    print(f"Loading dataset from {config_path}...")
    with open(config_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Processing {len(dataset)} samples...")
    
    # Prepare input texts for embedding
    texts_to_embed = []
    for sample in tqdm(dataset, desc="Preparing texts"):
        input_text = sample['inputs']
        texts_to_embed.append([instruction, input_text])
    
    # Compute embeddings in batches for efficiency
    print("Computing embeddings...")
    batch_size = 32
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding batches"):
        batch = texts_to_embed[i:i + batch_size]
        embeddings = model.encode(batch)
        all_embeddings.append(embeddings)
    
    # Stack all embeddings into a single array
    all_embeddings = np.vstack(all_embeddings)
    
    print(f"Computed embeddings shape: {all_embeddings.shape}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save embeddings
    print(f"Saving embeddings to {output_path}...")
    np.save(output_path, all_embeddings)
    
    print("Done!")
    print(f"Embeddings saved to: {output_path}")
    print(f"Shape: {all_embeddings.shape}")
    print(f"Dtype: {all_embeddings.dtype}")

if __name__ == "__main__":
    main()
