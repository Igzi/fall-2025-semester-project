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
from sentence_transformers import SentenceTransformer

def main():
    # Configuration
    config_path = 'dataset/config2_flat.json'
    output_path = 'embeddings/validation_embeddings_e5_large.npy'
    instruction = "Represent the sentence for similar task retrieval: "
    
    model = SentenceTransformer('intfloat/e5-large-v2')
    
    print(f"Loading dataset from {config_path}...")
    with open(config_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Processing {len(dataset)} samples...")
    
    # Prepare input texts for embedding (concatenate instruction + text)
    texts_to_embed = []
    for sample in tqdm(dataset, desc="Preparing texts"):
        input_text = sample['inputs']
        texts_to_embed.append(input_text)
    
    # Compute embeddings in batches for efficiency
    print("Computing embeddings...")
    batch_size = 32
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding batches"):
        batch = texts_to_embed[i:i + batch_size]
        embeddings = model.encode(batch, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embeddings)
    
    # Stack all embeddings into a single array
    all_embeddings = np.vstack(all_embeddings)

    print(f"Computed embeddings shape: {all_embeddings.shape}")

    # Build per-task centroids by averaging embeddings for each unique task
    labels = []
    for sample in dataset:
        lab = sample.get('model_name') or sample.get('task') or sample.get('model')
        labels.append(lab)

    # preserve first-seen order of tasks
    unique_labels = []
    label_to_idxs = {}
    for i, lab in enumerate(labels):
        if lab not in label_to_idxs:
            label_to_idxs[lab] = []
            unique_labels.append(lab)
        label_to_idxs[lab].append(i)

    centroids = []
    for lab in unique_labels:
        idxs = label_to_idxs[lab]
        emb_mean = all_embeddings[idxs].mean(axis=0)
        centroids.append(emb_mean)

    centroids = np.vstack(centroids)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save centroids and labels
    print(f"Saving {centroids.shape[0]} task centroids to {output_path}...")
    np.save(output_path, centroids)
    labels_path = output_path.replace('.npy', '_labels.json')
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(unique_labels, f, indent=2)

    print("Done!")
    print(f"Centroids saved to: {output_path}")
    print(f"Labels saved to: {labels_path}")
    print(f"Centroids shape: {centroids.shape}")
    print(f"Dtype: {centroids.dtype}")

if __name__ == "__main__":
    main()
