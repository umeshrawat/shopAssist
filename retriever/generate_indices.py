import pandas as pd
import numpy as np
import faiss
import os
import sys
from pathlib import Path

def is_kaggle():
    """Check if running in Kaggle environment"""
    return os.path.exists('/kaggle/input')

def is_colab():
    """Check if running in Google Colab environment"""
    return 'google.colab' in sys.modules

def generate_faiss_index(embeddings, output_path):
    """
    Generate a FAISS index from embeddings and save it to disk.
    
    Args:
        embeddings (np.ndarray): Embeddings to index
        output_path (str): Path to save the FAISS index
    """
    # Normalize embeddings
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save the index
    faiss.write_index(index, output_path)
    print(f"FAISS index saved to {output_path}")

def main():
    # Define models
    models = {
        "minilm": "all-MiniLM-L6-v2",
        "nomic_embed": "nomic-ai/nomic-embed-text-v1.5",
        "bge_m3": "BAAI/bge-m3",
        "gte_large": "Alibaba-NLP/gte-large-en-v1.5"
    }
    
    # Determine the environment and set paths
    if is_kaggle():
        data_dir = "/kaggle/working"
    elif is_colab():
        data_dir = "/content"
    else:
        data_dir = "data"
    
    # Read the parquet file with embeddings
    parquet_path = os.path.join(data_dir, "inScopeMetadata_with_embeddings.parquet")
    df = pd.read_parquet(parquet_path)
    
    # Generate indices for each model
    for model_key, model_name in models.items():
        embedding_column = f"{model_key}_embeddings"
        if embedding_column in df.columns:
            embeddings = np.vstack(df[embedding_column].values)
            output_path = os.path.join(data_dir, f"faiss_index_{model_key}.index")
            generate_faiss_index(embeddings, output_path)
        else:
            print(f"Embeddings for {model_name} not found in the parquet file.")

if __name__ == "__main__":
    main() 