"""
Script to generate embeddings for text data using various sentence transformer models.
This script processes data in chunks to manage memory usage efficiently.

Model Evaluation Strategy:
1. all-MiniLM-L6-v2 (Baseline)
   - Fast & lightweight
   - Good for quick benchmark on retrieval accuracy, latency, and memory
   - Best for short product names and lean metadata
   - Suitable for CPU execution

2. nomic-ai/nomic-embed-text-v1.5 (Primary Candidate)
   - Handles long context (8192 tokens)
   - Instruction-tuned
   - Best balance of performance + cost
   - Requires "search_document: " prefix

3. BAAI/bge-m3 (Hybrid-capable)
   - Excellent for multi-modal and hybrid RAG
   - Strong candidate for final deployment
   - Supports dense + sparse + multi-vector

4. Alibaba-NLP/gte-large-en-v1.5 (Top dense retriever)
   - Consistent performer on MTEB benchmarks
   - Long input support
   - Fast inference
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import gc
import os
import sys
import concurrent.futures

def is_kaggle():
    """Check if running in Kaggle environment"""
    return os.path.exists('/kaggle/input')

def is_colab():
    """Check if running in Google Colab environment"""
    return 'google.colab' in sys.modules

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def add_embedding_input_column(df, output_column='embedding_input', max_kw_words=100):
    """
    Builds a text column by concatenating key metadata fields.
    
    Args:
        df (pd.DataFrame): Input DataFrame with flattened metadata
        output_column (str): Name of the column to create
        max_kw_words (int): Max words to include from item_keywords_flat
    
    Returns:
        pd.DataFrame: The same DataFrame with a new embedding_input column
    """
    def build_input(row):
        parts = [
            str(row.get('item_name_flat', '')),
            str(row.get('brand_flat', '')),
            str(row.get('product_type_flat', '')),
            str(row.get('bullet_point_flat', '')),
            " ".join(str(row.get('item_keywords_flat', '')).split()[:max_kw_words]),
            str(row.get('node_flat', ''))
        ]
        return " ".join(p for p in parts if p and p != 'nan')

    df[output_column] = df.apply(build_input, axis=1)
    return df

def generate_embeddings_parallel(df, text_column, model_name, output_column, prefix='', normalize=True, chunk_size=4, trust_remote_code=False):
    """
    Generate embeddings for text data in parallel to manage memory usage.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the column containing text to embed
        model_name (str): Name of the sentence transformer model to use
        output_column (str): Name of the column to store embeddings
        prefix (str): Optional prefix to add to text before embedding
        normalize (bool): Whether to normalize embeddings
        chunk_size (int): Number of texts to process at once
        trust_remote_code (bool): Whether to trust remote code when loading models
    """
    print(f"Loading model: {model_name}")
    device = get_device()
    print(f"Using device: {device}")
    
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    model.to(device)
    
    texts = df[text_column].tolist()
    if prefix:
        texts = [prefix + text for text in texts]
    
    embeddings = []
    total_chunks = (len(texts) + chunk_size - 1) // chunk_size
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{total_chunks}")
            futures.append(executor.submit(model.encode, chunk, normalize_embeddings=normalize))
        
        for future in concurrent.futures.as_completed(futures):
            chunk_embeddings = future.result()
            embeddings.extend(chunk_embeddings)
    
    df[output_column] = embeddings
    return df

def main():
    # Print environment info
    print(f"Running in Kaggle environment: {is_kaggle()}")
    print(f"Running in Colab environment: {is_colab()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Ensure data directory exists
    if is_kaggle():
        data_dir = "/kaggle/working"
    elif is_colab():
        data_dir = "/content"
    else:
        data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Read the parquet file
    print("Reading parquet file...")
    if is_kaggle():
        inScopeMetadata = pd.read_parquet("/kaggle/input/english-language-abo-metadata/inScopeMetadata_flattened.parquet")
    elif is_colab():
        inScopeMetadata = pd.read_parquet("/content/inScopeMetadata_flattened.parquet")
    else:
        inScopeMetadata = pd.read_parquet(os.path.join(data_dir, "inScopeMetadata_flattened.parquet"))
    
    # Create embedding input
    print("Creating embedding input...")
    inScopeMetadata = add_embedding_input_column(inScopeMetadata)
    
    # Define models based on environment
    if is_kaggle() or is_colab():
        print("Using GPU-optimized configuration...")
        models = {
            "minilm": {
                "name": "all-MiniLM-L6-v2",
                "trust_remote_code": False,
                "chunk_size": 32,  # Larger chunks for GPU
                "prefix": ""
            },
            "nomic_embed": {
                "name": "nomic-ai/nomic-embed-text-v1.5",
                "trust_remote_code": True,
                "chunk_size": 16,  # Larger chunks for GPU
                "prefix": "search_document: "
            },
            "bge_m3": {
                "name": "BAAI/bge-m3",
                "trust_remote_code": False,
                "chunk_size": 16,  # Larger chunks for GPU
                "prefix": ""
            },
            "gte_large": {
                "name": "Alibaba-NLP/gte-large-en-v1.5",
                "trust_remote_code": False,
                "chunk_size": 16,  # Larger chunks for GPU
                "prefix": ""
            }
        }
    else:
        # Local configuration (MacBook-friendly)
        print("Using local configuration (MiniLM only)...")
        models = {
            "minilm": {
                "name": "all-MiniLM-L6-v2",
                "trust_remote_code": False,
                "chunk_size": 8,
                "prefix": ""
            }
        }
    
    # Check for existing embeddings in the target parquet file
    target_parquet_path = os.path.join(data_dir, "inScopeMetadata_with_embeddings.parquet")
    if os.path.exists(target_parquet_path):
        target_metadata = pd.read_parquet(target_parquet_path)
    else:
        target_metadata = pd.DataFrame()
    
    for model_key, model_info in models.items():
        embedding_column = f"{model_key}_embeddings"
        if embedding_column in target_metadata.columns:
            print(f"Embeddings for {model_info['name']} already exist in the target parquet. Skipping...")
            continue
        
        print(f"\nGenerating embeddings using {model_info['name']}")
        inScopeMetadata = generate_embeddings_parallel(
            inScopeMetadata,
            text_column="embedding_input",
            model_name=model_info['name'],
            output_column=embedding_column,
            chunk_size=model_info['chunk_size'],
            prefix=model_info['prefix'],
            trust_remote_code=model_info['trust_remote_code']
        )
        
        # Save embeddings back to the parquet file
        output_file = os.path.join(data_dir, "inScopeMetadata_with_embeddings.parquet")
        inScopeMetadata.to_parquet(output_file)
        print(f"Embeddings for {model_info['name']} saved to {output_file}")
    
    print("Done! All embeddings saved to the parquet file.")

if __name__ == "__main__":
    main() 