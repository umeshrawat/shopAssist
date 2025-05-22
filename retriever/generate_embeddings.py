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
import threading
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import logging
from pathlib import Path
import platform
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_kaggle():
    """Check if running in Kaggle environment"""
    return os.path.exists('/kaggle/input')

def is_colab():
    """Check if running in Google Colab environment"""
    return 'google.colab' in sys.modules

def is_aws_ec2():
    """Check if running on AWS EC2"""
    try:
        with open('/sys/hypervisor/uuid', 'r') as f:
            return 'ec2' in f.read().lower()
    except:
        return False

def is_gpu_available():
    """Check if GPU is available and return device info"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        for i in range(gpu_count):
            gpu_info.append({
                'name': torch.cuda.get_device_name(i),
                'memory': torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
            })
        return True, gpu_info
    return False, None

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

def get_optimal_chunk_size(model_name, is_gpu, gpu_memory=None):
    """Get optimal chunk size based on model and hardware"""
    if is_gpu and gpu_memory:
        # Adjust chunk size based on GPU memory
        if gpu_memory >= 16:  # For GPUs with 16GB+ memory
            return {
                'all-MiniLM-L6-v2': 512,
                'nomic-ai/nomic-embed-text-v1.5': 256,
                'BAAI/bge-m3': 128,
                'Alibaba-NLP/gte-large-en-v1.5': 64
            }.get(model_name, 128)
        else:  # For GPUs with less memory
            return {
                'all-MiniLM-L6-v2': 256,
                'nomic-ai/nomic-embed-text-v1.5': 128,
                'BAAI/bge-m3': 64,
                'Alibaba-NLP/gte-large-en-v1.5': 32
            }.get(model_name, 64)
    else:
        # CPU settings
        return {
            'all-MiniLM-L6-v2': 128,
            'nomic-ai/nomic-embed-text-v1.5': 64,
            'BAAI/bge-m3': 32,
            'Alibaba-NLP/gte-large-en-v1.5': 16
        }.get(model_name, 32)

def generate_embeddings_parallel(texts, model_name, chunk_size=32, max_workers=4):
    """Generate embeddings in parallel using the specified model"""
    try:
        # Load model with GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(model_name, device=device)
        
        # Convert texts to list if it's a numpy array
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        
        # Process in chunks
        embeddings = []
        for i in tqdm(range(0, len(texts), chunk_size), desc=f"Generating embeddings with {model_name}"):
            chunk = texts[i:i + chunk_size]
            # Convert chunk to list if it's a numpy array
            if isinstance(chunk, np.ndarray):
                chunk = chunk.tolist()
            chunk_embeddings = model.encode(chunk, show_progress_bar=False)
            embeddings.extend(chunk_embeddings)
            
            # Clear GPU memory after each chunk
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Clear memory
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings with {model_name}: {str(e)}")
        return None

def main():
    is_kaggle_env = is_kaggle()
    is_colab_env = is_colab()
    print(f"Running in Kaggle environment: {is_kaggle_env}")
    print(f"Running in Colab environment: {is_colab_env}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check for Colab first, then Kaggle
    if is_colab_env:
        data_dir = "data"  # Use the 'data' folder for Colab
    elif is_kaggle_env:
        data_dir = "/kaggle/working"
    else:
        print("This script is intended to run on Kaggle or Colab only.")
        return
    os.makedirs(data_dir, exist_ok=True)

    # Read the parquet file
    print("Reading parquet file...")
    try:
        if is_colab_env:
            parquet_path = "/content/shopAssist/data/inScopeMetadata_flattened.parquet"
        elif is_kaggle_env:
            parquet_path = "/kaggle/input/english-language-abo-metadata/inScopeMetadata_flattened.parquet"
        else:
            parquet_path = os.path.join(data_dir, "inScopeMetadata_flattened.parquet")

        if not os.path.exists(parquet_path):
            print(f"Error: Parquet file not found at {parquet_path}")
            return

        inScopeMetadata = pd.read_parquet(parquet_path)
        print(f"Successfully read {len(inScopeMetadata)} rows from {parquet_path}")

    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    # Create embedding input
    print("Creating embedding input...")
    inScopeMetadata = add_embedding_input_column(inScopeMetadata)
    print("Embedding input column created.")

    # Use all four models, GPU-optimized chunk sizes
    print("Using GPU-optimized configuration (all models)...")
    models = {
        "minilm": {
            "name": "all-MiniLM-L6-v2",
            "trust_remote_code": False,
            "chunk_size": get_optimal_chunk_size('all-MiniLM-L6-v2', torch.cuda.is_available(),
                torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None),
            "prefix": ""
        },
        "nomic_embed": {
            "name": "nomic-ai/nomic-embed-text-v1.5",
            "trust_remote_code": True,
            "chunk_size": get_optimal_chunk_size('nomic-ai/nomic-embed-text-v1.5', torch.cuda.is_available(),
                torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None),
            "prefix": "search_document: "
        },
        "bge_m3": {
            "name": "BAAI/bge-m3",
            "trust_remote_code": False,
            "chunk_size": get_optimal_chunk_size('BAAI/bge-m3', torch.cuda.is_available(),
                torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None),
            "prefix": ""
        },
        "gte_large": {
            "name": "Alibaba-NLP/gte-large-en-v1.5",
            "trust_remote_code": False,
            "chunk_size": get_optimal_chunk_size('Alibaba-NLP/gte-large-en-v1.5', torch.cuda.is_available(),
                torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None),
            "prefix": ""
        }
    }

    target_parquet_path = os.path.join(data_dir, "inScopeMetadata_with_embeddings.parquet")
    if os.path.exists(target_parquet_path):
        print(f"Found existing target parquet file at {target_parquet_path}. Reading...")
        try:
            target_metadata = pd.read_parquet(target_parquet_path)
            print(f"Successfully read {len(target_metadata)} rows from target parquet.")
        except Exception as e:
            print(f"Error reading existing target parquet: {e}")
            target_metadata = pd.DataFrame()
            print("Starting with an empty target DataFrame.")
    else:
        print(f"No existing target parquet file found at {target_parquet_path}. Creating a new one.")
        target_metadata = pd.DataFrame()

    if not target_metadata.empty:
        inScopeMetadata = inScopeMetadata.merge(target_metadata, left_index=True, right_index=True, how='left', suffixes=('', '_existing'))
        print("Merged with existing target parquet.")

    for model_key, model_info in models.items():
        embedding_column = f"{model_key}_embeddings"
        existing_embedding_column = f"{embedding_column}_existing"

        # Check if existing embeddings are valid (not all None/NaN)
        if existing_embedding_column in inScopeMetadata.columns:
            existing_embeddings = inScopeMetadata[existing_embedding_column]
            if not existing_embeddings.isna().all():
                print(f"Embeddings for {model_info['name']} ({embedding_column}) already exist and are not all missing. Skipping...")
                inScopeMetadata[embedding_column] = inScopeMetadata[existing_embedding_column]
                inScopeMetadata = inScopeMetadata.drop(columns=[existing_embedding_column])
                continue
            else:
                inScopeMetadata = inScopeMetadata.drop(columns=[existing_embedding_column])
                print(f"Found existing column {embedding_column}_existing but it was all null. Proceeding with generation.")

        print(f"\nGenerating embeddings using {model_info['name']}")

        if 'embedding_input' not in inScopeMetadata.columns:
            print("Error: 'embedding_input' column is missing. Cannot generate embeddings.")
            inScopeMetadata = add_embedding_input_column(inScopeMetadata)
            print("Re-created 'embedding_input' column.")

        embeddings = generate_embeddings_parallel(
            inScopeMetadata['embedding_input'].values,
            model_info['name'],
            chunk_size=model_info['chunk_size'],
            max_workers=4
        )

        if embeddings is not None:
            inScopeMetadata[embedding_column] = embeddings.tolist()
            columns_to_save = [col for col in inScopeMetadata.columns if not col.endswith('_existing')]
            inScopeMetadata[columns_to_save].to_parquet(target_parquet_path)
            print(f"Embeddings for {model_info['name']} saved to {target_parquet_path}")
        else:
            print(f"Failed to generate embeddings for {model_info['name']}")

    print("Done! All embeddings saved to the parquet file.")

if __name__ == "__main__":
    main()