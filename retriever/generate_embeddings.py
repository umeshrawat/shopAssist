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
import requests

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
    """Check if running on AWS EC2 using the instance metadata service."""
    try:
        # Attempt to connect to the instance metadata service (IMDSv1 or IMDSv2)
        # First, try to get a token (IMDSv2)
        token_url = "http://169.254.169.254/latest/api/token"
        token_headers = {'X-aws-ec2-metadata-token-ttl-seconds': '21600'}
        token_response = requests.put(token_url, headers=token_headers, timeout=1)

        if token_response.status_code == 200:
            token = token_response.text
            headers = {'X-aws-ec2-metadata-token': token}
            instance_id_url = "http://169.254.169.254/latest/meta-data/instance-id"
            instance_id_response = requests.get(instance_id_url, headers=headers, timeout=1)
            return instance_id_response.status_code == 200
        elif token_response.status_code == 405:
            # Method Not Allowed, likely IMDSv1 only
            instance_id_url = "http://169.254.169.254/latest/meta-data/instance-id"
            instance_id_response = requests.get(instance_id_url, timeout=1)
            return instance_id_response.status_code == 200

    except requests.exceptions.RequestException:
        # Connection error, not on EC2 or IMDS is disabled/firewalled
        return False
    except Exception as e:
        # Other unexpected errors
        print(f"Error during EC2 detection: {e}")
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

def generate_embeddings_parallel(texts, model_name, chunk_size=32, max_workers=4, trust_remote_code=False):
    """Generate embeddings in parallel using the specified model"""
    try:
        # Load model with GPU if available, respecting trust_remote_code
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(model_name, device=device, trust_remote_code=trust_remote_code)
        
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
    is_ec2_env = is_aws_ec2()
    print(f"Running in Kaggle environment: {is_kaggle_env}")
    print(f"Running in Colab environment: {is_colab_env}")
    print(f"Running in EC2 environment: {is_ec2_env}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Determine data directory based on environment
    if is_colab_env:
        data_dir = "data"  # Use the 'data' folder for Colab
    elif is_kaggle_env:
        data_dir = "/kaggle/working"
    elif is_ec2_env:
        data_dir = "data" # Use the 'data' folder for EC2
    else:
        print("This script is intended to run on Kaggle, Colab, or EC2 only.")
        return
    os.makedirs(data_dir, exist_ok=True)

    # Read the parquet file
    print("Reading source parquet file...")
    try:
        if is_colab_env:
            parquet_path = "/content/shopAssist/data/inScopeMetadata_flattened.parquet"
        elif is_kaggle_env:
            parquet_path = "/kaggle/input/english-language-abo-metadata/inScopeMetadata_flattened.parquet"
        elif is_ec2_env:
             # Assuming the data is in a 'data' folder relative to the script's execution directory
            parquet_path = os.path.join(data_dir, "inScopeMetadata_flattened.parquet")
        else:
             # This case should ideally not be reached due to the check above
            print("Unknown environment. Cannot determine parquet file path.")
            return

        if not os.path.exists(parquet_path):
            print(f"Error: Source parquet file not found at {parquet_path}")
            return

        inScopeMetadata = pd.read_parquet(parquet_path)
        print(f"Successfully read {len(inScopeMetadata)} rows from {parquet_path}")

    except Exception as e:
        print(f"Error reading source parquet file: {e}")
        return

    # Create embedding input
    print("Creating embedding input...")
    inScopeMetadata = add_embedding_input_column(inScopeMetadata)
    print("Embedding input column created.")

    # Define models to use
    models = {
        "minilm": {
            "name": "all-MiniLM-L6-v2",
            "trust_remote_code": False,
            "prefix": ""
        },
        "nomic_embed": {
            "name": "nomic-ai/nomic-embed-text-v1.5",
            "trust_remote_code": True,
            "prefix": "search_document: "
        },
        "bge_m3": {
            "name": "BAAI/bge-m3",
            "trust_remote_code": False,
            "prefix": ""
        },
        "gte_large": {
            "name": "Alibaba-NLP/gte-large-en-v1.5",
            "trust_remote_code": False,
            "prefix": ""
        }
    }

    target_parquet_path = os.path.join(data_dir, "inScopeMetadata_with_embeddings.parquet")
    existing_embeddings_df = pd.DataFrame()

    if os.path.exists(target_parquet_path):
        print(f"Found existing target parquet file at {target_parquet_path}. Reading...")
        try:
            existing_embeddings_df = pd.read_parquet(target_parquet_path)
            print(f"Successfully read {len(existing_embeddings_df)} rows from target parquet.")
            # Ensure indices align if we are to merge or update based on index
            if not existing_embeddings_df.index.equals(inScopeMetadata.index):
                 print("Warning: Indices of existing embeddings file do not match source data. Proceeding with caution.")
                 # Attempt to reindex existing data to match source data indices
                 existing_embeddings_df = existing_embeddings_df.reindex(inScopeMetadata.index)

        except Exception as e:
            print(f"Error reading existing target parquet: {e}. Starting with an empty existing embeddings DataFrame.")
            existing_embeddings_df = pd.DataFrame()

    # Get GPU info for chunk size calculation if available
    is_gpu, gpu_info = is_gpu_available()
    gpu_memory = gpu_info[0]['memory'] if is_gpu and gpu_info else None

    for model_key, model_info in models.items():
        embedding_column = f"{model_key}_embeddings"

        # Check if embeddings for this model already exist in the target file data
        if embedding_column in existing_embeddings_df.columns and not existing_embeddings_df[embedding_column].isna().all():
             print(f"Embeddings for {model_info['name']} ({embedding_column}) already exist in the target file and are not all missing. Skipping generation.")
             # Copy the existing embeddings to the main dataframe if needed (reindex handles alignment)
             inScopeMetadata[embedding_column] = existing_embeddings_df[embedding_column]
             continue # Skip generation for this model

        print(f"\nGenerating embeddings using {model_info['name']}. Column: {embedding_column}")

        if 'embedding_input' not in inScopeMetadata.columns:
            print("Error: 'embedding_input' column is missing. Cannot generate embeddings.")
            # This should not happen if previous steps were successful, but as a safeguard
            inScopeMetadata = add_embedding_input_column(inScopeMetadata)
            print("Re-created 'embedding_input' column.")

        # Get chunk size for the current model and available hardware
        current_chunk_size = get_optimal_chunk_size(
            model_info['name'],
            is_gpu,
            gpu_memory
        )
        print(f"Using chunk size: {current_chunk_size}")

        embeddings = generate_embeddings_parallel(
            inScopeMetadata['embedding_input'].values,
            model_info['name'],
            chunk_size=current_chunk_size,
            max_workers=4,
            trust_remote_code=model_info['trust_remote_code']
        )

        if embeddings is not None:
            inScopeMetadata[embedding_column] = embeddings.tolist()
            print(f"Generated embeddings for {model_info['name']}")
        else:
            print(f"Failed to generate embeddings for {model_info['name']}")

    # Save the final DataFrame with all embeddings
    print(f"Saving updated DataFrame to {target_parquet_path}...")
    try:
        # Ensure only necessary columns are kept if the source had extra after merge attempt (shouldn't now with refactor)
        # Let's just save the inScopeMetadata which now contains original + generated/copied embeddings
        inScopeMetadata.to_parquet(target_parquet_path)
        print("Updated DataFrame saved successfully.")
    except Exception as e:
        print(f"Error saving updated parquet file: {e}")

    print("Done!")

if __name__ == "__main__":
    main()