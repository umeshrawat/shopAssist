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
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Force output to stdout
    ]
)
logger = logging.getLogger(__name__)

# Add a function to force flush stdout
def force_flush():
    sys.stdout.flush()
    sys.stderr.flush()

def is_kaggle():
    """Check if running in Kaggle environment"""
    return os.path.exists('/kaggle/input')

def is_colab():
    """Check if running in Google Colab environment"""
    return 'google.colab' in sys.modules

def is_aws_ec2():
    """Check if running on AWS EC2 using the instance metadata service."""
    try:
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
            instance_id_url = "http://169.254.169.254/latest/meta-data/instance-id"
            instance_id_response = requests.get(instance_id_url, timeout=1)
            return instance_id_response.status_code == 200

    except requests.exceptions.RequestException:
        return False
    except Exception as e:
        print(f"Error during EC2 detection: {e}")
        return False

def get_available_memory():
    """Get available system memory in GB"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        return psutil.virtual_memory().available / 1024**3

def add_embedding_input_column(df, output_column='embedding_input', max_kw_words=100):
    """Builds a text column by concatenating key metadata fields."""
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
    """Get optimal chunk size based on model, hardware, and available memory"""
    available_memory = get_available_memory()
    
    # Base chunk sizes for different environments
    if is_kaggle():
        # Kaggle typically has limited memory
        base_sizes = {
            'all-MiniLM-L6-v2': 64,
            'nomic-ai/nomic-embed-text-v1.5': 32,
            'BAAI/bge-m3': 16,
            'Alibaba-NLP/gte-large-en-v1.5': 8
        }
    elif is_colab():
        # Colab has more memory but still needs to be conservative
        base_sizes = {
            'all-MiniLM-L6-v2': 128,
            'nomic-ai/nomic-embed-text-v1.5': 64,
            'BAAI/bge-m3': 32,
            'Alibaba-NLP/gte-large-en-v1.5': 16
        }
    else:
        # Default sizes for other environments
        base_sizes = {
            'all-MiniLM-L6-v2': 256,
            'nomic-ai/nomic-embed-text-v1.5': 128,
            'BAAI/bge-m3': 64,
            'Alibaba-NLP/gte-large-en-v1.5': 32
        }
    
    # Adjust based on available memory
    if available_memory < 4:  # Less than 4GB available
        return {k: max(1, v // 4) for k, v in base_sizes.items()}.get(model_name, 8)
    elif available_memory < 8:  # Less than 8GB available
        return {k: max(1, v // 2) for k, v in base_sizes.items()}.get(model_name, 16)
    else:
        return base_sizes.get(model_name, 32)

def get_checkpoint_paths(data_dir, model_key):
    """Get paths for checkpoint files in a platform-agnostic way"""
    checkpoint_dir = os.path.join(data_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return {
        "parquet": os.path.join(checkpoint_dir, f"checkpoint_{model_key}.parquet"),
        "temp": os.path.join(checkpoint_dir, f"temp_embeddings_{model_key}.npy")
    }

def save_checkpoint(df, model_key, data_dir):
    """Save checkpoint after each model's embeddings are generated"""
    paths = get_checkpoint_paths(data_dir, model_key)
    try:
        # Save parquet checkpoint
        df.to_parquet(paths["parquet"])
        logger.info(f"Checkpoint saved to {paths['parquet']}")
        return True
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        return False

def load_latest_checkpoint(data_dir, model_key):
    """Load the latest checkpoint for a specific model"""
    try:
        paths = get_checkpoint_paths(data_dir, model_key)
        if os.path.exists(paths["parquet"]):
            logger.info(f"Loading checkpoint from {paths['parquet']}")
            return pd.read_parquet(paths["parquet"])
        return None
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return None

def generate_embeddings_parallel(texts, model_name, chunk_size=32, max_workers=4, trust_remote_code=False):
    """Generate embeddings in parallel using the specified model"""
    try:
        # Load model with GPU if available, respecting trust_remote_code
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        force_flush()
        
        model = SentenceTransformer(model_name, device=device, trust_remote_code=trust_remote_code)
        logger.info(f"Model loaded: {model_name}")
        force_flush()
        
        # Convert texts to list if it's a numpy array
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        
        # Process in chunks with memory management
        embeddings = []
        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
        logger.info(f"Processing {len(texts)} texts in {total_chunks} chunks")
        force_flush()
        
        # Get checkpoint paths
        model_key = model_name.replace('/', '_')
        paths = get_checkpoint_paths(os.getcwd(), model_key)
        
        # Try to load existing temp embeddings if they exist
        if os.path.exists(paths["temp"]):
            try:
                embeddings = np.load(paths["temp"]).tolist()
                logger.info(f"Loaded existing temp embeddings from {paths['temp']}")
                force_flush()
            except Exception as e:
                logger.error(f"Error loading temp embeddings: {e}")
                embeddings = []
        
        for i in tqdm(range(0, len(texts), chunk_size), desc=f"Generating embeddings with {model_name}"):
            chunk = texts[i:i + chunk_size]
            if isinstance(chunk, np.ndarray):
                chunk = chunk.tolist()
            
            # Add prefix for nomic model if needed
            if model_name == 'nomic-ai/nomic-embed-text-v1.5':
                chunk = [f"search_document: {text}" for text in chunk]
            
            chunk_embeddings = model.encode(chunk, show_progress_bar=False)
            embeddings.extend(chunk_embeddings)
            
            # Log progress
            if (i // chunk_size) % max(1, total_chunks // 10) == 0:
                logger.info(f"Progress: {i}/{len(texts)} texts processed")
                force_flush()
            
            # Aggressive memory cleanup after each chunk
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Save intermediate results every 10% progress
            if (i // chunk_size) % max(1, total_chunks // 10) == 0:
                temp_embeddings = np.array(embeddings)
                np.save(paths["temp"], temp_embeddings)
                logger.info(f"Saved intermediate checkpoint at {i}/{len(texts)} texts")
                force_flush()
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        logger.info(f"Completed processing all {len(texts)} texts")
        force_flush()
        
        # Cleanup
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings with {model_name}: {str(e)}")
        force_flush()
        return None

def main():
    # Force flush at start
    force_flush()
    
    is_kaggle_env = is_kaggle()
    is_colab_env = is_colab()
    is_ec2_env = is_aws_ec2()
    
    logger.info(f"Running in Kaggle environment: {is_kaggle_env}")
    logger.info(f"Running in Colab environment: {is_colab_env}")
    logger.info(f"Running in EC2 environment: {is_ec2_env}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    force_flush()
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        force_flush()
    
    # Determine data directory based on environment
    if is_colab_env:
        data_dir = "/content/shopAssist/data"
    elif is_kaggle_env:
        data_dir = "/kaggle/working"
    elif is_ec2_env:
        data_dir = "data"
    else:
        data_dir = "data"  # Default to local data directory
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Read the parquet file
    logger.info("Reading source parquet file...")
    try:
        if is_colab_env:
            parquet_path = "/content/shopAssist/data/inScopeMetadata_flattened.parquet"
        elif is_kaggle_env:
            parquet_path = "/kaggle/input/english-language-abo-metadata/inScopeMetadata_flattened.parquet"
        else:
            parquet_path = os.path.join(data_dir, "inScopeMetadata_flattened.parquet")

        if not os.path.exists(parquet_path):
            logger.error(f"Error: Source parquet file not found at {parquet_path}")
            return

        inScopeMetadata = pd.read_parquet(parquet_path)
        logger.info(f"Successfully read {len(inScopeMetadata)} rows from {parquet_path}")

    except Exception as e:
        logger.error(f"Error reading source parquet file: {e}")
        return

    # Create embedding input
    logger.info("Creating embedding input...")
    inScopeMetadata = add_embedding_input_column(inScopeMetadata)
    logger.info("Embedding input column created.")

    # Define models to use with environment-specific settings
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
        }
    }
    
    # Only include larger models if we have enough memory
    available_memory = get_available_memory()
    if available_memory >= 8:  # Only include larger models if we have at least 8GB
        models.update({
            "bge_m3": {
                "name": "BAAI/bge-m3",
                "trust_remote_code": False,
                "prefix": ""
            },
            "gte_large": {
                "name": "Alibaba-NLP/gte-large-en-v1.5",
                "trust_remote_code": True,
                "prefix": ""
            }
        })

    target_parquet_path = os.path.join(data_dir, "inScopeMetadata_with_embeddings.parquet")
    existing_embeddings_df = pd.DataFrame()

    if os.path.exists(target_parquet_path):
        logger.info(f"Found existing target parquet file at {target_parquet_path}. Reading...")
        try:
            existing_embeddings_df = pd.read_parquet(target_parquet_path)
            logger.info(f"Successfully read {len(existing_embeddings_df)} rows from target parquet.")
            if not existing_embeddings_df.index.equals(inScopeMetadata.index):
                logger.warning("Indices of existing embeddings file do not match source data. Reindexing...")
                existing_embeddings_df = existing_embeddings_df.reindex(inScopeMetadata.index)
        except Exception as e:
            logger.error(f"Error reading existing target parquet: {e}. Starting with an empty DataFrame.")
            existing_embeddings_df = pd.DataFrame()

    # Get GPU info for chunk size calculation
    is_gpu = torch.cuda.is_available()
    gpu_memory = get_available_memory() if is_gpu else None

    for model_key, model_info in models.items():
        embedding_column = f"{model_key}_embeddings"
        
        # Check for existing embeddings
        if embedding_column in existing_embeddings_df.columns and not existing_embeddings_df[embedding_column].isna().all():
            logger.info(f"Embeddings for {model_info['name']} already exist. Skipping generation.")
            inScopeMetadata[embedding_column] = existing_embeddings_df[embedding_column]
            continue

        # Try to load from checkpoint
        checkpoint_df = load_latest_checkpoint(data_dir, model_key)
        if checkpoint_df is not None and embedding_column in checkpoint_df.columns:
            logger.info(f"Resuming from checkpoint for {model_info['name']}")
            inScopeMetadata[embedding_column] = checkpoint_df[embedding_column]
            continue

        logger.info(f"\nGenerating embeddings using {model_info['name']}. Column: {embedding_column}")

        if 'embedding_input' not in inScopeMetadata.columns:
            logger.error("'embedding_input' column is missing. Recreating...")
            inScopeMetadata = add_embedding_input_column(inScopeMetadata)

        # Get optimal chunk size
        current_chunk_size = get_optimal_chunk_size(
            model_info['name'],
            is_gpu,
            gpu_memory
        )
        logger.info(f"Using chunk size: {current_chunk_size}")

        embeddings = generate_embeddings_parallel(
            inScopeMetadata['embedding_input'].values,
            model_info['name'],
            chunk_size=current_chunk_size,
            max_workers=4,
            trust_remote_code=model_info['trust_remote_code']
        )

        if embeddings is not None:
            inScopeMetadata[embedding_column] = embeddings.tolist()
            logger.info(f"Generated embeddings for {model_info['name']}")
            
            # Save checkpoint after each model
            save_checkpoint(inScopeMetadata, model_key, data_dir)
        else:
            logger.error(f"Failed to generate embeddings for {model_info['name']}")

    # Save the final DataFrame
    logger.info(f"Saving updated DataFrame to {target_parquet_path}...")
    try:
        inScopeMetadata.to_parquet(target_parquet_path)
        logger.info("Updated DataFrame saved successfully.")
    except Exception as e:
        logger.error(f"Error saving updated parquet file: {e}")

    logger.info("Done!")

if __name__ == "__main__":
    main()