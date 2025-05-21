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
    Handles potential failures in chunks by assigning None for failed rows.

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

    try:
        # SentenceTransformer is generally preferred for semantic search tasks
        model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
        model.to(device)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        # Assign None to the output column for all rows if model loading fails
        df[output_column] = pd.Series([None] * len(df), index=df.index)
        return df # Return the DataFrame with None embeddings

    texts = df[text_column].tolist()
    if prefix:
        texts = [prefix + text for text in texts]

    # Initialize embeddings list with None for every row
    embeddings = [None] * len(texts)
    total_chunks = (len(texts) + chunk_size - 1) // chunk_size

    # Dictionary to store results by their starting index in the original texts list
    results = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        # Store future and the starting index of the chunk it processes
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            # Store the starting index of this chunk
            future = executor.submit(lambda c: model.encode(c, normalize_embeddings=normalize, show_progress_bar=False), chunk)
            futures[future] = i # Map future to the starting index of the chunk

        for future in concurrent.futures.as_completed(futures):
            start_index = futures[future]
            chunk_number = start_index // chunk_size + 1
            print(f"Processing chunk {chunk_number}/{total_chunks}", end='\r') # Use carriage return for progress on one line
            try:
                chunk_embeddings = future.result()
                # Convert numpy array to list if necessary
                if isinstance(chunk_embeddings, np.ndarray):
                    chunk_embeddings = chunk_embeddings.tolist()
                # Place the generated embeddings directly into the correct slice of the results list
                results[start_index] = chunk_embeddings # Store result by start index
            except Exception as exc:
                print(f'\nChunk {chunk_number}/{total_chunks} generated an exception: {exc}')
                # The corresponding entries in 'embeddings' remain None, which is the desired behavior for failed chunks.

    # Reconstruct the final embeddings list in the correct order
    for i in range(0, len(texts), chunk_size):
        if i in results:
            chunk_embeddings = results[i]
            # Ensure chunk_embeddings is a list before assignment
            if isinstance(chunk_embeddings, np.ndarray):
                chunk_embeddings = chunk_embeddings.tolist()
            embeddings[i : i + len(chunk_embeddings)] = chunk_embeddings

    # Assign the potentially partial (with None values) embeddings list to the DataFrame column
    try:
        # Convert embeddings to a list of lists if they're numpy arrays
        embeddings_list = []
        for emb in embeddings:
            if emb is not None and isinstance(emb, np.ndarray):
                embeddings_list.append(emb.tolist())
            else:
                embeddings_list.append(emb)
        df[output_column] = pd.Series(embeddings_list, index=df.index)
    except Exception as e:
        print(f"Error assigning embeddings to DataFrame column {output_column}: {e}")
        df[output_column] = pd.Series([None] * len(df), index=df.index)

    # Clean up GPU memory
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    # Count non-None embeddings
    non_none_count = sum(1 for emb in embeddings if emb is not None)
    print(f"\nFinished generating embeddings for {model_name}. Rows processed: {non_none_count}/{len(df)}")

    return df

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
            "chunk_size": 32,  # Larger chunks for GPU
            "prefix": ""
        },
        "nomic_embed": {
            "name": "nomic-ai/nomic-embed-text-v1.5",
            "trust_remote_code": True,
            "chunk_size": 8,  # Reduced chunk size to prevent OOM
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

        inScopeMetadata = generate_embeddings_parallel(
            inScopeMetadata,
            text_column="embedding_input",
            model_name=model_info['name'],
            output_column=embedding_column,
            chunk_size=model_info['chunk_size'],
            prefix=model_info['prefix'],
            trust_remote_code=model_info['trust_remote_code']
        )

        columns_to_save = [col for col in inScopeMetadata.columns if not col.endswith('_existing')]
        inScopeMetadata[columns_to_save].to_parquet(target_parquet_path)
        print(f"Embeddings for {model_info['name']} saved to {target_parquet_path}")

    print("Done! All embeddings saved to the parquet file.")

if __name__ == "__main__":
    main()