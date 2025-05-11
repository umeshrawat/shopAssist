"""
Script to generate embeddings for text data using various sentence transformer models.
This script processes data in chunks to manage memory usage efficiently.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import gc
import os

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

def generate_embeddings(df, text_column, model_name, output_column, prefix='', normalize=True, chunk_size=4, trust_remote_code=False):
    """
    Generate embeddings for text data in chunks to manage memory usage.
    
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
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    
    texts = df[text_column].tolist()
    if prefix:
        texts = [prefix + text for text in texts]
    
    embeddings = []
    total_chunks = (len(texts) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{total_chunks}")
        
        # Generate embeddings for the chunk
        chunk_embeddings = model.encode(chunk, normalize_embeddings=normalize)
        embeddings.extend(chunk_embeddings)
        
        # Clear memory
        del chunk_embeddings
        torch.cuda.empty_cache()
        gc.collect()
    
    df[output_column] = embeddings
    return df

def main():
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Read the parquet file
    print("Reading parquet file...")
    inScopeMetadata = pd.read_parquet("data/inScopeMetadata_flattened.parquet")
    
    # Create embedding input
    print("Creating embedding input...")
    inScopeMetadata = add_embedding_input_column(inScopeMetadata)
    
    # Generate embeddings for different models
    models = {
        "nomic_embed": {
            "name": "nomic-ai/nomic-embed-text-v1.5",
            "trust_remote_code": True
        },
        "bge_m3": {
            "name": "BAAI/bge-m3",
            "trust_remote_code": False
        },
        "gte_large": {
            "name": "Alibaba-NLP/gte-large-en-v1.5",
            "trust_remote_code": False
        },
        "minilm": {
            "name": "all-MiniLM-L6-v2",
            "trust_remote_code": False
        }
    }
    
    for model_key, model_info in models.items():
        print(f"\nGenerating embeddings using {model_info['name']}")
        inScopeMetadata = generate_embeddings(
            inScopeMetadata,
            text_column="embedding_input",
            model_name=model_info['name'],
            output_column=f"{model_key}_embeddings",
            chunk_size=4,
            trust_remote_code=model_info['trust_remote_code']
        )
    
    # Save results
    print("\nSaving results...")
    inScopeMetadata.to_parquet("data/inScopeMetadata_with_embeddings.parquet")
    print("Done!")

if __name__ == "__main__":
    main() 