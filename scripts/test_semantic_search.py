"""
Test script for semantic search functionality using pre-computed embeddings.
This script demonstrates how to perform semantic search on the inScope metadata
using the pre-computed embeddings stored in the parquet file.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from sentence_transformers import SentenceTransformer
import torch
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.generate_embeddings import load_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model mapping
MODEL_MAPPING = {
    'minilm_embeddings': 'sentence-transformers/all-MiniLM-L6-v2',
    'nomic_embed_embeddings': 'nomic-ai/nomic-embed-text-v1.5',
    'bge_m3_embeddings': 'BAAI/bge-m3',
    'gte_large_embeddings': 'Alibaba-NLP/gte-large-en-v1.5'
}

def get_english_value(item):
    """Extract English value from multilingual data structure"""
    if isinstance(item, list) and len(item) > 0:
        # Log all language tags for debugging
        language_tags = [entry.get('language_tag', '') for entry in item if isinstance(entry, dict)]
        if language_tags:
            logger.debug(f"Found language tags: {language_tags}")
        
        # First try to find exact English match
        for entry in item:
            if isinstance(entry, dict):
                lang_tag = entry.get('language_tag', '')
                if lang_tag == 'en_US' or lang_tag == 'en_GB':
                    return entry.get('value', '')
        
        # If no exact match, try any English variant
        for entry in item:
            if isinstance(entry, dict):
                lang_tag = entry.get('language_tag', '')
                if lang_tag.startswith('en_'):
                    return entry.get('value', '')
    
    # If item is a string, assume it's English
    if isinstance(item, str):
        return item
        
    return ''

def encode_query(query, model_name):
    """Encode the query text using the specified model"""
    try:
        # Load model with trust_remote_code for models that need it
        if model_name in ['nomic_embed_embeddings', 'gte_large_embeddings']:
            model = SentenceTransformer(MODEL_MAPPING[model_name], trust_remote_code=True)
        else:
            model = SentenceTransformer(MODEL_MAPPING[model_name])
            
        # Force CPU for Alibaba model to avoid memory issues
        if model_name == 'gte_large_embeddings':
            device = 'cpu'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
            
        query_embedding = model.encode(query, device=device)
        return query_embedding
    except Exception as e:
        logger.error(f"Error encoding query: {e}")
        return None

def find_similar_products(query, df, model_name='gte_large_embeddings', top_k=5, min_similarity=0.5):
    """
    Find similar products using semantic search
    
    Args:
        query (str): The search query
        model_name (str): Which embedding model to use
        top_k (int): Number of results to return
        min_similarity (float): Minimum similarity score to include
    """
    try:
        # Get the embedding column name
        embedding_col = f"{model_name}"
        if embedding_col not in df.columns:
            logger.error(f"Embedding column {embedding_col} not found")
            return None
            
        # Encode the query
        query_embedding = encode_query(query, model_name)
        if query_embedding is None:
            return None
            
        # Convert embeddings to numpy array for faster computation
        embeddings = np.array(df[embedding_col].tolist())
        
        # Compute cosine similarity
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Get top k results with minimum similarity
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity < min_similarity:
                continue
                
            # Get values from the same columns used for embedding generation
            result = {
                'item_name': df.iloc[idx]['item_name_flat'],
                'brand': df.iloc[idx]['brand_flat'],
                'product_type': df.iloc[idx]['product_type_flat'],
                'bullet_points': df.iloc[idx]['bullet_point_flat'],
                'keywords': df.iloc[idx]['item_keywords_flat'],
                'node': df.iloc[idx]['node_flat'],
                'similarity_score': similarity,
                'price': df.iloc[idx].get('price', 'N/A')
            }
            
            # Skip if no item name found
            if not result['item_name']:
                continue
                
            results.append(result)
            
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return None

def main():
    # Load the data
    df = load_embeddings()
    if df is None:
        return
        
    # Example search queries
    queries = [
        "wireless noise cancelling headphones",
        "professional DSLR camera",
        "organic cotton t-shirt",
        "gaming laptop with high performance"
    ]
    
    # Try each embedding model
    models = ['minilm_embeddings', 'nomic_embed_embeddings', 
              'bge_m3_embeddings', 'gte_large_embeddings']
              
    for query in queries:
        print(f"\nSearching for: {query}")
        print("-" * 50)
        
        for model in models:
            print(f"\nUsing {model}:")
            results = find_similar_products(query, df, model, min_similarity=0.5)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['item_name']}")
                    if result['brand']:
                        print(f"   Brand: {result['brand']}")
                    if result['price'] != 'N/A':
                        print(f"   Price: ${result['price']:.2f}")
                    print(f"   Similarity Score: {result['similarity_score']:.4f}")
                    print()

if __name__ == "__main__":
    main() 