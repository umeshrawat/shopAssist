# faiss_retriever.py
import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import faiss
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrieverError(Exception):
    """Base exception for retriever errors."""
    pass

class IndexError(RetrieverError):
    """Exception raised for index-related errors."""
    pass

class MetadataError(RetrieverError):
    """Exception raised for metadata-related errors."""
    pass

class EmbeddingError(RetrieverError):
    """Exception raised for embedding-related errors."""
    pass

class FAISSRetriever:
    """FAISS-based retriever for semantic search with text embedding capabilities.
    
    This class provides functionality for:
    1. Loading and managing a FAISS index and associated metadata
    2. Converting text queries to embeddings using Sentence Transformers
    3. Performing similarity search with deduplication
    4. Converting distances to similarity scores
    5. Batch processing of queries
    6. Caching of embeddings
    
    Example:
        ```python
        # Initialize retriever
        retriever = FAISSRetriever(
            index_path="data/faiss_index.index",
            metadata_path="data/inScopeMetadata_with_embeddings.parquet",
            model_name="all-MiniLM-L12-v2",  # Optional, for text embedding
            nprobe=10,  # Number of clusters to search
            cache_size=1000  # Number of embeddings to cache
        )
        
        # Search using text query
        results = retriever.search_text("men's casual denim jacket with hood", top_k=5)
        
        # Batch search multiple queries
        queries = ["men's jacket", "women's dress", "kids shoes"]
        batch_results = retriever.search_text_batch(queries, top_k=5)
        
        # Or search using pre-computed embedding
        embedding = retriever.embed_text("men's casual denim jacket with hood")
        results = retriever.search(embedding, top_k=5)
        
        # Example result structure:
        # {
        #     'item_id': 'B07XYZ1234',
        #     'item_name_flat': 'Men\'s Casual Denim Jacket with Hood',
        #     'brand_flat': 'Fashion Brand',
        #     'color_flat': 'Blue',
        #     'material_flat': 'Denim',
        #     'product_type_flat': 'Clothing > Men > Jackets',
        #     'item_keywords_flat': 'jacket, hood, denim, casual',
        #     'similarity_score': 0.92
        # }
        ```
    """
    
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        model_name: Optional[str] = 'all-MiniLM-L12-v2',
        top_k: int = 5,
        nprobe: int = 10,
        cache_size: int = 1000
    ):
        """
        Initialize the FAISS retriever.
        
        Args:
            index_path (str): Path to the FAISS index file
            metadata_path (str): Path to the metadata parquet file containing Amazon product data
            model_name (Optional[str]): Name of the Sentence Transformer model for text embedding.
                                      If None, text embedding functionality will be disabled.
            top_k (int): Default number of top results to return
            nprobe (int): Number of clusters to search in the FAISS index
            cache_size (int): Number of embeddings to cache
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.top_k = top_k
        self.nprobe = nprobe
        self.index = None
        self.metadata = None
        self.model = None
        
        # Load the index and metadata
        self._load_index()
        self._load_metadata()
        
        # Initialize text embedding model if specified
        if model_name:
            self.model = SentenceTransformer(model_name)
            
        # Set up caching
        self._setup_caching(cache_size)
        
        logger.info(f"Initialized FAISSRetriever with {self.get_total_items()} items")
    
    def _setup_caching(self, cache_size: int) -> None:
        """Set up caching for embeddings."""
        self.embed_text = lru_cache(maxsize=cache_size)(self.embed_text)
    
    def _load_index(self) -> None:
        """Load the FAISS index from disk."""
        try:
            if not self.index_path.exists():
                raise IndexError(f"FAISS index not found at {self.index_path}")
            
            self.index = faiss.read_index(str(self.index_path))
            
            # Set search parameters
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
                
            logger.info(f"Loaded FAISS index with dimension {self.index.d}")
        except Exception as e:
            raise IndexError(f"Failed to load FAISS index: {str(e)}")
    
    def _load_metadata(self) -> None:
        """Load the Amazon product metadata from parquet file."""
        try:
            if not self.metadata_path.exists():
                raise MetadataError(f"Metadata file not found at {self.metadata_path}")
            
            self.metadata = pd.read_parquet(self.metadata_path)
            
            # Verify required columns exist
            required_columns = ['item_id', 'item_name_flat', 'brand_flat', 'color_flat', 
                              'material_flat', 'product_type_flat', 'item_keywords_flat']
            missing_columns = [col for col in required_columns if col not in self.metadata.columns]
            if missing_columns:
                raise MetadataError(f"Metadata file missing required columns: {missing_columns}")
                
            logger.info(f"Loaded metadata with {len(self.metadata)} items")
        except Exception as e:
            raise MetadataError(f"Failed to load metadata: {str(e)}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector using the Sentence Transformer model.
        
        Args:
            text (str): Input text to embed
            
        Returns:
            np.ndarray: Embedding vector of shape (384,)
            
        Raises:
            EmbeddingError: If model is not initialized or embedding fails
        """
        try:
            if self.model is None:
                raise EmbeddingError("Text embedding model not initialized. Set model_name in constructor.")
            
            return self.model.encode([text])[0].astype('float32')
        except Exception as e:
            raise EmbeddingError(f"Failed to embed text: {str(e)}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar products using the query embedding.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector of shape (384,)
            top_k (Optional[int]): Number of results to return (overrides default)
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing product information and similarity scores
            
        Raises:
            IndexError: If index is not loaded or search fails
        """
        try:
            if self.index is None or self.metadata is None:
                raise IndexError("Index or metadata not loaded")
            
            # Ensure query embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Get number of results to return
            k = top_k if top_k is not None else self.top_k
            
            # Search the index - fetch more results to account for deduplication
            start_time = time.time()
            distances, indices = self.index.search(query_embedding, k * 2)
            search_time = time.time() - start_time
            
            # Get results from metadata with deduplication
            seen_item_ids = OrderedDict()
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # Skip invalid indices
                    continue
                    
                item_id = self.metadata.iloc[idx]['item_id']
                if item_id not in seen_item_ids:
                    item = self.metadata.iloc[idx].to_dict()
                    item['similarity_score'] = float(1.0 / (1.0 + dist))
                    seen_item_ids[item_id] = item
                    
                if len(seen_item_ids) >= k:
                    break
            
            logger.debug(f"Search completed in {search_time:.3f}s, found {len(seen_item_ids)} results")
            return list(seen_item_ids.values())
        except Exception as e:
            raise IndexError(f"Search failed: {str(e)}")
    
    def search_text(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar products using a text query.
        
        Args:
            query (str): Text query describing the product to search for
            top_k (Optional[int]): Number of results to return (overrides default)
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing product information and similarity scores
            
        Raises:
            EmbeddingError: If text embedding fails
            IndexError: If search fails
        """
        try:
            query_embedding = self.embed_text(query)
            return self.search(query_embedding, top_k)
        except Exception as e:
            raise EmbeddingError(f"Text search failed: {str(e)}")
    
    def search_text_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar products using multiple text queries in batch.
        
        Args:
            queries (List[str]): List of text queries
            top_k (Optional[int]): Number of results to return per query (overrides default)
            
        Returns:
            List[List[Dict[str, Any]]]: List of result lists, one for each query
            
        Raises:
            EmbeddingError: If text embedding fails
            IndexError: If search fails
        """
        try:
            # Embed all queries at once
            embeddings = self.model.encode(queries).astype('float32')
            
            # Search for each embedding
            results = []
            for embedding in embeddings:
                results.append(self.search(embedding, top_k))
                
            return results
        except Exception as e:
            raise EmbeddingError(f"Batch search failed: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings used in the index."""
        if self.index is None:
            raise IndexError("Index not loaded")
        return self.index.d
    
    def get_total_items(self) -> int:
        """Get the total number of products in the index."""
        if self.index is None:
            raise IndexError("Index not loaded")
        return self.index.ntotal
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embed_text.cache_clear()
        logger.info("Cleared embedding cache")
