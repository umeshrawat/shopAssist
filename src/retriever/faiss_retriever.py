# faiss_retriever.py
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import OrderedDict

class FaissRetriever:
    def __init__(self, index_path, metadata_path, model_name='all-MiniLM-L6-v2'):
        self.index = faiss.read_index(index_path)
        self.metadata_df = pd.read_parquet(metadata_path)
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query):
        return self.model.encode([query])[0].astype('float32')

    def search(self, query, top_k=50, final_k=5):
        query_vec = self.embed_query(query)
        D, I = self.index.search(np.array([query_vec]), top_k)
        distances = D[0]
        indices = I[0]

        seen_item_ids = OrderedDict()
        for dist, idx in zip(distances, indices):
            item_id = self.metadata_df.iloc[idx]['item_id']
            if item_id not in seen_item_ids:
                seen_item_ids[item_id] = {
                    'item_id': item_id,
                    'index': idx,
                    'distance': dist,
                    'metadata': self.metadata_df.iloc[idx].to_dict()
                }
            if len(seen_item_ids) >= final_k:
                break

        return list(seen_item_ids.values())
