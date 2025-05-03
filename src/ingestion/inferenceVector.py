"""
Create a FAISS vector database from the given documents.

Args:
    documents (list): List of documents to be indexed.
    name (str): Name of the vector database.

Returns:
    faiss_index: The FAISS index object.
"""
from langchain_community.vectorstores import Chroma
from ingestion.inferenceModel import getSentenceTransformerMiniModel, getSentenceTransformerGeminiModel
import os
import json
import faiss

INDEX_PATH_FAISS = "faiss_index"
INDEX_PATH_CHROMADB = "chromadb_index"
DATA_PATH = "data.json"
texts=[]
documents=[]
# def load_documents(file_path):
#     """Load documents from a JSON file."""
#     with open(file_path, "r") as f:
#         documents = json.load(f)
#     texts = [doc["text"] for doc in documents]
#     return texts
with open(DATA_PATH, "r") as f:
    documents = json.load(f)
texts = [doc["text"] for doc in documents]


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    # Load model
    model = getSentenceTransformerMiniModel()
    index = faiss.IndexFlatL2(dimension)
     # Check if FAISS index exists
    if os.path.exists(INDEX_PATH_FAISS):
        index = faiss.read_index(INDEX_PATH_FAISS)
        print("Loaded existing FAISS index ✅")
    else:
        print("Creating new FAISS index...")
        # texts = load_documents(DATA_PATH)
        print(f"Loaded {len(texts)} documents from {DATA_PATH} ✅")
        embeddings = model.encode(texts).astype('float32')
        print(f"FAISS index created with {index.ntotal} documents ✅")
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH_FAISS)
    return index,documents

def build_chromadb_index(embeddings):
    dimension = embeddings.shape[1]
    # Load model
    model = getSentenceTransformerGeminiModel()
    index = Chroma.IndexFlatL2(dimension)
     # Check if Chromadb index exists
    if os.path.exists(INDEX_PATH_CHROMADB):
        index = Chroma.read_index(INDEX_PATH_CHROMADB)
        print("Loaded existing Chromadb index ✅")
    else:
        print("Creating new Chromadb index...")
        # texts = load_documents(DATA_PATH)
        embeddings = model.encode(texts).astype('float32')
        print(f"Chromadb index created with {index.ntotal} documents ✅")
        index.add(embeddings)
        Chroma.write_index(index, INDEX_PATH_CHROMADB)
    return index

