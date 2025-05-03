import numpy as np
from fastapi import FastAPI, Query
from ingestion.ingest import retrive, getModel

app = FastAPI()

# Search endpoint
@app.get("/search")
def search(query: str = Query(..., description="Query string to search"), vectorType: str = "faiss", k: int = 2):
    embedding = getModel().encode([query]).astype('float32')
    results = retrive(embedding, query, vectorType, k)
    return {"query": query, "results": results}