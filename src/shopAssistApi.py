import numpy as np
from fastapi import FastAPI, Query
from ingestion.ingest import retrive, getModel
from ingestion.model import SearchRequest,SearchResponse
app = FastAPI()
app.mount("/images", static_dir="images")

# Search endpoint
@app.post("/search")
def search(request: SearchRequest):
    embedding = getModel().encode([request.query]).astype('float32')
    results = retrive(embedding, request)
    return results