import numpy as np
from fastapi import FastAPI, Query
from ingestion.ingest import retrive, getModel
from ingestion.model import SearchRequest,SearchResponse
from fastapi.staticfiles import StaticFiles
import os
app = FastAPI()
# Calculate the absolute path to the static directory
static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))

# Mount it
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Search endpoint
@app.post("/search")
def search(request: SearchRequest):
    embedding = getModel().encode([request.query]).astype('float32')
    results = retrive(embedding, request)
    return results