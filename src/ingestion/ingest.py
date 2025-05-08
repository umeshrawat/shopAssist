
from ingestion.inferenceVector import build_faiss_index, build_chromadb_index
from ingestion.model import SearchRequest,SearchResponse
# Paths
INDEX_PATH = "faiss_index"
DATA_PATH = "data.json"

def retrive(embeddings,query: SearchRequest):
# Function to build index
    index: any = None
    if query.vectorType == "faiss":
        index,documents = build_faiss_index(embeddings)
    elif query.vectorType == "chromadb":
        index,documents = build_chromadb_index(embeddings)
    
    D,I = index.search(embeddings, query.k)
    print(f"Index '{I[0]}':")
   
    searchResponses = []
    for idx in I[0]:
        print(f"Document {idx}: {documents[idx]}")
        if idx < len(documents):
            doc=documents[idx]
            print(f"Document {idx}: {doc}")
            searchResponses.append(SearchResponse(result=doc.page_content, url=doc.metadata["image_url"]))
    return {"query": query, "results": searchResponses}

def getModel(modelType:str = "mini"):
    # Load the model
    if modelType == "mini":
        from ingestion.inferenceModel import getSentenceTransformerMiniModel
        return getSentenceTransformerMiniModel()
    else:
        from ingestion.inferenceModel import getSentenceTransformerGeminiModel
        return getSentenceTransformerGeminiModel()




   
