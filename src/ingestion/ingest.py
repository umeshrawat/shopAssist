
from ingestion.inferenceVector import build_faiss_index, build_chromadb_index
# Paths
INDEX_PATH = "faiss_index"
DATA_PATH = "data.json"



def retrive(embeddings,query, vectorType:str,k=2):
# Function to build index
    index: any = None
    if vectorType == "faiss":
        index,documents = build_faiss_index(embeddings)
    elif vectorType == "chromadb":
        index,documents = build_chromadb_index(embeddings)
    
    D,I = index.search(embeddings, k)
    print(f"Index '{I[0]}':")
    results = []
    for idx in I[0]:
        print(f"Document {idx}: {documents[idx]}")
        if idx < len(documents):
            results.append(documents[idx])    
    return {"query": query, "results": results}

def getModel(modelType:str = "mini"):
    # Load the model
    if modelType == "mini":
        from ingestion.inferenceModel import getSentenceTransformerMiniModel
        return getSentenceTransformerMiniModel()
    else:
        from ingestion.inferenceModel import getSentenceTransformerGeminiModel
        return getSentenceTransformerGeminiModel()




   
