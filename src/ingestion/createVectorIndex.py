import ingestion.dataAnalysis as dataAnalysis
import pandas as pd
from ingestion.inferenceModel import getSentenceTransformerMiniModel, getSentenceTransformerGeminiModel
import faiss
import os

INDEX_PATH_FAISS = "faiss_index"
model = getSentenceTransformerMiniModel()
def create_index():
    # Implementation for creating a vector index
    df = dataAnalysis.DataAnalysis().analyze()
    print(f"Data shape: {df.shape}, data type: {df.dtypes}, df: {df.head()}")
    # model = getSentenceTransformerMiniModel()
    embeddings = model.encode(df.astype(str).tolist(), convert_to_tensor=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.cpu().numpy())
    return index
   
def search(query, top_k=2):
    query_vector = model.encode([query], convert_to_tensor=True)
    if os.path.exists(INDEX_PATH_FAISS):
        index = faiss.read_index(INDEX_PATH_FAISS)
        print("Loaded existing FAISS index ✅")
        distances,indices = index.search(query_vector, top_k)
        return distances, indices
    else:
        print("Creating new FAISS index...")
        index = create_index()
        print(f"FAISS index created with {index.ntotal} documents ✅")
        # index.add(embeddings)
        faiss.write_index(index,INDEX_PATH_FAISS)
        distances,indices = index.search(query_vector, top_k)
        print(f"Search results: {distances}, {indices}")
        return distances, indices
    
def main_search(query, top_k=2):
    distances, indices = search(query, top_k)
    df = pd.read_csv('data/shopAssist_ori.csv')
    result_df = df.iloc[indices[0]]

    images_df = pd.read_csv('data/images.csv')
    image_id = result_df['main_image_id'].tolist()
    image_mapping = images_df[images_df['image_id'].isin(image_id)]['path'].tolist()
    for i, (idx,row) in enumerate(result_df.iterrows()):
      result_df['main_image_id'] = "static/"+ image_mapping[i]
    return result_df.to_json(orient='records').replace("\'", "'")

    # Process the results

    re