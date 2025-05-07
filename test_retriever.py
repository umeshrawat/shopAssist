from retriever.faiss_retriever import FAISSRetriever
import logging
import time

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

"""
Observations from nprobe testing (5, 10, 20, 50):

1. Performance:
   - Initialization time: Consistent ~4-5 seconds across all nprobe values
   - Search time:
     * nprobe=5: 0.14-0.43s (slowest)
     * nprobe=10: 0.01-0.02s (fastest)
     * nprobe=20: 0.02-0.03s
     * nprobe=50: 0.02s
   - Higher nprobe values (≥10) actually perform better than nprobe=5

2. Result Quality:
   - Results and similarity scores are identical across all nprobe values:
     * "men's casual denim jacket": 0.456
     * "women's summer floral dress": 0.454
     * "kids' running shoes": 0.469
   - This suggests the index is well-structured and nprobe=10 is sufficient

3. Recommendation:
   - Use nprobe=10 as default because:
     * Fastest search times
     * Good result quality
     * Stable performance
     * No significant improvement with higher values
     * Good balance between speed and accuracy

4. Notes:
   - The stability of results across nprobe values suggests the index is well-clustered
   - Search times are very fast (0.01-0.02s) for nprobe ≥ 10
   - No need to use higher nprobe values as they don't improve results
"""

def test_retriever(nprobe: int, queries: list):
    print(f"\n{'='*80}")
    print(f"Testing with nprobe = {nprobe}")
    print(f"{'='*80}")
    
    # Initialize the retriever
    start_time = time.time()
    retriever = FAISSRetriever(
        index_path="data/faiss_index.index",
        metadata_path="data/inScopeMetadata_with_embeddings.parquet",
        model_name="all-MiniLM-L12-v2",
        top_k=5,
        nprobe=nprobe,
        cache_size=1000
    )
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.2f}s")
    
    # Test each query
    for query in queries:
        print(f"\nQuery: '{query}'")
        start_time = time.time()
        results = retriever.search_text(query)
        search_time = time.time() - start_time
        
        print(f"Search time: {search_time:.2f}s")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['item_name_flat']}")
            print(f"   Brand: {result['brand_flat']}")
            print(f"   Color: {result['color_flat']}")
            print(f"   Material: {result['material_flat']}")
            print(f"   Product Type: {result['product_type_flat']}")
            print(f"   Keywords: {result['item_keywords_flat']}")
            print(f"   Similarity Score: {result['similarity_score']:.3f}")

def main():
    # Test queries
    queries = [
        "men's casual denim jacket with hood",
        "women's summer floral dress",
        "kids' running shoes"
    ]
    
    # Test different nprobe values
    # Note: nprobe=10 seems optimal based on testing
    nprobe_values = [5, 10, 20, 50]
    
    for nprobe in nprobe_values:
        test_retriever(nprobe, queries)

if __name__ == "__main__":
    main() 