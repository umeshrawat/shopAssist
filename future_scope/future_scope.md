
## Future Scope: ShopTalk - AI-Powered Shopping Assistant

### 1. Fine-tuning with Domain Expansion
- Explore fine-tuning the embedding models (e.g., Nomic, GTE, BGE-M3) using a broader e-commerce corpus.
- Focus areas: fashion, electronics, decor — where generalization across synonyms and style context matters.
- Use contrastive loss (triplet loss or NT-Xent) or LoRA/QLoRA-based adapter tuning.

### 2. Common Crawl Integration
- Leverage Common Crawl (https://commoncrawl.org/get-started) as a **supplementary corpus** for large-scale fine-tuning.
- Filter to product-rich domains: `*.amazon.com`, `*.etsy.com`, `*.walmart.com`, `*.homedepot.com`, etc.
- Use structured wrappers to extract product titles, descriptions, prices, reviews, and structured markup (JSON-LD or Schema.org).
- Cleaned crawl data can enhance retrieval model generalization beyond ABO.

### 3. Image + Text Fusion Advancements
- Use models like BLIP-2 or GIT to generate high-quality image captions.
- Fuse these with structured metadata during embedding input construction.
- Experiment with early vs. late fusion for vector database queries.

### 4. Voice-based Query Support
- Incorporate Whisper or Google Speech-to-Text APIs to allow voice input.
- Ensure seamless transition from voice → query → retrieval → generation.

### 5. Personalization Layer
- Use historical product interaction data to fine-tune rerankers.
- Explore embedding-based collaborative filtering or vector reranking using user embeddings.

### 6. Feedback Loops & Reinforcement
- Capture thumbs-up/down signals in UI.
- Use them to fine-tune reward-based re-rankers (e.g., DPO or offline RLHF).

### 7. Multilingual Support
- Expand product discovery for non-English markets using multilingual embedding models (e.g., `bge-m3`, `LaBSE`, `distiluse` multilingual).
- Leverage multilingual captioning or translation via MarianMT or SeamlessM4T.

### 8. Dynamic Retrieval Stack Optimization
- Build a hybrid retriever: dense (FAISS) + sparse (BM25 or SPLADE).
- Deploy auto-switching pipeline depending on query type (specific vs exploratory).


### 9. Finetuning LLMs
- Understand how to perform instruction-based fine-tuning of LLMs using techniques such as LoRA or QLoRA.
- Focus on adapting open-source models to domain-specific language and query patterns found in ecommerce.
- Investigate inference optimization for open-source LLMs to reduce latency in production.

### 10. Image Input Extension
- Enable direct visual input from users (e.g., uploading a photo of a product like shoes or furniture).
- Extract semantic meaning from the photo using vision transformers or multimodal models (e.g., CLIP, BLIP-2).
- Enhance retrieval by combining visual semantics with metadata for hybrid search.

### 11. Personalisation
- Leverage historical user–item interaction data to create personalized product embeddings or retrieval filters.
- Implement personalization as part of initial retrieval or as a re-ranking strategy using user profiles.


### 12. Follow-Up Support
- Integrate conversational memory to handle multi-turn user interactions.
- Retain past queries and responses using LangChain memory modules to improve contextual continuity.
- Support cross-turn reference resolution (e.g., "something like the last one" queries).

### 13. Voice Input Integration
- Enable voice-to-text conversion using Whisper or equivalent APIs.
- Integrate voice commands into the front-end (e.g., Streamlit/Gradio) for hands-free product search.

### 14. Feedback Loop (Explicit Signals)
- Capture thumbs-up/down or click-based feedback to monitor result relevance.
- Store feedback logs for training rerankers or fine-tuning embedding models using Reinforcement Learning from Human Feedback (RLHF).
