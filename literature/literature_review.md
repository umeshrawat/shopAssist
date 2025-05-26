
## Literature Review: ShopTalk - AI-Powered Shopping Assistant

### 1. Virtual Assistant Technologies

#### Key Frameworks & Papers:
- **T5X: Scaling Up Text-to-Text Tasks** – Lays the foundation for LLM-based unified query understanding and generation.
- **Task-Oriented Dialogue Systems (TODs)** – Shift from intent/slot systems to flexible LLM orchestration.
- **AlexaTM 20B** – Demonstrates how large language models can scale semantic understanding and conversation retention.

#### Key Trends:
- LLMs replacing traditional NLP pipelines for search and question answering
- Integration with agents/tools (LangChain agents, ReAct)
- Conversational context managed via memory buffers or embeddings

---

### 2. Embedding Models for Product Retrieval

#### Selected Models:
- `nomic-ai/nomic-embed-text-v1.5` – Long-context (8192 tokens), instruction-aligned
- `BAAI/bge-m3` – Multi-vector output, hybrid (dense + sparse) retrieval
- `Alibaba-NLP/gte-large-en-v1.5` – High-quality dense embedding, English-optimized
- `all-MiniLM-L6-v2` – Baseline for short metadata, fast inference

#### Foundational Papers:
- **BEIR Benchmark** – Standard for evaluating embedding models across domains
- **E5: Embedding-based Efficient Retriever** – Introduced instruction-tuned dense retrieval
- **ColBERTv2** – Late interaction approach for high-precision retrieval

#### Tools:
- **LangChain** – Flexible integration of retrievers, prompt templates, and conversational chains
- **LlamaIndex** – Lightweight abstraction over document indexing and query resolution

---

### 3. Image Captioning for Multimodal Input

#### Recommended Models:
- `Salesforce/BLIP-2` – Strong captioning + retrieval performance
- `BLIP` (BLIP-1) – Foundation model for captioning
- `OFASys/ofa-base` – Multimodal captioning and classification
- `GIT` – Generative, fluent captioning ideal for ecommerce

#### Key Papers:
- **BLIP** – Bootstrapping Language-Image Pre-training
- **GIT** – Grounded Image Transformer, strong caption fluency
- **Florence (Microsoft)** – VL multitask pretraining at scale

#### Integration Strategy:
- Caption product images → append to `embedding_input`
- Use image embeddings directly or fuse with text

---

### Summary Alignment
| Component | Strategy |
|----------|----------|
| Virtual Assistant | LangChain + Retriever + Generator |
| Embedding Models | Nomic, BGE-M3, GTE, MiniLM (baseline) |
| Vector DBs | FAISS → Weaviate/Qdrant optional hybrid upgrade |
| Image Captioning | BLIP-2 + integration into embedding_input |
| Multimodal RAG | Late fusion or dual-vector retrieval
