# shopAssist
Retrieval-Augmented Shopping Assistant (RAG-based)

## Project Overview
An intelligent shopping assistant that uses Retrieval-Augmented Generation (RAG) to answer user queries about products.

- Dataset: Amazon Berkeley Objects (ABO)
- Embedding: Sentence Transformers / OpenAI embeddings
- Vectorstore: FAISS / Chroma
- LLM: GPT-4 / Open Source alternatives
- Hosting: AWS EC2 + Docker
- UI: Streamlit / Gradio

## Project Code Repo Structure Description

### src/: 
Source code modules (retriever, generator, embedding, inference, ui)

### notebooks/:
EDA, experiments, ad-hoc analysis (non-production code)

### docs/:
Architecure, setup guides, testing reports (for collaborators and future scope)

### data/:
Cleaned samples, vectorstore dumps

### outputs/:
Retrieval results, logs

### tests/:
Unit tests, E2E scripts

## Setup Instructions (Local Development)

1. Clone this repo:
   ```bash
   git clone https://github.com/umeshrawat/shopAssist.git
   cd rag-shopping-assistant
   ```
2.  $env:PYTHONPATH = "$PWD\src"
2. Create environment:
   ```bash
   conda create -n rag-assistant python=3.10
   conda activate rag-assistant
   pip install -r requirements.txt
   ```

3. Run locally:
   ```bash
   python src/main.py
   ```

## AWS Deployment Instructions
Refer to [AWS Deployment Guide](docs/aws_deployment_guide.md)

## Hosted Notebooks
- [EDA Notebook on Kaggle](link_to_be_added)
- [Embedding Experiments on Kaggle](link_to_be_added)

## How to Use
- Access the Streamlit app at `http://<your-aws-public-ip>:8501`
- Input your shopping queries
- View recommendations in real-time!
