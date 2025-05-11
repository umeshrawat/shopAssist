# shopAssist
Retrieval-Augmented Shopping Assistant (RAG-based)

## Project Overview
An intelligent shopping assistant that uses Retrieval-Augmented Generation (RAG) to answer user queries about products.

- Dataset: Amazon Berkeley Objects (ABO)
- Embedding: Sentence Transformers / OpenAI embeddings
- Vectorstore: FAISS / Chroma
- LLM: GPT-4 / Open Source alternatives
- Hosting: AWS EC2 + Docker

## Project Scope (4 Week Duration)
1. English language support only.
2. 

## Future Capabilities
1. International Language support at user input (UI).
2. 

## Project Code Repo Structure Description

### src/: 
Source code modules (retriever, generator, embedding, inference)

### scripts/:
Utility scripts for data processing and testing:
- `generate_embeddings.py`: Generates embeddings for text data using various sentence transformer models
- `test_retriever.py`: Tests the FAISS retriever with different nprobe values
- `analyze_word_counts.py`: Analyzes word counts in the metadata

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

### Prerequisites
- Python 3.12.4 (required)
- Git
- pip (Python package installer)

### Installation Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/umeshrawat/shopAssist.git
   cd shopAssist
   ```

2. Create and activate a virtual environment:

   **On Windows:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

   **On macOS/Linux:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required data files (if not already present):
   ```bash
   # The data files will be downloaded automatically when you first run the application
   # They are tracked using Git LFS
   ```

### Development Workflow

1. Activate the virtual environment before starting work:
   ```bash
   # Windows
   .\venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

2. Install new dependencies:
   ```bash
   pip install <package-name>
   pip freeze > requirements.txt  # Update requirements.txt
   ```

3. Deactivate the virtual environment when done:
   ```bash
   deactivate
   ```

### Script Usage

1. Generate embeddings:
   ```bash
   python scripts/generate_embeddings.py
   ```
   This script processes the metadata in chunks to generate embeddings using multiple models:
   - Nomic Embed (nomic-ai/nomic-embed-text-v1.5)
   - BGE-M3 (BAAI/bge-m3)
   - GTE-large (Alibaba-NLP/gte-large-en-v1.5)
   - MiniLM (all-MiniLM-L6-v2)

2. Test retriever:
   ```bash
   python scripts/test_retriever.py
   ```
   This script tests the FAISS retriever with different nprobe values (5, 10, 20, 50).
   Based on testing, nprobe=10 is recommended as it provides the best balance of speed and accuracy.

### Troubleshooting

1. If you encounter permission issues:
   - Windows: Run your terminal as Administrator
   - macOS/Linux: Use `sudo` for commands that require elevated privileges

2. If pip install fails:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. If you get "command not found" errors:
   - Ensure Python is in your system PATH
   - Verify the virtual environment is activated

4. Python Version Issues:
   - This project requires Python 3.12.4 exactly
   - Check your Python version: `python --version`
   - If your version is different, please install Python 3.12.4
   - Higher versions are not tested and may cause compatibility issues

## AWS Deployment Instructions
Refer to [AWS Deployment Guide](docs/aws_deployment_guide.md)

## Hosted Notebooks
- [EDA Notebook on Kaggle](link_to_be_added)
- [Embedding Experiments on Kaggle](link_to_be_added)
