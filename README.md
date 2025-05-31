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
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies for local (Mac/CPU) development:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required data files (if not already present):
   ```bash
   # The data files will be downloaded automatically when you first run the application
   # They are tracked using Git LFS
   ```

### Download Required Data Files

1. **Download the Embeddings File from Kaggle:**
   - Visit the Kaggle dataset page: [Link to your Kaggle dataset]
   - Download the file `inScopeMetadata_with_embeddings.parquet` and place it in the `data/` directory of your project.

2. **Verify the File:**
   - Ensure the file is located at `data/inScopeMetadata_with_embeddings.parquet`.
   - This file is required for the semantic search functionality.

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
   python retriever/generate_embeddings.py
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

This section provides instructions for deploying the shopAssist application on an AWS EC2 instance, particularly for environments requiring GPU support.

### 1. Launch an EC2 Instance

- Use the AWS Console to launch a new EC2 instance (Ubuntu 22.04 or similar recommended).
- Choose an instance type (e.g., g4dn.12xlarge for GPU support).
- Set up the security group to allow SSH (port 22) and any other ports your app needs.
- Download the SSH key (`.pem` file) for access.

### 2. Connect to Your EC2 Instance

```bash
ssh -i /path/to/your-key.pem ubuntu@<your-ec2-public-ip>
```

### 3. Clone the Repository

```bash
git clone https://github.com/umeshrawat/shopAssist.git
cd shopAssist
```

### 4. Run the Environment Setup Script

Navigate to the cloned repository directory:

```bash
cd shopAssist
```

Make the setup script executable:

```bash
chmod +x setup_venv.sh
```

Run the script to install system dependencies (like git, python3, python3-venv), create the Python virtual environment, and install all necessary Python packages from `requirements.txt`:

```bash
./setup_venv.sh
```

After the script completes, activate the virtual environment in your current shell:

```bash
source venv/bin/activate
```

---

**Note:**
As we continue development, we'll update these instructions with steps for running the app, setting environment variables, and more.

#### Example EC2 Hardware Specification

These are the specs for the EC2 instance used in this deployment:

- **Instance type:** g4dn.12xlarge
- **CPU:** Intel(R) Xeon(R) Platinum 8259CL @ 2.50GHz (48 vCPUs)
- **RAM:** ~187 GB

> Update this section if you use a different instance type.

---

### Running the Embedding Generation Script

> **Note:**
> All commands below assume you are in the project root directory:
> `/home/ubuntu/shopAssist`

#### 1. Ensure Git LFS is installed and pull large files

If you have not already, install Git LFS and pull the required data files:

```bash
sudo apt update
sudo apt install git-lfs
cd /home/ubuntu/shopAssist
git lfs install
git lfs pull
```

#### 2. Set up the environment

Navigate to the project root and run the setup script:

```bash
cd /home/ubuntu/shopAssist
chmod +x setup_venv.sh
./setup_venv.sh
```

After running the setup script, activate the virtual environment in your current shell:

```bash
source venv/bin/activate
```

#### 3. Run the embedding generation script

```bash
python3 retriever/generate_embeddings.py
```

---

**Troubleshooting:**
- If you see errors about missing or corrupted Parquet files, ensure you have run `git lfs pull` in the project root.
- Always run the script from the project root so relative paths resolve correctly.

## Hosted Notebooks
- [EDA Notebook on Kaggle](https://www.kaggle.com/code/umeshrawat7/shopassist-eda)
- [Embedding Experiments on Kaggle](https://www.kaggle.com/code/umeshrawat7/retrieval-augmented-shopping-assistant-embeddi)
