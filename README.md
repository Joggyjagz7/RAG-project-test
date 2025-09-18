# Q&A with RAG

This project uses a Retrieval-Augmented Generation (RAG) system to answer questions about a local code repository.

## How It Works



1.  **Indexing (`setup_vectorstore.py`)**: The script loads files from your repository, splits them into chunks, creates numerical vector embeddings for each chunk, and saves them in a local FAISS vector store.
2.  **Querying (`app.py`)**: When you ask a question, the app converts it into an embedding, finds the most similar document chunks from the vector store, and sends them along with your question to a Large Language Model (LLM) to generate a context-aware answer.

## Setup and Usage

### 1. Install Dependencies

It's best practice to use a Python virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt


