import os
import sys
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
# 1. Path to the repository you want to process.
#    Update this to the relative or absolute path of your target repository.
REPO_PATH = "../your-repo-here" 

# 2. Glob pattern to select files (e.g., "**/*.py" for Python, "**/*.md" for Markdown).
FILE_GLOB = "**/*"

# 3. Path to save the FAISS vector store index.
INDEX_PATH = "faiss_index"

# 4. Name of the sentence transformer model for embeddings.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 5. Chunking parameters.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def create_vector_store():
    """Loads, chunks, embeds, and stores repository documents in a FAISS vector store."""
    if not os.path.exists(REPO_PATH):
        print(f"Error: Repository path not found at '{REPO_PATH}'")
        sys.exit(1)
        
    print(f"Loading documents from: {REPO_PATH}")

    loader = DirectoryLoader(
        REPO_PATH, 
        glob=FILE_GLOB, 
        loader_cls=lambda path: TextLoader(path, encoding='utf-8'),
        show_progress=True,
        use_multithreading=True,
        silent_errors=True # Skips files it can't read
    )
    
    documents = loader.load()

    if not documents:
        print("No documents were loaded. Check your REPO_PATH and FILE_GLOB settings.")
        return

    print(f"Successfully loaded {len(documents)} documents.")

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    # Create embeddings
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Create a FAISS vector store from the chunks
    print("Creating vector store... This may take a few minutes.")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Save the vector store locally
    vectorstore.save_local(INDEX_PATH)
    print(f"Vector store created and saved to '{INDEX_PATH}'")

if __name__ == "__main__":
    create_vector_store()
