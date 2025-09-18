import os
import sys
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Configuration ---
INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "google/flan-t5-large"

def setup_qa_chain():
    """Sets up the RetrievalQA chain."""
    load_dotenv()
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("Error: Hugging Face API token not found.")
        print("Please create a .env file with HUGGINGFACEHUB_API_TOKEN='your_token'")
        sys.exit(1)

    if not os.path.exists(INDEX_PATH):
        print(f"Error: FAISS index not found at '{INDEX_PATH}'")
        print("Please run 'python setup_vectorstore.py' first.")
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Loading vector store...")
    vectorstore = FAISS.load_local(
        INDEX_PATH, 
        embeddings,
        allow_dangerous_deserialization=True 
    )
    print("Vector store loaded successfully.")

    retriever = vectorstore.as_ retriever()

    llm = HuggingFaceHub(
        repo_id=LLM_REPO_ID,
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

def main():
    """Main function to run the RAG application."""
    print("Initializing RAG chain...")
    qa_chain = setup_qa_chain()
    print("\nWelcome to the Repository Q&A! Type 'exit' to quit.")
    
    while True:
        question = input("\nPlease ask a question about your repository: ")
        if question.lower() == 'exit':
            break
        
        if not question.strip():
            continue

        print("\nThinking...")
        
        response = qa_chain.invoke({"query": question})
        
        print("\n--- Answer ---")
        print(response['result'].strip())
        print("----------------\n")
        
        show_sources = input("Show sources? (y/n): ").lower()
        if show_sources == 'y':
            print("\n--- Sources ---")
            for i, doc in enumerate(response['source_documents']):
                print(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}")
            print("-----------------\n")

if __name__ == "__main__":
    main()
