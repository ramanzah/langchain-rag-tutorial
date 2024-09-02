import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Define file paths and directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "db")
DOCUMENT_PATH = os.path.join(CURRENT_DIR, "documents", "moby_dick.txt")

def load_document(file_path):
    """Load the document from the given file path."""
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

def split_document(documents):
    """Split the document into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings():
    """Initialize and return the embedding model."""
    return OpenAIEmbeddings(model="text-embedding-ada-002")

def create_vector_store(embeddings, chunks, store_name):
    """Create a vector store from the embeddings and chunks."""
    persistent_directory = os.path.join(DB_DIR, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(f"Loading existing vector store {store_name}")
        vector_store = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    return vector_store

def perform_retrieval(vector_store, query, k=3):
    """Perform a similarity search to retrieve relevant chunks."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    results = retriever.invoke(query)
    return results

def main():
    # Load the document
    documents = load_document(DOCUMENT_PATH)
    print(f"Loaded {len(documents)} document(s).")
    
    # Split the document into chunks
    chunks = split_document(documents)
    print(f"Split the document into {len(chunks)} chunks.")
    
    # Create embeddings
    embeddings = create_embeddings()
    print("Created embeddings model.")
    
    # Create a vector store
    vector_store = create_vector_store(embeddings, chunks, "chroma_db")
    print("Created vector store.")
    
    # Perform a retrieval
    query = "What is the first line of Moby Dick?"
    results = perform_retrieval(vector_store, query)
    print(f"\nQuery: {query}")
    print("Retrieved chunks:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(result.page_content)

if __name__ == "__main__":
    main()