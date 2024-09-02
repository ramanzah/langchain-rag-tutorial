import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

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
    
    # Show an example embedding
    example_chunk = chunks[0].page_content
    example_embedding = embeddings.embed_query(example_chunk)
    print("\nExample chunk:")
    print(example_chunk)
    print("\nExample embedding:")
    print(example_embedding)

if __name__ == "__main__":
    main()