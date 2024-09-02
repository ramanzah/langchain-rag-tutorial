import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Define file paths and directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
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

def main():
    # Load the document
    documents = load_document(DOCUMENT_PATH)
    print(f"Loaded {len(documents)} document(s).")
    
    # Split the document into chunks
    chunks = split_document(documents)
    print(f"Split the document into {len(chunks)} chunks.")
    
    # Print the first chunk as an example
    if chunks:
        print("\nFirst chunk:")
        print(chunks[0].page_content)
        print("-" * 50)

        print("\nSecond chunk:")
        print(chunks[1].page_content)

if __name__ == "__main__":
    main()