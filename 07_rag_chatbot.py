import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


# Load environment variables
load_dotenv()

# Define file paths and directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "db")
DOCUMENT_PATH = os.path.join(CURRENT_DIR, "documents", "moby_dick.txt")

def initialize_embeddings():
    """Initialize and return the embedding model."""
    return OpenAIEmbeddings(model="text-embedding-3-small")

def load_and_split_document(file_path):
    """Load the document and split it into chunks."""
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_or_load_vector_store(embeddings, store_name):
    """Create a new vector store or load an existing one."""
    persistent_directory = os.path.join(DB_DIR, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        docs = load_and_split_document(DOCUMENT_PATH)
        vector_store = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(f"Loading existing vector store {store_name}")
        vector_store = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    return vector_store

def create_retriever(vector_store):
    """Create a retriever from the vector store."""
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def initialize_llm():
    """Initialize and return the language model."""
    return ChatOpenAI(model="gpt-4o-mini")

def create_history_aware_retriever_chain(llm, retriever):
    """Create a history-aware retriever chain."""
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

def create_qa_chain(llm):
    """Create a question-answering chain."""
    qa_system_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return create_stuff_documents_chain(llm, qa_prompt)

def create_rag_chain(history_aware_retriever, qa_chain):
    """Create the final RAG (Retrieval-Augmented Generation) chain."""
    return create_retrieval_chain(history_aware_retriever, qa_chain)

def chat_loop(rag_chain):
    """Run the main chat loop."""
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"AI: {result['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))

def main():
    # Initialize components
    embeddings = initialize_embeddings()
    vector_store = create_or_load_vector_store(embeddings, "chroma_db")
    retriever = create_retriever(vector_store)
    llm = initialize_llm()
    
    # Create chains
    history_aware_retriever = create_history_aware_retriever_chain(llm, retriever)
    qa_chain = create_qa_chain(llm)
    rag_chain = create_rag_chain(history_aware_retriever, qa_chain)
    
    # Start chat
    chat_loop(rag_chain)

if __name__ == "__main__":
    main()