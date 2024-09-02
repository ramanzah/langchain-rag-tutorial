from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

def initialize_llm():
    """Initialize and return the language model."""
    return ChatOpenAI(model="gpt-4o-mini")

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

def main():
    # Initialize the language model
    llm = initialize_llm()
    
    # Create the QA chain
    qa_chain = create_qa_chain(llm)
    
    # Example usage of the QA chain
    example_context = Document(page_content="Moby Dick is a novel by Herman Melville.")
    example_question = "Who wrote Moby Dick?"
    # example_question = "what is my name?"

    chat_history = []

    # chat_history = [
    #     {"role": "human", "content": "Hello, my name is Raman."},
    #     {"role": "assistant", "content": "Hello Raman, how can I help you today?"}
    # ]

    result = qa_chain.invoke({
        "input": example_question,
        "context": [example_context],
        "chat_history": chat_history
    })
    print(f"Question: {example_question}")
    print(f"Answer: {result}")

if __name__ == "__main__":
    main()