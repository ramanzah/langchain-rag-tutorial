from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def initialize_llm():
    """Initialize and return the language model."""
    return ChatOpenAI(model="gpt-4o-mini")

def chat_loop(llm):
    """Run the main chat loop."""
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Send the query to the language model
        result = llm.invoke(query)  # Pass the query as a string
        print(f"AI: {result.content}")  # Access the content of the AIMessage

def main():
    # Initialize the language model
    llm = initialize_llm()
    # Start the chat loop
    chat_loop(llm)

if __name__ == "__main__":
    main()