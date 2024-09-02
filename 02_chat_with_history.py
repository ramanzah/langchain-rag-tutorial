from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

def initialize_llm():
    """Initialize and return the language model."""
    return ChatOpenAI(model="gpt-4o-mini")

def chat_loop(llm):
    """Run the main chat loop."""
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Create a list of messages for the input
        messages = chat_history + [HumanMessage(content=query)]

        # Show the messages that will be sent to the language model
        # print("Messages to be sent to the language model:")
        # print(f"{messages}")

        # Send the query to the language model
        result = llm.invoke(messages)  # Pass the list of messages
        print(f"AI: {result.content}")  # Access the content of the AIMessage
        # Append the conversation to the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result.content))

def main():
    # Initialize the language model
    llm = initialize_llm()
    # Start the chat loop
    chat_loop(llm)

if __name__ == "__main__":
    main()