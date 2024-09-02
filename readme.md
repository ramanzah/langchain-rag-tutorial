# RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain and OpenAI's models. The chatbot can answer questions based on the content of "Moby Dick" by Herman Melville.

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ramanzah/langchain-rag-tutorial.git
   cd langchain-rag-tutorial
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip3 install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Ensure you have the "Moby Dick" text file in the `documents` folder. If not, you can download it from Project Gutenberg or another source.

2. Run the chatbot:
   ```
   python rag_chatbot.py
   ```

3. The script will create a vector store (if it doesn't exist) or load an existing one. This may take a few minutes the first time.

4. Once loaded, you can start chatting with the AI. Type your questions about "Moby Dick", and the AI will respond based on the content of the book.

5. To exit the chat, type 'exit'.

## Customization

- To use a different document, replace the `moby_dick.txt` file in the `documents` folder and update the `DOCUMENT_PATH` in `rag_chatbot.py`.
- You can adjust the chunk size and overlap in the `load_and_split_document` function to optimize for your specific use case.
- To use a different OpenAI model, modify the `model` parameter in the `initialize_llm` function.

## Troubleshooting

- If you encounter any issues with the OpenAI API, ensure your API key is correct and you have sufficient credits.
- Make sure all required packages are installed correctly.
- Check that the `documents` and `db` folders exist in the project directory.
