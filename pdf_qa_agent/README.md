# PDF Q&A Agent

A local, Retrieval-Augmented Generation (RAG) based Q&A agent that allows you to chat with a PDF document.

## Architecture

This project implements a standard RAG pipeline to provide accurate, grounded answers based on the content of a PDF:

1. **Ingestion (`ingest.py`)**: Loads the PDF (`data/sample.pdf`) using PyPDFLoader and chunks the text into overlapping segments to preserve contextual flow.
2. **Retrieval (`retriever.py`)**: Converts the text chunks into dense vectors using local HuggingFace sentence embeddings (`all-MiniLM-L6-v2`) and builds a FAISS vector index. The index is saved locally in `faiss_index/` so that subsequent runs are instantaneous.
3. **Question & Answering (`qa.py`)**: Takes the user's natural language query, retrieves the top 3 most relevant context chunks from the FAISS index, and feeds them into a large language model (`ChatGroq` using `llama-3.1-8b-instant`) to generate a grounded answer alongside the source pages.

## Technologies Used

- **LangChain**: Application framework for building the RAG pipeline.
- **FAISS**: Efficient similarity search and vector storage.
- **HuggingFace Embeddings**: Local text vectorization via Sentence-transformers.
- **Groq API (Llama 3.1 8B)**: Extremely fast local inference to generate human-like answers based on the prompt.
- **pypdf**: Essential utility for reading PDF pages.
- **uv**: Lightning-fast package manager.

## Prerequisites

- **Python**: `>=3.11`
- **uv**: Python package manager installed.
- **Groq API Key**: You need a free API key from [Groq](https://console.groq.com/keys) for the LLM.

## Setup Instructions

1. **Environment Setup**:
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

2. **Add Your PDF**:
   Place your desired PDF in the `data/` directory and ensure the correct path is configured in `main.py` (`PDF_PATH = "data/sample.pdf"`).

3. **Install Dependencies**:
   This project uses `uv` for dependency management. Install the dependencies by running:
   ```bash
   uv sync
   ```

4. **Run the Application**:
   Start the interactive chatbot. The initial run will take a few seconds to chunk the PDF and generate vector embeddings. The results will be stored locally to speed up future runs.
   ```bash
   uv run main.py
   ```

## Usage

Once running, the CLI will wait for your input. Type your questions directly into the terminal. 
Type `quit` when you're done explicitly to exit the chat session. Wait time is minimal due to Groq's high-speed inference.

```
💬 PDF Q&A Bot ready! Type 'quit' to exit.

You: What is the main topic of this document?
🤖 Answer: [Grounded Answer based on your PDF]
📄 Source pages: [1, 2]
```
