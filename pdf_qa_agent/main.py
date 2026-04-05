"""
main.py — Entry point for the PDF Q&A Bot

Overall architecture (RAG — Retrieval-Augmented Generation):
─────────────────────────────────────────────────────────────
  PDF file
    │
    ▼
  ingest.py      → load pages, split into ~500-char chunks
    │
    ▼
  retriever.py   → embed chunks with a local model,
                   store vectors in a FAISS index (saved to disk)
    │
    ▼  (on later runs, skip straight to loading the saved index)
    │
    ▼
  qa.py          → embed the user question, find the 3 most
                   similar chunks, pass them + question to an LLM,
                   print the answer and source pages

This pattern is called RAG:
  - Retrieval  : find relevant text from the document
  - Augmented  : inject that text into the LLM's prompt as context
  - Generation : LLM generates an answer grounded in that context
"""

import os
from ingest import load_and_chunk
from retriever import create_vectorstore, load_vectorstore
from qa import get_answer

PDF_PATH = "data/sample.pdf"

# ── Step 1 & 2: Build the FAISS index (only on the very first run) ──────────
# If the index folder already exists we skip the expensive embedding step
# and just reload the saved index from disk.
if not os.path.exists("faiss_index"):
    print("📄 Loading and chunking PDF...")
    chunks = load_and_chunk(PDF_PATH)       # ingest.py  → list of Document chunks
    vectorstore = create_vectorstore(chunks) # retriever.py → embed + save FAISS index
else:
    vectorstore = load_vectorstore()         # retriever.py → reload saved FAISS index

# ── Step 3: Interactive Q&A loop ────────────────────────────────────────────
print("\n💬 PDF Q&A Bot ready! Type 'quit' to exit.\n")

while True:
    question = input("You: ")

    if question.lower() == "quit":
        break

    # qa.py: retrieve top-3 chunks → call LLM → print answer + source pages
    get_answer(question, vectorstore)
