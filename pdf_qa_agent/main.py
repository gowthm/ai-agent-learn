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
import hashlib
from ingest import load_and_chunk
from retriever import create_vectorstore, load_vectorstore
from qa import get_answer

PDF_PATH = "data/sample.pdf"
HASH_FILE = "faiss_index/.pdf_hash"  # stores the hash of the last indexed PDF


def get_pdf_hash(path):
    """Return the MD5 hash of the PDF file contents."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def index_is_stale():
    """
    Return True if the FAISS index doesn't exist yet,
    or if the PDF has changed since the index was built.
    """
    if not os.path.exists("faiss_index"):
        return True  # index was never built

    if not os.path.exists(HASH_FILE):
        return True  # hash file missing — can't verify, rebuild to be safe

    current_hash = get_pdf_hash(PDF_PATH)
    with open(HASH_FILE, "r") as f:
        saved_hash = f.read().strip()

    return current_hash != saved_hash  # True if PDF changed


def save_pdf_hash():
    """Save the current PDF's hash so we can detect future changes."""
    with open(HASH_FILE, "w") as f:
        f.write(get_pdf_hash(PDF_PATH))


# ── Step 1 & 2: Build or reload the FAISS index ─────────────────────────────
if index_is_stale():
    print("📄 PDF changed or index missing — re-indexing...")
    chunks = load_and_chunk(PDF_PATH)        # ingest.py  → list of Document chunks
    vectorstore = create_vectorstore(chunks) # retriever.py → embed + save FAISS index
    save_pdf_hash()                          # record hash of newly indexed PDF
else:
    print("✅ PDF unchanged — loading existing index.")
    vectorstore = load_vectorstore()         # retriever.py → reload saved FAISS index

# ── Step 3: Interactive Q&A loop ────────────────────────────────────────────
print("\n💬 PDF Q&A Bot ready! Type 'quit' to exit.\n")

while True:
    question = input("You: ")

    if question.lower() == "quit":
        break

    # qa.py: retrieve top-3 chunks → call LLM → print answer + source pages
    get_answer(question, vectorstore)
