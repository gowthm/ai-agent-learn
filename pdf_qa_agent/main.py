"""
main.py — Entry point for the PDF Q&A Bot

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY RAG? (Retrieval-Augmented Generation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LLMs have a token limit — you cannot dump an entire PDF into them.
RAG solves this by finding only the relevant pieces first:

  Whole PDF approach (BAD):
    PDF (10,000 words) → LLM   ← expensive, slow, hits token limits

  RAG approach (USED HERE):
    PDF (10,000 words) → find 3 relevant chunks (150 words) → LLM
                         ↑
                    FAISS does this instantly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLETE FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────┐
  │  sample.pdf │
  └──────┬──────┘
         │ PyPDFLoader  (ingest.py)
         ▼
  ┌─────────────┐
  │    Pages    │
  └──────┬──────┘
         │ RecursiveCharacterTextSplitter  (ingest.py)
         ▼
  ┌─────────────┐
  │   Chunks    │  500-char pieces, 50-char overlap
  └──────┬──────┘
         │ HuggingFace all-MiniLM-L6-v2  (retriever.py)
         │ runs locally — no API key, no cost
         ▼
  ┌─────────────┐
  │   Vectors   │──── saved to ────► faiss_index/ (disk)
  └──────┬──────┘
         │
         │  User types a question
         ▼
  ┌─────────────┐
  │  Question   │──── same HuggingFace model ──► question vector
  └──────┬──────┘
         │ FAISS similarity search  (qa.py)
         ▼
  ┌─────────────┐
  │   Top 3     │  most similar chunks from the PDF
  │   Chunks    │
  └──────┬──────┘
         │ RetrievalQA builds prompt  (qa.py)
         ▼
  ┌─────────────┐
  │    Groq     │  llama-3.1-8b-instant
  │    LLM      │  cloud-based, needs GROQ_API_KEY
  └──────┬──────┘
         │
         ▼
    Answer + Source pages printed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PDF CHANGE DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
An MD5 hash of the PDF is saved in faiss_index/.pdf_hash after
every index build. On each startup, the current PDF hash is
compared to the saved one:

  Same hash  → PDF unchanged → load existing index (fast)
  Different  → PDF replaced  → rebuild index automatically

This means you never get stale answers from an old PDF.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STORAGE LAYOUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  faiss_index/
    ├── index.faiss   ← chunk vectors (binary, not human-readable)
    ├── index.pkl     ← chunk texts + page metadata (pickle)
    └── .pdf_hash     ← MD5 fingerprint of last indexed PDF
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

    current
    _hash = get_pdf_hash(PDF_PATH)
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
