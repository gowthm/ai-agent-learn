"""
retriever.py — Step 2: Embed chunks and store/load a FAISS vector index

Responsibilities:
- Convert text chunks into numeric vectors (embeddings)
- Store those vectors in a FAISS index for fast similarity search
- Save the index to disk so it is built only once
- Reload the index on subsequent runs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS AN EMBEDDING?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The embedding model converts text into a list of 384 numbers
that capture the *meaning* of the text. Similar meaning = similar
numbers, so we can search by meaning rather than exact keywords:

  "employee contribution"  → [0.12, 0.85, 0.33, ...]  384 numbers
  "worker deposit"         → [0.11, 0.83, 0.35, ...]  ← close! (similar meaning)
  "cricket score"          → [0.91, 0.02, 0.77, ...]  ← far away (different meaning)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY HUGGINGFACE? (no API key needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HuggingFace (used here)     vs     OpenAI Embeddings
  ────────────────────────           ─────────────────
  No API key required                Needs OPENAI_API_KEY
  Free forever                       Paid per request
  Runs on your CPU locally           Sent to OpenAI servers
  Data never leaves your machine     Data sent externally

  The model (all-MiniLM-L6-v2, ~87MB) is downloaded once to:
    C:\\Users\\<you>\\.cache\\huggingface\\hub\\
  After that it runs fully offline — no internet, no cost.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS FAISS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAISS is a vector index — the core concept behind vector databases.
It stores all chunk vectors and can find the most similar ones to
a query vector in milliseconds, even with thousands of chunks.

  FAISS (used here)              Full Vector DB (Pinecone, Chroma)
  ─────────────────              ─────────────────────────────────
  Local files on disk            Dedicated server / cloud
  Zero setup                     Needs a running server
  Good for small-medium data     Scales to millions of vectors

  Files saved to disk:
    faiss_index/index.faiss  ← vectors (binary, not human-readable)
    faiss_index/index.pkl    ← original chunk texts + page metadata
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    """
    Load the sentence-transformer embedding model.

    Returns:
        HuggingFaceEmbeddings: An embedding model that converts
                               text into 384-dimensional vectors.

    Model used:
        'all-MiniLM-L6-v2' — a lightweight but accurate model
        that runs entirely locally (no API key needed).
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_vectorstore(chunks):
    """
    Embed all chunks and build a FAISS vector index, then save it to disk.

    Args:
        chunks (list[Document]): Chunked documents from ingest.py.

    Returns:
        FAISS: An in-memory vector store ready for similarity search.

    How it works:
        1. Each chunk's text is passed through the embedding model,
           producing a float vector.
        2. All vectors are stored in a FAISS index (an efficient
           nearest-neighbour data structure).
        3. The index is saved to the 'faiss_index/' folder so the
           next run can skip this expensive step.
    """
    embeddings = get_embeddings()

    # FAISS.from_documents embeds every chunk and indexes them
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist to disk for reuse
    vectorstore.save_local("faiss_index")
    print("✅ Vector store created and saved")
    return vectorstore


def load_vectorstore():
    """
    Load a previously saved FAISS index from disk.

    Returns:
        FAISS: The restored vector store, ready for similarity search.

    Note:
        'allow_dangerous_deserialization=True' is required because
        FAISS uses pickle internally. It is safe here because we
        created the index ourselves.
    """
    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True  # safe — we built this index
    )
    print("✅ Vector store loaded")
    return vectorstore
