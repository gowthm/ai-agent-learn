"""
ingest.py — Step 1: Load and chunk the PDF

Responsibilities:
- Read the PDF file from disk
- Split the text into smaller overlapping chunks
  so that relevant pieces can be retrieved efficiently

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  data/sample.pdf
        │
        ▼ PyPDFLoader — reads page by page
        │
  Page 1: "Introduction to machine learning..."
  Page 2: "Supervised learning algorithms..."
        │
        ▼ RecursiveCharacterTextSplitter
        │
  chunk 1: "Introduction to machine learning..."         (500 chars)
  chunk 2: "machine learning covers supervised..."        (500 chars, 50 overlap)
  chunk 3: "supervised learning algorithms include..."    (500 chars, 50 overlap)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY OVERLAP?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Without overlap, a sentence split across two chunks would lose
context at the boundary. With 50-char overlap, the tail of
chunk N is repeated at the head of chunk N+1:

  chunk 1: "...this algorithm works by"
  chunk 2: "algorithm works by fitting a model..."  ← repeated 50 chars
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk(pdf_path):
    """
    Load a PDF and split it into chunks suitable for embedding.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list[Document]: A list of LangChain Document objects,
                        each containing a chunk of text and
                        metadata (e.g. page number).

    How it works:
        1. PyPDFLoader reads each page of the PDF.
        2. RecursiveCharacterTextSplitter breaks the full text into
           chunks of ~500 characters, with 50-character overlaps so
           that sentences/context at boundaries are not lost.
    """

    # Load all pages from the PDF as LangChain Document objects
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()   # returns one Document per page

    # Split each page's text into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # max characters per chunk
        chunk_overlap=50    # characters shared between consecutive chunks
                            # (prevents losing context at boundaries)
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks
