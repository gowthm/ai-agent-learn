"""
ingest.py — Step 1: Load and chunk the PDF

Responsibilities:
- Read the PDF file from disk
- Split the text into smaller overlapping chunks
  so that relevant pieces can be retrieved efficiently
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
