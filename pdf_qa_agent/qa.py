"""
qa.py — Step 3: Answer a question using retrieved chunks + an LLM

Responsibilities:
- Take the user's question and find the most relevant chunks
  from the vector store (retrieval)
- Pass those chunks as context to a Groq-hosted LLM
- Return the answer and the source page numbers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  You: "What is supervised learning?"
        │
        ▼ same HuggingFace model converts question → vector
        │
  [0.10, 0.80, 0.34, ... 384 numbers]
        │
        ▼ FAISS compares against all stored chunk vectors
        │
  Top 3 most similar chunks:
    chunk 2: "supervised learning uses labeled data..."   similarity: 0.95 ✅
    chunk 5: "classification and regression are..."       similarity: 0.91 ✅
    chunk 3: "training a model on examples..."           similarity: 0.87 ✅
        │
        ▼ RetrievalQA builds this prompt automatically:
        │
  ┌──────────────────────────────────────────────────┐
  │ Use the following context to answer the question │
  │                                                  │
  │ Context:                                         │
  │   "supervised learning uses labeled data..."     │
  │   "classification and regression are..."         │
  │   "training a model on examples..."              │
  │                                                  │
  │ Question: What is supervised learning?           │
  └──────────────────────────────────────────────────┘
        │
        ▼ sent to Groq (needs GROQ_API_KEY in .env)
        │
  llama-3.1-8b-instant reads context + question
        │
        ▼
  Answer: "The pension balance is ..."
  Source pages: [0, 1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOTE: Both the question and chunks MUST use the same embedding
model so their vectors are in the same mathematical space.
If different models were used, similarity search would be meaningless.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load GROQ_API_KEY (and any other secrets) from the .env file
load_dotenv()


def get_answer(question, vectorstore):
    """
    Retrieve relevant chunks for a question and generate an answer.

    Args:
        question (str):     The user's natural-language question.
        vectorstore (FAISS): The loaded vector store from retriever.py.

    How it works (RAG pipeline):
        1. Retrieval — the question is embedded with the same model
           used during indexing. FAISS finds the 3 most similar chunks
           (k=3) by cosine similarity.
        2. Augmentation — those 3 chunks are injected into a prompt
           as context alongside the original question.
        3. Generation — the LLM (llama-3.1-8b-instant on Groq) reads
           the context and generates a grounded answer.
        4. The answer text and source page numbers are printed.
    """

    # Step 1 — wrap the vector store as a retriever that returns top-3 chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 2 — set up the LLM
    # temperature=0 makes responses deterministic (no randomness)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # Step 3 — build the RetrievalQA chain
    # RetrievalQA internally:
    #   a) calls retriever.get_relevant_documents(question)
    #   b) stuffs those docs into a prompt template
    #   c) calls llm with that prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True   # include chunks used in the answer
    )

    # Step 4 — run the chain; returns {"result": "...", "source_documents": [...]}
    result = qa_chain.invoke({"query": question})

    print("\n🤖 Answer:", result["result"])

    # Show which PDF pages the answer was drawn from
    print("📄 Source pages:", [d.metadata["page"] for d in result["source_documents"]])
