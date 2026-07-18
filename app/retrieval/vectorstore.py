"""
Builds, saves, and loads the FAISS vector index using Gemini embeddings.
"""

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import GOOGLE_API_KEY, EMBEDDING_MODEL, INDEX_DIR


def get_embeddings():
    """Returns the embeddings model used to convert text into vectors."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def build_vectorstore(chunks):
    """
    Takes a list of chunked Documents and builds a FAISS index from them.
    Returns the FAISS vectorstore object (kept in memory).
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vectorstore, path: str = INDEX_DIR):
    """Persists the FAISS index to disk so it doesn't need rebuilding every run."""
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)


def load_vectorstore(path: str = INDEX_DIR):
    """
    Loads a previously saved FAISS index from disk.
    Returns None if no saved index exists yet.
    """
    if not os.path.exists(path) or not os.listdir(path):
        return None

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore