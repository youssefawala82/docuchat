"""
Splits loaded documents into overlapping chunks.
Chunking matters a lot for retrieval quality: too big and you retrieve
irrelevant noise alongside the answer; too small and you lose context.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents):
    """
    Takes a list of LangChain Documents (full pages/files) and splits them
    into smaller overlapping chunks, preserving metadata (like source filename).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks