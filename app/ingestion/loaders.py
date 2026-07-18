"""
Loaders for turning uploaded PDF/TXT/DOCX files into LangChain Document objects.
Each Document has .page_content (the text) and .metadata (like source filename).
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader


def load_document(file_path: str):
    """
    Load a single file (PDF, TXT, or DOCX) into a list of LangChain Documents.
    Raises ValueError for unsupported file types.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()

    # Tag every chunk with its source filename so we can cite it later
    filename = os.path.basename(file_path)
    for doc in docs:
        doc.metadata["source"] = filename

    return docs


def load_documents_from_folder(folder_path: str):
    """
    Load every supported file in a folder and return one combined list
    of Documents.
    """
    all_docs = []
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext in (".pdf", ".txt", ".docx"):
            all_docs.extend(load_document(full_path))
    return all_docs