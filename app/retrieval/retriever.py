"""
Wraps the FAISS vectorstore's search functionality into a simple retriever.
"""

from app.config import TOP_K


def get_retriever(vectorstore, k: int = TOP_K):
    """
    Returns a LangChain retriever object from the given vectorstore.
    'k' controls how many chunks are pulled back per question.
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_chunks(vectorstore, query: str, k: int = TOP_K):
    """
    Directly retrieves the top-k most relevant chunks for a query.
    Useful for debugging/evaluation, separate from the full QA chain.
    """
    retriever = get_retriever(vectorstore, k)
    return retriever.invoke(query)