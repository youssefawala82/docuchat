"""
Combines the retriever and the LLM into a single question-answering chain.
Given a question, it retrieves relevant chunks, builds a prompt, calls Gemini,
and returns both the answer and which source documents were used.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GOOGLE_API_KEY, CHAT_MODEL
from app.generation.prompts import QA_PROMPT
from app.retrieval.retriever import get_retriever


def get_llm():
    """Returns the Gemini chat model used for answer generation."""
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )


def answer_question(vectorstore, question: str):
    """
    Runs the full RAG pipeline for a single question:
    1. Retrieve relevant chunks from the vectorstore
    2. Build a prompt with those chunks as context
    3. Call Gemini to generate an answer
    4. Return the answer plus the list of source filenames used

    Returns a dict: {"answer": str, "sources": list[str]}
    """
    retriever = get_retriever(vectorstore)
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt_text = QA_PROMPT.format(context=context, question=question)

    llm = get_llm()
    response = llm.invoke(prompt_text)

    sources = sorted(set(
        doc.metadata.get("source", "unknown") for doc in retrieved_docs
    ))

    # Gemini responses can come back as a plain string or as a list of
    # content blocks (e.g. [{"type": "text", "text": "..."}]).
    # Normalize to a plain string either way.
    if isinstance(response.content, list):
        answer_text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in response.content
        )
    else:
        answer_text = response.content

    return {
        "answer": answer_text,
        "sources": sources,
    }