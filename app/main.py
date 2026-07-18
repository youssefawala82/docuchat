"""
DocuChat - main Streamlit entrypoint.
Run with: streamlit run app/main.py
"""

import os
import sys
import tempfile

import streamlit as st

# Allow imports like "from app.config import ..." when run via streamlit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import GOOGLE_API_KEY, INDEX_DIR
from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_documents
from app.retrieval.vectorstore import build_vectorstore, save_vectorstore, load_vectorstore
from app.generation.qa_chain import answer_question

st.set_page_config(page_title="DocuChat", page_icon="📄")
st.title("📄 DocuChat")
st.caption("Ask questions answered only from the documents you upload.")

# --- Sidebar: uploads ---
with st.sidebar:
    st.header("Setup")

    if not GOOGLE_API_KEY:
        st.error("No GOOGLE_API_KEY found. Add it to your .env file.")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, or DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )
    build_button = st.button("Build / Rebuild Index")

# --- Session state ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore(INDEX_DIR)
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.vectorstore is not None and not build_button:
    st.sidebar.success("Loaded a previously saved index.")


def process_uploaded_files(files):
    """Save uploaded files to temp paths, load + chunk + embed them."""
    all_docs = []
    for f in files:
        suffix = os.path.splitext(f.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        docs = load_document(tmp_path)
        # Fix the source name to the ORIGINAL filename, not the temp path
        for d in docs:
            d.metadata["source"] = f.name
        all_docs.extend(docs)

        os.unlink(tmp_path)

    chunks = chunk_documents(all_docs)
    vectorstore = build_vectorstore(chunks)
    return vectorstore, len(chunks)


# --- Build index ---
if build_button:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one document.")
    else:
        with st.spinner("Reading, chunking, and embedding documents..."):
            vectorstore, n_chunks = process_uploaded_files(uploaded_files)
            save_vectorstore(vectorstore, INDEX_DIR)
            st.session_state.vectorstore = vectorstore
        st.sidebar.success(f"Index built from {n_chunks} chunks and saved.")

# --- Chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    if st.session_state.vectorstore is None:
        st.error("Please build the index first (upload documents + click 'Build / Rebuild Index').")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_question(st.session_state.vectorstore, user_question)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.markdown(f"- {s}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })