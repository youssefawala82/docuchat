import time

def build_vectorstore(chunks):
    """
    Takes a list of chunked Documents and builds a FAISS index from them.
    Embeds in small batches with pauses to stay under Gemini's free-tier
    rate limit (100 embedding requests per minute).
    """
    embeddings = get_embeddings()

    batch_size = 80  # stay safely under the 100/minute free-tier limit
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_store)

        # Pause before the next batch, unless this was the last one
        if i + batch_size < len(chunks):
            time.sleep(65)

    return vectorstore