# DocuChat

A RAG (Retrieval-Augmented Generation) chatbot that answers questions using
only the content of documents you upload — no hallucinated answers, no
outside knowledge. Upload PDF, TXT, or DOCX files and ask questions in a
chat interface; the bot retrieves the most relevant passages and generates
an answer grounded in them, citing which file each answer came from.

## Demo

Upload a document, ask a question, get an answer with sources:

\```
You: How many vacation days do employees get after 3 years?
Bot: Based on the provided documents, after three years of employment,
     full-time employees accrue 23 days of paid vacation per year.
     Sources: test-handbook.txt
\```

If the answer isn't in the uploaded documents, it says so instead of
making something up:

\```
You: What is the CEO's email address?
Bot: I don't know based on the provided documents.
\```

## Architecture

\```mermaid
flowchart TD
    subgraph Ingestion
        A[Upload PDF/TXT/DOCX] --> B[Parse to text]
        B --> C[Chunk with overlap]
        C --> D[Embed chunks - Gemini]
        D --> E[(FAISS vector index)]
    end

    subgraph Query
        F[User question] --> G[Embed question]
        G --> H[Retrieve top-k similar chunks]
        E --> H
        H --> I[Build grounded prompt]
        I --> J[Generate answer - Gemini]
        J --> K[Answer + cited sources]
    end
\```

## Tech Stack

| Layer | Choice |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| LLM | Google Gemini (`gemini-flash-latest`) |
| Vector store | FAISS (local, persisted to disk) |
| Document parsing | pypdf, docx2txt |

Gemini was chosen over OpenAI specifically because Google's API offers a
genuinely free tier with no billing/credit card required, making the
project runnable by anyone without cost.

## Project Structure

\```
docuchat/
├── app/
│   ├── main.py                 # Streamlit entrypoint
│   ├── config.py                # model names, chunk size, paths
│   ├── ingestion/
│   │   ├── loaders.py           # PDF/TXT/DOCX -> LangChain Documents
│   │   └── chunker.py           # overlapping text splitting
│   ├── retrieval/
│   │   ├── vectorstore.py       # build/save/load FAISS index
│   │   └── retriever.py         # top-k similarity search
│   └── generation/
│       ├── prompts.py           # grounded-answer prompt template
│       └── qa_chain.py          # retrieval + generation pipeline
├── data/
│   ├── raw/                     # (gitignored) source documents
│   └── index/                   # (gitignored) saved FAISS index
├── evaluation/                  # test questions + scoring (in progress)
├── tests/                       # unit tests (in progress)
└── requirements.txt
\```

## Setup

**1. Clone and install dependencies**

\```bash
git clone https://github.com/youssefawala82/docuchat.git
cd docuchat
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
\```

**2. Get a free Gemini API key**

Create one at [Google AI Studio](https://aistudio.google.com/apikey) — no
billing required for the free tier.

**3. Set up your environment file**

Copy `.env.example` to `.env` and add your key:

\```
GOOGLE_API_KEY=your-key-here
\```

**4. Run the app**

\```bash
streamlit run app/main.py
\```

Opens at `http://localhost:8501`. Upload a document, click **Build /
Rebuild Index**, and start asking questions.

## Design Decisions

- **Chunking strategy**: chunks are split with a 1000-character size and
  150-character overlap, using an ordered separator list (paragraph →
  line → sentence → word) so splits happen at natural boundaries rather
  than mid-sentence wherever possible.
- **Hallucination control**: the prompt explicitly instructs the model to
  answer only from retrieved context and to say "I don't know" when the
  answer isn't present, rather than filling gaps with outside knowledge.
- **Persistence**: the FAISS index is saved to disk after each build, so
  the app doesn't need to re-embed documents on every restart.
- **Source citation**: every chunk is tagged with its originating filename
  at load time, so every answer can show exactly which document(s) it
  drew from.

## Known Limitations

- The FAISS index is rebuilt from scratch each time new documents are
  uploaded (no incremental updates yet).
- Large documents (100+ pages) will be slower to index due to embedding
  API call volume.
- No authentication — intended for local/personal use, not multi-user
  deployment as-is.

## Roadmap

- [ ] Evaluation harness with measured retrieval precision and answer
      faithfulness on a fixed test set
- [ ] Automated tests for chunking and retrieval
- [ ] Hybrid search (FAISS + keyword search)
- [ ] Deployed live demo

## License

MIT