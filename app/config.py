"""
Central configuration for DocuChat.
Keeping these in one place makes it easy to tune the system
and to reference these values in the README/evaluation report.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# --- Models ---
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-flash-latest"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retrieval ---
TOP_K = 4

# --- Paths ---
DATA_RAW_DIR = "data/raw"
INDEX_DIR = "data/index"