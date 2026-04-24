"""
Central configuration for CineAgent.
All paths and settings live here — never hardcode paths in other modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDICES_DIR = DATA_DIR / "indices"
RESULTS_DIR = DATA_DIR / "results"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, INDICES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Personal film list ────────────────────────────────────────────────────────
PERSONAL_FILMS_PATH = DATA_DIR / "personal_films.json"
FAILED_FILMS_PATH = DATA_DIR / "failed_films.json"

# ── API Keys ──────────────────────────────────────────────────────────────────
TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── TMDB Settings ─────────────────────────────────────────────────────────────
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_STILL_BASE_URL = "https://image.tmdb.org/t/p/w780"
FILMS_TARGET_COUNT = 500
STILLS_PER_FILM = 3

# ── Embedding Models ──────────────────────────────────────────────────────────
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
IMAGE_EMBEDDING_MODEL = "clip-ViT-B-32"

# ── ChromaDB Settings ─────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = str(INDICES_DIR)
TEXT_COLLECTION_NAME = "cineagent_text"
IMAGE_COLLECTION_NAME = "cineagent_images"

# ── LLM Settings ─────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-1.5-flash"
OLLAMA_MODEL = "llava"
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.3  # Lower = more grounded, less creative. Good for RAG.

# ── Retrieval Settings ────────────────────────────────────────────────────────
TOP_K = 5           # Recall@5 requires top-5 retrieval
RRF_K = 60          # Standard RRF constant
MIN_RELEVANCE_SCORE = 0.3  # Filter out low-confidence results

# ── Evaluation Settings ───────────────────────────────────────────────────────
EVAL_RESULTS_FILE = RESULTS_DIR / "eval_results.json"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
