"""
Build Knowledge Base — Master wrapper script for the entire data pipeline.

Orchestrates all three phases sequentially:
  1. Phase 1: TMDB Fetcher (fetch metadata, posters, stills)
  2. Phase 2: Caption Generator (auto-caption images with Gemini Flash)
  3. Phase 3: KB Builder (embed and index into ChromaDB)

If any phase fails, stops with helpful error message and exits code 1.
On success, validates the KB and reports totals.

Usage:
    python src/pipeline/build_kb.py

Exit codes:
    0 = Success
    1 = Phase failed
    2 = Validation failed
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIR,
    TEXT_COLLECTION_NAME,
    IMAGE_COLLECTION_NAME,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_phase(phase_name: str, module_path: str, function_name: str) -> bool:
    """
    Import and execute a single pipeline phase.

    Imports the module dynamically, calls the phase function, and handles
    any exceptions. Logs success/failure with phase name.

    Args:
        phase_name: Human-readable phase name (e.g., "TMDB Fetcher")
        module_path: Module import path (e.g., "pipeline.tmdb_fetcher")
        function_name: Function to call in the module (e.g., "run_pipeline")

    Returns:
        True if phase succeeded, False if it failed
    """
    logger.info(f"Starting Phase: {phase_name}")
    logger.info(f"  Module: {module_path}")
    logger.info(f"  Function: {function_name}")

    try:
        # Dynamic import
        module = __import__(module_path, fromlist=[function_name])
        phase_func = getattr(module, function_name)

        # Execute
        start_time = time.time()
        phase_func()
        elapsed = time.time() - start_time

        logger.info(f"✓ Phase complete in {elapsed:.1f}s")
        return True

    except ImportError as e:
        logger.error(f"✗ Failed to import {module_path}: {e}")
        return False
    except AttributeError as e:
        logger.error(f"✗ Function {function_name} not found in {module_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Phase failed with exception: {e}", exc_info=True)
        return False


def validate_kb() -> bool:
    """
    Validate the knowledge base after all phases complete.

    Checks:
      1. ChromaDB collections exist and have documents
      2. Smoke test: query both collections to ensure they're searchable

    Returns:
        True if KB is valid, False if validation failed
    """
    logger.info("Validating knowledge base...")

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Load CLIP model for image collection queries
        from config import IMAGE_EMBEDDING_MODEL
        clip_model = SentenceTransformer(IMAGE_EMBEDDING_MODEL)

        # Check collections exist
        text_col = client.get_collection(TEXT_COLLECTION_NAME)
        image_col = client.get_collection(IMAGE_COLLECTION_NAME)

        text_count = text_col.count()
        image_count = image_col.count()

        logger.info(f"Text collection: {text_count} documents")
        logger.info(f"Image collection: {image_count} documents")

        # Validate counts
        if text_count == 0:
            logger.error("Text collection is empty")
            return False
        if image_count == 0:
            logger.error("Image collection is empty")
            return False

        # Smoke test: query each collection
        logger.info("Running smoke test queries...")

        # Text query
        text_results = text_col.query(query_texts=["drama film"], n_results=1)
        if not text_results["ids"] or not text_results["ids"][0]:
            logger.error("Text collection query returned no results")
            return False

        logger.info(f"  Text query found: {text_results['ids'][0][0]}")

        # Image query (using CLIP text embeddings)
        query_text = "visual style"
        query_embedding = clip_model.encode([query_text]).tolist()
        image_results = image_col.query(query_embeddings=query_embedding, n_results=1)
        if not image_results["ids"] or not image_results["ids"][0]:
            logger.error("Image collection query returned no results")
            return False

        logger.info(f"  Image query found: {image_results['ids'][0][0]}")

        logger.info(f"✓ KB validation passed ({text_count} text, {image_count} images)")
        return True

    except Exception as e:
        logger.error(f"✗ KB validation failed: {e}", exc_info=True)
        return False


def main() -> int:
    """
    Main entry point. Runs all three phases sequentially.

    Returns:
        0 on success, 1 if any phase fails, 2 if validation fails
    """
    logger.info("=" * 70)
    logger.info("CineAgent Knowledge Base Builder")
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    start_time = time.time()

    # ── Phase 1: TMDB Fetcher ─────────────────────────────────────────────────
    if not run_phase(
        "TMDB Fetcher",
        "pipeline.tmdb_fetcher",
        "run_pipeline",
    ):
        logger.error("Pipeline failed at Phase 1 (TMDB Fetcher)")
        return 1

    # ── Phase 2: Caption Generator ────────────────────────────────────────────
    if not run_phase(
        "Caption Generator",
        "pipeline.caption_generator",
        "run_captioning",
    ):
        logger.error("Pipeline failed at Phase 2 (Caption Generator)")
        return 1

    # ── Phase 3: KB Builder ───────────────────────────────────────────────────
    if not run_phase(
        "Knowledge Base Builder",
        "pipeline.kb_builder",
        "run_kb_builder",
    ):
        logger.error("Pipeline failed at Phase 3 (KB Builder)")
        return 1

    # ── Validation ────────────────────────────────────────────────────────────
    if not validate_kb():
        logger.error("KB validation failed")
        return 2

    # ── Success ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"✓ Pipeline complete in {elapsed:.1f}s")
    logger.info("✓ KB is ready for retrieval and agent use")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
