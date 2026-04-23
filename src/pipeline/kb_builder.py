"""
Knowledge Base Builder — Phase 1 data pipeline.

Reads processed film data and captions, embeds everything, and stores in
two ChromaDB collections:
  - cineagent_text: plot text + reviews + captions (MiniLM embeddings)
  - cineagent_images: posters + stills (CLIP embeddings)

This is the final step of the data pipeline. Run after tmdb_fetcher.py
and caption_generator.py.

Usage:
    python src/pipeline/kb_builder.py
"""

import json
import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIR,
    TEXT_COLLECTION_NAME,
    IMAGE_COLLECTION_NAME,
    TEXT_EMBEDDING_MODEL,
    IMAGE_EMBEDDING_MODEL,
    RAW_DIR,
    PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_film_documents(raw_dir: Path, captions: dict) -> list[dict]:
    """
    Convert raw TMDB JSON files into structured documents ready for indexing.

    Each film produces multiple documents:
      - 1 plot_text document
      - 1 reviews_text document (placeholder — extend with real reviews)
      - 1 poster_caption document
      - N still_caption documents

    Args:
        raw_dir: Path to data/raw/ directory
        captions: Dict mapping image_key → caption string

    Returns:
        List of document dicts with fields: doc_id, film_id, title,
        modality, content, metadata
    """
    documents = []

    for json_path in sorted(raw_dir.glob("*.json")):
        with open(json_path) as f:
            film = json.load(f)

        film_id = str(film.get("id", ""))
        title = film.get("title", "Unknown")
        year = film.get("release_date", "")[:4] if film.get("release_date") else ""
        genres = [g["name"] for g in film.get("genres", [])]
        directors = film.get("directors", [])
        cast = film.get("top_cast", [])
        language = film.get("original_language", "")
        runtime = film.get("runtime", 0)
        rating = film.get("vote_average", 0.0)

        base_metadata = {
            "film_id": film_id,
            "title": title,
            "year": int(year) if year.isdigit() else 0,
            "directors": ", ".join(directors),
            "genres": ", ".join(genres),
            "language": language,
            "runtime_mins": runtime,
            "tmdb_rating": rating,
            "cast_top3": ", ".join(cast),
        }

        # 1. Plot text document
        overview = film.get("overview", "")
        if overview:
            documents.append({
                "doc_id": f"{film_id}_plot",
                "film_id": film_id,
                "title": title,
                "modality": "text",
                "doc_type": "plot",
                "content": f"{title} ({year}). Directed by {', '.join(directors)}. "
                           f"Cast: {', '.join(cast)}. Genres: {', '.join(genres)}. "
                           f"Plot: {overview}",
                "metadata": {**base_metadata, "doc_type": "plot"},
            })

        # 2. Poster caption document
        poster_key = f"{film_id}_poster"
        if poster_key in captions:
            documents.append({
                "doc_id": f"{film_id}_poster_caption",
                "film_id": film_id,
                "title": title,
                "modality": "caption",
                "doc_type": "poster_caption",
                "content": f"Movie poster for '{title}': {captions[poster_key]}",
                "metadata": {**base_metadata, "doc_type": "poster_caption",
                             "image_key": poster_key},
            })

        # 3. Still caption documents
        for i in range(3):
            still_key = f"{film_id}_still_{i}"
            if still_key in captions:
                documents.append({
                    "doc_id": f"{film_id}_still_{i}_caption",
                    "film_id": film_id,
                    "title": title,
                    "modality": "caption",
                    "doc_type": "still_caption",
                    "content": f"Scene from '{title}': {captions[still_key]}",
                    "metadata": {**base_metadata, "doc_type": "still_caption",
                                 "image_key": still_key, "still_index": i},
                })

    logger.info(f"Built {len(documents)} text/caption documents from {len(list(raw_dir.glob('*.json')))} films")
    return documents


def build_image_documents(raw_dir: Path) -> list[dict]:
    """
    Build image document records for all poster and still images.

    Args:
        raw_dir: Path to data/raw/ directory

    Returns:
        List of image document dicts with image_path and metadata
    """
    images_dir = raw_dir / "images"
    image_docs = []

    for json_path in sorted(raw_dir.glob("*.json")):
        with open(json_path) as f:
            film = json.load(f)

        film_id = str(film.get("id", ""))
        title = film.get("title", "Unknown")
        year = film.get("release_date", "")[:4] if film.get("release_date") else ""
        genres = [g["name"] for g in film.get("genres", [])]
        directors = film.get("directors", [])

        base_metadata = {
            "film_id": film_id,
            "title": title,
            "year": int(year) if year.isdigit() else 0,
            "directors": ", ".join(directors),
            "genres": ", ".join(genres),
        }

        # Poster image
        poster_path = images_dir / f"{film_id}_poster.jpg"
        if poster_path.exists():
            image_docs.append({
                "doc_id": f"{film_id}_poster",
                "film_id": film_id,
                "image_type": "poster",
                "image_path": str(poster_path),
                "metadata": {**base_metadata, "image_type": "poster"},
            })

        # Still images
        for i in range(3):
            still_path = images_dir / f"{film_id}_still_{i}.jpg"
            if still_path.exists():
                image_docs.append({
                    "doc_id": f"{film_id}_still_{i}",
                    "film_id": film_id,
                    "image_type": "still",
                    "image_path": str(still_path),
                    "metadata": {**base_metadata, "image_type": "still",
                                 "still_index": i},
                })

    logger.info(f"Found {len(image_docs)} image documents")
    return image_docs


def index_text_collection(
    client: chromadb.PersistentClient,
    documents: list[dict],
    model: SentenceTransformer,
) -> None:
    """
    Embed and store all text/caption documents in ChromaDB text collection.

    Args:
        client: ChromaDB persistent client
        documents: List of document dicts from build_film_documents()
        model: SentenceTransformer model for text embedding
    """
    collection = client.get_or_create_collection(
        name=TEXT_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Skip documents already indexed
    existing_ids = set(collection.get()["ids"])
    new_docs = [d for d in documents if d["doc_id"] not in existing_ids]

    if not new_docs:
        logger.info("Text collection already up to date")
        return

    logger.info(f"Indexing {len(new_docs)} new text documents...")

    batch_size = 100
    for i in tqdm(range(0, len(new_docs), batch_size), desc="Text indexing"):
        batch = new_docs[i : i + batch_size]
        contents = [d["content"] for d in batch]
        embeddings = model.encode(contents, show_progress_bar=False).tolist()

        collection.add(
            ids=[d["doc_id"] for d in batch],
            embeddings=embeddings,
            documents=contents,
            metadatas=[d["metadata"] for d in batch],
        )

    logger.info(f"Text collection size: {collection.count()} documents")


def index_image_collection(
    client: chromadb.PersistentClient,
    image_docs: list[dict],
    model: SentenceTransformer,
) -> None:
    """
    Embed and store all images in ChromaDB image collection using CLIP.

    Args:
        client: ChromaDB persistent client
        image_docs: List of image document dicts from build_image_documents()
        model: CLIP SentenceTransformer model for image embedding
    """
    from PIL import Image

    collection = client.get_or_create_collection(
        name=IMAGE_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get()["ids"])
    new_docs = [d for d in image_docs if d["doc_id"] not in existing_ids]

    if not new_docs:
        logger.info("Image collection already up to date")
        return

    logger.info(f"Indexing {len(new_docs)} new images with CLIP...")

    batch_size = 32  # Smaller batches for image embedding (memory)
    for i in tqdm(range(0, len(new_docs), batch_size), desc="Image indexing"):
        batch = new_docs[i : i + batch_size]
        images = []
        valid_batch = []

        for doc in batch:
            try:
                img = Image.open(doc["image_path"]).convert("RGB")
                images.append(img)
                valid_batch.append(doc)
            except Exception as e:
                logger.warning(f"Could not open image {doc['image_path']}: {e}")

        if not images:
            continue

        embeddings = model.encode(images, show_progress_bar=False).tolist()

        collection.add(
            ids=[d["doc_id"] for d in valid_batch],
            embeddings=embeddings,
            documents=[d["image_path"] for d in valid_batch],  # store path as document
            metadatas=[d["metadata"] for d in valid_batch],
        )

    logger.info(f"Image collection size: {collection.count()} documents")


def run_kb_builder() -> None:
    """
    Main entry point. Builds both ChromaDB collections from raw data.
    """
    logger.info("Starting knowledge base builder...")

    # Load captions
    captions_file = PROCESSED_DIR / "captions.json"
    if not captions_file.exists():
        logger.warning("captions.json not found — run caption_generator.py first")
        captions = {}
    else:
        with open(captions_file) as f:
            captions = json.load(f)
        logger.info(f"Loaded {len(captions)} captions")

    # Build document lists
    text_docs = build_film_documents(RAW_DIR, captions)
    image_docs = build_image_documents(RAW_DIR)

    # Save processed documents for inspection
    processed_file = PROCESSED_DIR / "films_processed.json"
    with open(processed_file, "w") as f:
        json.dump(text_docs[:10], f, indent=2)  # Save sample for debugging
    logger.info(f"Saved sample to {processed_file}")

    # Initialise ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Load embedding models
    logger.info(f"Loading text model: {TEXT_EMBEDDING_MODEL}")
    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)

    logger.info(f"Loading image model: {IMAGE_EMBEDDING_MODEL}")
    image_model = SentenceTransformer(IMAGE_EMBEDDING_MODEL)

    # Index both collections
    index_text_collection(client, text_docs, text_model)
    index_image_collection(client, image_docs, image_model)

    # Final report
    text_col = client.get_collection(TEXT_COLLECTION_NAME)
    image_col = client.get_collection(IMAGE_COLLECTION_NAME)
    logger.info(f"KB build complete:")
    logger.info(f"  text_collection: {text_col.count()} documents")
    logger.info(f"  image_collection: {image_col.count()} documents")


if __name__ == "__main__":
    run_kb_builder()
