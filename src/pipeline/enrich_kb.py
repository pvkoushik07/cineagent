"""
KB Enrichment Script

Enriches existing TMDB data with:
1. Keywords/themes (from TMDB keywords endpoint)
2. Reviews (from TMDB reviews endpoint)
3. Enhanced plot descriptions

This improves semantic search by adding thematic content.

Usage:
    python src/pipeline/enrich_kb.py
"""

import json
import logging
import time
from pathlib import Path
import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TMDB_API_KEY,
    TMDB_BASE_URL,
    RAW_DIR,
    PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_keywords(film_id: int) -> list[str]:
    """
    Fetch thematic keywords for a film from TMDB.

    Args:
        film_id: TMDB film ID

    Returns:
        List of keyword strings (e.g., ["social-commentary", "class-struggle"])
    """
    try:
        url = f"{TMDB_BASE_URL}/movie/{film_id}/keywords"
        params = {"api_key": TMDB_API_KEY}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        keywords = [kw["name"] for kw in data.get("keywords", [])]
        return keywords

    except Exception as e:
        logger.warning(f"Failed to fetch keywords for film {film_id}: {e}")
        return []


def fetch_reviews(film_id: int, max_reviews: int = 3) -> list[dict]:
    """
    Fetch user reviews for a film from TMDB.

    Args:
        film_id: TMDB film ID
        max_reviews: Maximum number of reviews to fetch

    Returns:
        List of review dicts with 'author' and 'content' fields
    """
    try:
        url = f"{TMDB_BASE_URL}/movie/{film_id}/reviews"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        reviews = []
        for review in data.get("results", [])[:max_reviews]:
            reviews.append({
                "author": review.get("author", "Unknown"),
                "content": review.get("content", "")[:500],  # First 500 chars
            })

        return reviews

    except Exception as e:
        logger.warning(f"Failed to fetch reviews for film {film_id}: {e}")
        return []


def enrich_film_data(film_path: Path) -> dict:
    """
    Enrich a single film's JSON data with keywords and reviews.

    Args:
        film_path: Path to film's JSON file

    Returns:
        Enriched film data dict
    """
    # Load existing data
    with open(film_path) as f:
        film_data = json.load(f)

    film_id = film_data.get("id")
    if not film_id:
        logger.warning(f"No film ID in {film_path}")
        return film_data

    # Check if already enriched
    if "enriched" in film_data and film_data["enriched"]:
        logger.debug(f"Film {film_id} already enriched, skipping")
        return film_data

    # Fetch keywords
    keywords = fetch_keywords(film_id)
    film_data["keywords"] = keywords

    # Fetch reviews
    reviews = fetch_reviews(film_id, max_reviews=2)
    film_data["reviews"] = reviews

    # Mark as enriched
    film_data["enriched"] = True

    # Save back to file
    with open(film_path, "w") as f:
        json.dump(film_data, f, indent=2)

    logger.info(f"Enriched {film_data.get('title', 'Unknown')} ({film_id}): "
                f"{len(keywords)} keywords, {len(reviews)} reviews")

    return film_data


def enrich_all_films(batch_size: int = 10, delay: float = 0.5):
    """
    Enrich all films in data/raw/ with keywords and reviews.

    Args:
        batch_size: Number of films to process before showing progress
        delay: Seconds to wait between API calls (rate limiting)
    """
    json_files = sorted(RAW_DIR.glob("*.json"))
    logger.info(f"Found {len(json_files)} films to enrich")

    enriched_count = 0
    skipped_count = 0

    for film_path in tqdm(json_files, desc="Enriching films"):
        try:
            # Load and check if already enriched
            with open(film_path) as f:
                film_data = json.load(f)

            if film_data.get("enriched"):
                skipped_count += 1
                continue

            # Enrich
            enrich_film_data(film_path)
            enriched_count += 1

            # Rate limiting
            time.sleep(delay)

        except Exception as e:
            logger.error(f"Failed to enrich {film_path}: {e}")
            continue

    logger.info(f"Enrichment complete: {enriched_count} enriched, {skipped_count} skipped")

    # Create enrichment summary
    summary = {
        "total_films": len(json_files),
        "enriched": enriched_count,
        "skipped": skipped_count,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_path = PROCESSED_DIR / "enrichment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Enrichment summary saved to {summary_path}")


if __name__ == "__main__":
    logger.info("Starting KB enrichment...")
    enrich_all_films(delay=0.5)
    logger.info("Enrichment complete! Next: re-run kb_builder.py to rebuild indices")
