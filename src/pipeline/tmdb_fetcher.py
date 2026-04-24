"""
TMDB Fetcher — Phase 1 data pipeline.

Fetches film metadata, poster images, and scene stills from the TMDB API.
Saves raw JSON responses to data/raw/ and images to data/raw/images/.

Usage:
    python src/pipeline/tmdb_fetcher.py
"""

import json
import logging
import time
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TMDB_API_KEY,
    TMDB_BASE_URL,
    TMDB_IMAGE_BASE_URL,
    TMDB_STILL_BASE_URL,
    RAW_DIR,
    FILMS_TARGET_COUNT,
    STILLS_PER_FILM,
    PERSONAL_FILMS_PATH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGES_DIR = RAW_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_personal_films_ids(file_path: Path = PERSONAL_FILMS_PATH) -> list[int]:
    """
    Load manually curated film IDs from personal_films.json.

    Args:
        file_path: Path to personal_films.json

    Returns:
        List of TMDB film IDs
    """
    if not file_path.exists():
        logger.warning(f"{file_path} not found, using auto-discovery only")
        return []

    try:
        with open(file_path) as f:
            data = json.load(f)

        film_ids = [film["id"] for film in data]
        logger.info(f"Loaded {len(film_ids)} personal films from {file_path}")
        return film_ids

    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []


def fetch_film_metadata(film_id: int) -> dict:
    """
    Fetch full metadata for a single film from TMDB.

    Args:
        film_id: TMDB integer film ID

    Returns:
        Dict with title, overview, genres, director, cast, release_date, etc.
    """
    url = f"{TMDB_BASE_URL}/movie/{film_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits,images",
        "include_image_language": "en,null",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_popular_film_ids(pages: int = 25) -> list[int]:
    """
    Fetch a list of film IDs from TMDB discover endpoint.

    Uses multiple sort strategies to get a diverse, not just popular, set.
    Target: ~500 film IDs across different genres, years, languages.

    Args:
        pages: number of pages to fetch (20 films per page)

    Returns:
        List of TMDB film IDs
    """
    film_ids: list[int] = []

    # Strategy 1: top rated (quality films)
    for page in range(1, pages + 1):
        url = f"{TMDB_BASE_URL}/movie/top_rated"
        params = {"api_key": TMDB_API_KEY, "page": page}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        film_ids.extend([f["id"] for f in data.get("results", [])])
        time.sleep(0.25)  # TMDB rate limit: 40 req/10s

    logger.info(f"Fetched {len(film_ids)} film IDs from TMDB")
    return list(set(film_ids))[:FILMS_TARGET_COUNT]


def download_image(url: str, save_path: Path) -> bool:
    """
    Download a single image from a URL and save to disk.

    Args:
        url: Full image URL
        save_path: Path to save the image file

    Returns:
        True if successful, False if failed
    """
    if save_path.exists():
        return True  # Skip already downloaded

    try:
        resp = requests.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def fetch_and_save_film(film_id: int) -> dict | None:
    """
    Fetch metadata + images for one film and save everything to disk.

    Saves:
        - data/raw/{film_id}.json — full TMDB metadata
        - data/raw/images/{film_id}_poster.jpg — poster image
        - data/raw/images/{film_id}_still_0.jpg etc — scene stills

    Args:
        film_id: TMDB film ID

    Returns:
        Processed film dict, or None if fetch failed
    """
    json_path = RAW_DIR / f"{film_id}.json"

    # Skip if already fetched
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)

    try:
        data = fetch_film_metadata(film_id)
    except Exception as e:
        logger.warning(f"Failed to fetch film {film_id}: {e}")
        return None

    # Download poster
    poster_path_str = data.get("poster_path")
    if poster_path_str:
        poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path_str}"
        poster_save = IMAGES_DIR / f"{film_id}_poster.jpg"
        download_image(poster_url, poster_save)
        data["local_poster_path"] = str(poster_save)

    # Download stills (backdrops)
    stills = data.get("images", {}).get("backdrops", [])[:STILLS_PER_FILM]
    data["local_still_paths"] = []
    for i, still in enumerate(stills):
        still_url = f"{TMDB_STILL_BASE_URL}{still['file_path']}"
        still_save = IMAGES_DIR / f"{film_id}_still_{i}.jpg"
        if download_image(still_url, still_save):
            data["local_still_paths"].append(str(still_save))

    # Extract director from credits
    crew = data.get("credits", {}).get("crew", [])
    directors = [c["name"] for c in crew if c["job"] == "Director"]
    data["directors"] = directors

    # Extract top cast
    cast = data.get("credits", {}).get("cast", [])[:3]
    data["top_cast"] = [c["name"] for c in cast]

    # Save raw JSON
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return data


def run_pipeline() -> None:
    """
    Main entry point. Fetches all films and saves to data/raw/.
    """
    logger.info("Starting TMDB fetch pipeline...")

    film_ids = fetch_popular_film_ids(pages=25)
    logger.info(f"Target: {len(film_ids)} films")

    success = 0
    for i, film_id in enumerate(film_ids):
        result = fetch_and_save_film(film_id)
        if result:
            success += 1
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(film_ids)} films ({success} successful)")
        time.sleep(0.25)  # Respect TMDB rate limit

    logger.info(f"Pipeline complete. {success}/{len(film_ids)} films fetched.")


if __name__ == "__main__":
    run_pipeline()
