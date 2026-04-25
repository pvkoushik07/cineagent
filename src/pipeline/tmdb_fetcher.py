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
from datetime import datetime
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
    AUTO_FILMS_PATH,
    FAILED_FILMS_PATH,
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


def fetch_film_metadata(film_id: int, max_retries: int = 2) -> dict:
    """
    Fetch full metadata for a single film from TMDB with retry logic.

    Args:
        film_id: TMDB integer film ID
        max_retries: Maximum number of retry attempts

    Returns:
        Dict with title, overview, genres, director, cast, release_date, etc.

    Raises:
        requests.RequestException: If all retries fail
    """
    for attempt in range(max_retries + 1):
        try:
            url = f"{TMDB_BASE_URL}/movie/{film_id}"
            params = {
                "api_key": TMDB_API_KEY,
                "append_to_response": "credits,images",
                "include_image_language": "en,null",
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} for film {film_id} "
                    f"after {wait_time}s (error: {e})"
                )
                time.sleep(wait_time)
                continue

            # Final attempt failed
            logger.error(f"Failed to fetch film {film_id} after {max_retries} retries: {e}")
            raise


def fetch_popular_film_ids(target_count: int = 300) -> list[int]:
    """
    Fetch a diverse list of film IDs from TMDB using 3 strategies.

    Uses multiple sort strategies to get a diverse set for personalisation:
    - Strategy 1: top_rated (150 films, quality baseline)
    - Strategy 2: popular (100 films, cultural relevance)
    - Strategy 3: discover with genre filters (50 films, fill gaps)

    Target: ~300 film IDs across different genres, years, languages.

    Caches results to auto_films.json to avoid re-fetching on subsequent runs.

    Args:
        target_count: Target number of unique films (default 300)

    Returns:
        List of TMDB film IDs
    """
    # Check cache first
    if AUTO_FILMS_PATH.exists():
        logger.info(f"Loading cached auto-discovered film IDs from {AUTO_FILMS_PATH}")
        with open(AUTO_FILMS_PATH) as f:
            cached_ids = json.load(f)
        logger.info(f"Loaded {len(cached_ids)} auto-discovered film IDs from cache")
        return cached_ids[:target_count]

    film_ids: list[int] = []

    # Strategy 1: top_rated (quality baseline) — aim for 150 films
    logger.info("Strategy 1: Fetching top_rated films...")
    pages_needed = (150 // 20) + 1  # 20 films per page
    for page in range(1, pages_needed + 1):
        url = f"{TMDB_BASE_URL}/movie/top_rated"
        params = {"api_key": TMDB_API_KEY, "page": page}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        film_ids.extend([f["id"] for f in data.get("results", [])])
        time.sleep(0.25)  # TMDB rate limit: 40 req/10s

    logger.info(f"Strategy 1 added {len(film_ids)} films (target: 150)")

    # Strategy 2: popular (cultural relevance) — aim for 100 films
    logger.info("Strategy 2: Fetching popular films...")
    pages_needed = (100 // 20) + 1
    for page in range(1, pages_needed + 1):
        url = f"{TMDB_BASE_URL}/movie/popular"
        params = {"api_key": TMDB_API_KEY, "page": page}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        film_ids.extend([f["id"] for f in data.get("results", [])])
        time.sleep(0.25)

    logger.info(f"Strategy 2 total: {len(film_ids)} films (target: 250 cumulative)")

    # Strategy 3: discover with genre filters (fill gaps) — aim for 50 films
    logger.info("Strategy 3: Fetching discover films with genre filters...")
    genres = [28, 12, 16, 35, 80, 99, 18, 10751, 14, 36, 27, 10402, 9648, 10749, 878]  # Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Sci-Fi
    pages_per_genre = 1  # Fetch 1 page per genre
    for genre_id in genres[:5]:  # Sample 5 genres to get ~50 films
        url = f"{TMDB_BASE_URL}/discover/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "with_genres": genre_id,
            "page": 1,
            "sort_by": "vote_average.desc",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        film_ids.extend([f["id"] for f in data.get("results", [])])
        time.sleep(0.25)

    logger.info(f"Strategy 3 total: {len(film_ids)} films (before dedup)")

    # Deduplicate and limit to target
    unique_ids = list(set(film_ids))
    logger.info(f"Fetched {len(unique_ids)} unique film IDs from TMDB")

    # Cache for future runs
    result = unique_ids[:target_count]
    with open(AUTO_FILMS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Cached {len(result)} auto-discovered film IDs to {AUTO_FILMS_PATH}")

    return result


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
    # Request 10 backdrops, filter for landscape (real stills vs promotional images)
    all_backdrops = data.get("images", {}).get("backdrops", [])[:10]

    # Filter: keep only landscape images (width > height × 1.3)
    landscape_stills = [
        bd for bd in all_backdrops
        if bd.get("width", 0) > bd.get("height", 1) * 1.3
    ]

    # Take first 5 landscape stills
    stills_to_download = landscape_stills[:5]

    data["local_still_paths"] = []
    for i, still in enumerate(stills_to_download):
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
    Main entry point. Fetches all films (auto-discovered + personal) and saves to data/raw/.

    Merges auto-discovered films (~300) with personal films (~200) for hybrid selection.
    Tracks failures and saves to failed_films.json with timestamp.
    """
    logger.info("Starting TMDB fetch pipeline (hybrid selection)...")

    # Fetch auto-discovered films
    auto_ids = fetch_popular_film_ids(target_count=300)
    logger.info(f"Auto-discovered: {len(auto_ids)} films")

    # Load personal films
    personal_ids = load_personal_films_ids()
    logger.info(f"Personal films: {len(personal_ids)} films")

    # Merge and deduplicate
    all_ids = list(set(auto_ids + personal_ids))
    logger.info(f"Total films (merged + deduplicated): {len(all_ids)} films")

    # Track failures
    failures: list[dict] = []
    success = 0

    for i, film_id in enumerate(all_ids):
        result = fetch_and_save_film(film_id)
        if result:
            success += 1
        else:
            failures.append({
                "film_id": film_id,
                "timestamp": datetime.now().isoformat(),
            })

        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(all_ids)} films ({success} successful)")
        time.sleep(0.25)  # Respect TMDB rate limit

    # Save failures to failed_films.json
    if failures:
        with open(FAILED_FILMS_PATH, "w") as f:
            json.dump(failures, f, indent=2)
        logger.warning(f"Saved {len(failures)} failed films to {FAILED_FILMS_PATH}")

    # Calculate and report success rate
    total = len(all_ids)
    success_rate = (success / total * 100) if total > 0 else 0
    logger.info(f"Pipeline complete. {success}/{total} films fetched ({success_rate:.1f}% success)")

    # Warn if success rate < 90%
    if success_rate < 90:
        logger.warning(f"Success rate {success_rate:.1f}% is below 90% threshold!")


if __name__ == "__main__":
    run_pipeline()
