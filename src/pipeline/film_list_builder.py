"""
Film List Builder — Interactive CLI for curating personal film list.

Usage:
    python src/pipeline/film_list_builder.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TMDB_API_KEY, TMDB_BASE_URL, PERSONAL_FILMS_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def search_films_by_title(query: str) -> list[dict]:
    """
    Search TMDB for films by title.

    Args:
        query: Film title to search for

    Returns:
        List of dicts with keys: id, title, year, director (if available)
    """
    url = f"{TMDB_BASE_URL}/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for film in data.get("results", [])[:10]:  # Limit to top 10
            year = film.get("release_date", "")[:4] if film.get("release_date") else ""
            results.append({
                "id": film["id"],
                "title": film.get("title", "Unknown"),
                "year": int(year) if year.isdigit() else 0,
            })

        return results
    except Exception as e:
        logger.error(f"Search failed for '{query}': {e}")
        return []


def search_films_by_director(director_name: str) -> list[dict]:
    """
    Search TMDB for all films directed by a person.

    Args:
        director_name: Director's name

    Returns:
        List of dicts with keys: id, title, year
    """
    # Step 1: Search for person
    url = f"{TMDB_BASE_URL}/search/person"
    params = {"api_key": TMDB_API_KEY, "query": director_name}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        people = resp.json().get("results", [])

        if not people:
            logger.warning(f"No person found for '{director_name}'")
            return []

        person_id = people[0]["id"]

        # Step 2: Get their credits
        url = f"{TMDB_BASE_URL}/person/{person_id}/movie_credits"
        params = {"api_key": TMDB_API_KEY}

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        credits = resp.json()

        # Filter to director credits only
        results = []
        for film in credits.get("crew", []):
            if film.get("job") == "Director":
                year = film.get("release_date", "")[:4] if film.get("release_date") else ""
                results.append({
                    "id": film["id"],
                    "title": film.get("title", "Unknown"),
                    "year": int(year) if year.isdigit() else 0,
                })

        # Sort by year descending
        results.sort(key=lambda x: x["year"], reverse=True)
        return results

    except Exception as e:
        logger.error(f"Director search failed for '{director_name}': {e}")
        return []


def load_personal_films(file_path: Path = PERSONAL_FILMS_PATH) -> list[dict]:
    """
    Load personal film list from JSON file.

    Args:
        file_path: Path to personal_films.json

    Returns:
        List of film dicts, or empty list if file doesn't exist
    """
    if not file_path.exists():
        return []

    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []


def save_personal_films(films: list[dict], file_path: Path = PERSONAL_FILMS_PATH) -> None:
    """
    Save personal film list to JSON file.

    Args:
        films: List of film dicts
        file_path: Path to save to
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(films, f, indent=2)

    logger.info(f"Saved {len(films)} films to {file_path}")


def add_film(films: list[dict], film_info: dict, category: str) -> bool:
    """
    Add a film to the list with category tag and timestamp.

    Args:
        films: Existing film list (modified in-place)
        film_info: Dict with keys: id, title, year
        category: Category tag (watchlist/director/genre/want/canonical)

    Returns:
        True if added, False if duplicate
    """
    # Check for duplicate
    if any(f["id"] == film_info["id"] for f in films):
        logger.warning(f"Film {film_info['title']} already in list")
        return False

    # Add film with metadata
    film_entry = {
        "id": film_info["id"],
        "title": film_info["title"],
        "year": film_info["year"],
        "category": category,
        "added_at": datetime.now().isoformat(),
    }

    films.append(film_entry)
    logger.info(f"Added: {film_info['title']} ({film_info['year']}) [{category}]")
    return True
