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


def display_films(films: list[dict], limit: int = 20) -> None:
    """Display current film list."""
    if not films:
        print("No films in list yet.")
        return

    print(f"\n{'='*60}")
    print(f"Current list: {len(films)} films")
    print(f"{'='*60}")

    # Group by category
    by_category = {}
    for film in films:
        cat = film.get("category", "unknown")
        by_category.setdefault(cat, []).append(film)

    for category, cat_films in sorted(by_category.items()):
        print(f"\n{category.upper()} ({len(cat_films)} films):")
        for i, film in enumerate(cat_films[:limit], 1):
            print(f"  {i}. {film['title']} ({film['year']})")

        if len(cat_films) > limit:
            print(f"  ... and {len(cat_films) - limit} more")


def display_search_results(results: list[dict]) -> None:
    """Display search results with numbered selection."""
    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} results:")
    for i, film in enumerate(results, 1):
        print(f"  {i}. {film['title']} ({film['year']})")


def get_user_selection(results: list[dict], multi: bool = False) -> list[dict]:
    """
    Prompt user to select from search results.

    Args:
        results: List of search results
        multi: Allow multiple selections (True) or single (False)

    Returns:
        List of selected film dicts
    """
    if not results:
        return []

    if multi:
        prompt = f"Select [1-{len(results)}, comma-separated, or 'all']: "
    else:
        prompt = f"Select [1-{len(results)}]: "

    try:
        choice = input(prompt).strip().lower()

        if multi and choice == "all":
            return results

        # Parse selection
        if multi:
            indices = [int(c.strip()) for c in choice.split(",")]
        else:
            indices = [int(choice)]

        # Validate range
        selected = []
        for idx in indices:
            if 1 <= idx <= len(results):
                selected.append(results[idx - 1])
            else:
                print(f"Invalid selection: {idx}")

        return selected

    except (ValueError, KeyboardInterrupt):
        print("\nCancelled.")
        return []


def get_category() -> str:
    """Prompt user for category tag."""
    categories = ["watchlist", "director", "genre", "want", "canonical"]

    while True:
        choice = input(f"Category? [{'/'.join(categories)}]: ").strip().lower()
        if choice in categories:
            return choice
        print(f"Invalid category. Choose from: {', '.join(categories)}")


def print_help() -> None:
    """Print command help."""
    print("""
Commands:
  s <title>   - Search by film title
  d <name>    - Search by director name
  l           - List current selections
  r <number>  - Remove a film by list number
  q           - Quit and save
  h           - Show this help
""")


def run_interactive() -> None:
    """Main interactive CLI loop."""
    print("="*60)
    print("CineAgent Film List Builder")
    print("="*60)
    print(f"Target: 200 personally curated films")
    print_help()

    films = load_personal_films()
    print(f"\nLoaded {len(films)} existing films")

    try:
        while True:
            print(f"\nCurrent: {len(films)}/200 films")
            command = input("> ").strip()

            if not command:
                continue

            parts = command.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "q":
                save_personal_films(films)
                print(f"✓ Saved {len(films)} films to {PERSONAL_FILMS_PATH}")
                break

            elif cmd == "h":
                print_help()

            elif cmd == "l":
                display_films(films)

            elif cmd == "s":
                if not arg:
                    print("Usage: s <title>")
                    continue

                results = search_films_by_title(arg)
                display_search_results(results)

                if results:
                    selected = get_user_selection(results, multi=False)
                    if selected:
                        category = get_category()
                        if add_film(films, selected[0], category):
                            save_personal_films(films)  # Auto-save

            elif cmd == "d":
                if not arg:
                    print("Usage: d <director name>")
                    continue

                results = search_films_by_director(arg)
                display_search_results(results)

                if results:
                    selected = get_user_selection(results, multi=True)
                    if selected:
                        category = get_category()
                        added = 0
                        for film in selected:
                            if add_film(films, film, category):
                                added += 1
                        if added > 0:
                            save_personal_films(films)
                            print(f"✓ Added {added} films")

            elif cmd == "r":
                if not arg.isdigit():
                    print("Usage: r <number>")
                    continue

                idx = int(arg) - 1
                if 0 <= idx < len(films):
                    removed = films.pop(idx)
                    save_personal_films(films)
                    print(f"✓ Removed: {removed['title']}")
                else:
                    print(f"Invalid number. List has {len(films)} films.")

            else:
                print(f"Unknown command: {cmd}. Type 'h' for help.")

    except KeyboardInterrupt:
        print("\n\nSaving before exit...")
        save_personal_films(films)
        print(f"✓ Saved {len(films)} films")


if __name__ == "__main__":
    run_interactive()
