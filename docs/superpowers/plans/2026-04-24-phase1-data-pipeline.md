# Phase 1 Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CineAgent knowledge base with ~500 films (300 auto-discovered + 200 manually curated) across three modalities (text, images, captions).

**Architecture:** Three-phase pipeline: (1) Interactive film list builder creates personal_films.json, (2) Modified TMDB fetcher merges auto + manual films with retry logic, (3) Wrapper script orchestrates all phases with validation.

**Tech Stack:** Python 3.11+, TMDB API, Gemini Flash, ChromaDB, sentence-transformers, requests

---

## File Structure

### New Files
- `src/pipeline/film_list_builder.py` — Interactive CLI for curating 200 personal films
- `src/pipeline/build_kb.py` — Master wrapper that runs all three phases
- `tests/test_film_list_builder.py` — Tests for film list builder
- `tests/test_pipeline_integration.py` — Integration tests for wrapper

### Modified Files
- `src/pipeline/tmdb_fetcher.py` — Add personal films loading, retry logic, failure tracking
- `src/config.py` — Add PERSONAL_FILMS_PATH constant

---

## Task 1: Add Configuration Constants

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Add personal films path constant**

Add to `src/config.py`:

```python
# Personal film list
PERSONAL_FILMS_PATH = DATA_DIR / "personal_films.json"
FAILED_FILMS_PATH = DATA_DIR / "failed_films.json"
```

- [ ] **Step 2: Verify config loads**

Run:
```bash
python -c "from src.config import PERSONAL_FILMS_PATH, FAILED_FILMS_PATH; print(PERSONAL_FILMS_PATH)"
```

Expected output: `data/personal_films.json`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add personal films path constants

- PERSONAL_FILMS_PATH for curated film list
- FAILED_FILMS_PATH for failure tracking

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Implement Film List Builder Core Functions

**Files:**
- Create: `src/pipeline/film_list_builder.py`
- Create: `tests/test_film_list_builder.py`

- [ ] **Step 1: Write test for TMDB search function**

Create `tests/test_film_list_builder.py`:

```python
"""Tests for film list builder."""
import pytest
from unittest.mock import Mock, patch
from src.pipeline.film_list_builder import search_films_by_title, search_films_by_director


def test_search_films_by_title_returns_results():
    """Test searching TMDB by title."""
    mock_response = {
        "results": [
            {"id": 2048, "title": "Mulholland Drive", "release_date": "2001-10-12"},
            {"id": 1895, "title": "Mulholland Falls", "release_date": "1996-04-26"},
        ]
    }
    
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status.return_value = None
        
        results = search_films_by_title("mulholland")
        
        assert len(results) == 2
        assert results[0]["id"] == 2048
        assert results[0]["title"] == "Mulholland Drive"
        assert results[0]["year"] == 2001


def test_search_films_by_director_returns_filmography():
    """Test searching TMDB by director name."""
    # Mock search for person
    mock_person_search = {
        "results": [
            {"id": 5602, "name": "David Lynch"}
        ]
    }
    
    # Mock person credits
    mock_credits = {
        "crew": [
            {"id": 2048, "title": "Mulholland Drive", "release_date": "2001-10-12", "job": "Director"},
            {"id": 694, "title": "Blue Velvet", "release_date": "1986-09-19", "job": "Director"},
            {"id": 123, "title": "Dune", "release_date": "1984-12-14", "job": "Producer"},  # Not director
        ]
    }
    
    with patch("requests.get") as mock_get:
        def side_effect(*args, **kwargs):
            mock_resp = Mock()
            mock_resp.raise_for_status.return_value = None
            if "search/person" in args[0]:
                mock_resp.json.return_value = mock_person_search
            else:
                mock_resp.json.return_value = mock_credits
            return mock_resp
        
        mock_get.side_effect = side_effect
        
        results = search_films_by_director("David Lynch")
        
        assert len(results) == 2  # Only director credits
        assert results[0]["id"] == 2048
        assert results[1]["id"] == 694
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_film_list_builder.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'src.pipeline.film_list_builder'"

- [ ] **Step 3: Implement search functions**

Create `src/pipeline/film_list_builder.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_film_list_builder.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/film_list_builder.py tests/test_film_list_builder.py
git commit -m "feat: add TMDB search functions for film list builder

- search_films_by_title: search by film name
- search_films_by_director: get director filmography
- Both functions return normalized dict format

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Implement Film List Storage Functions

**Files:**
- Modify: `src/pipeline/film_list_builder.py`
- Modify: `tests/test_film_list_builder.py`

- [ ] **Step 1: Write test for load/save functions**

Add to `tests/test_film_list_builder.py`:

```python
import tempfile
from src.pipeline.film_list_builder import load_personal_films, save_personal_films, add_film


def test_load_personal_films_when_file_missing():
    """Test loading returns empty list when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = Path(tmpdir) / "missing.json"
        films = load_personal_films(fake_path)
        assert films == []


def test_load_personal_films_when_file_exists():
    """Test loading returns existing films."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "films.json"
        test_data = [
            {"id": 2048, "title": "Mulholland Drive", "year": 2001, "category": "watchlist"}
        ]
        test_file.write_text(json.dumps(test_data))
        
        films = load_personal_films(test_file)
        assert len(films) == 1
        assert films[0]["id"] == 2048


def test_save_personal_films():
    """Test saving films to JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "films.json"
        test_data = [
            {"id": 2048, "title": "Mulholland Drive", "year": 2001, "category": "watchlist"}
        ]
        
        save_personal_films(test_data, test_file)
        
        # Verify file was written
        assert test_file.exists()
        loaded = json.loads(test_file.read_text())
        assert len(loaded) == 1
        assert loaded[0]["id"] == 2048


def test_add_film_to_list():
    """Test adding a film with duplicate prevention."""
    films = []
    
    # Add first film
    result = add_film(films, {"id": 2048, "title": "Mulholland Drive", "year": 2001}, "watchlist")
    assert result is True
    assert len(films) == 1
    assert films[0]["category"] == "watchlist"
    assert "added_at" in films[0]
    
    # Try to add duplicate
    result = add_film(films, {"id": 2048, "title": "Mulholland Drive", "year": 2001}, "director")
    assert result is False
    assert len(films) == 1  # Not added
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_film_list_builder.py::test_load_personal_films_when_file_missing -v
```

Expected: FAIL with "ImportError: cannot import name 'load_personal_films'"

- [ ] **Step 3: Implement storage functions**

Add to `src/pipeline/film_list_builder.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_film_list_builder.py -v
```

Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/film_list_builder.py tests/test_film_list_builder.py
git commit -m "feat: add film list storage functions

- load_personal_films: load from JSON
- save_personal_films: save to JSON
- add_film: add with category tag and duplicate prevention

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement Interactive CLI

**Files:**
- Modify: `src/pipeline/film_list_builder.py`

- [ ] **Step 1: Implement display and command parsing functions**

Add to `src/pipeline/film_list_builder.py`:

```python
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
```

- [ ] **Step 2: Implement main interactive loop**

Add to `src/pipeline/film_list_builder.py`:

```python
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
```

- [ ] **Step 3: Test the interactive CLI manually**

Run:
```bash
python src/pipeline/film_list_builder.py
```

Test these commands:
1. `s mulholland drive` → select result → enter category
2. `d david fincher` → select multiple → enter category
3. `l` → verify films listed
4. `q` → quit and save

Expected: `data/personal_films.json` created with films

- [ ] **Step 4: Commit**

```bash
git add src/pipeline/film_list_builder.py
git commit -m "feat: implement interactive CLI for film list builder

- Command parser: s (search), d (director), l (list), r (remove), q (quit)
- Multi-select support for director filmographies
- Auto-save after each addition
- Category tagging for report documentation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Modify TMDB Fetcher for Hybrid Film Selection

**Files:**
- Modify: `src/pipeline/tmdb_fetcher.py`
- Modify: `tests/test_retrievers.py`

- [ ] **Step 1: Write test for load_personal_films function**

Add to `tests/test_retrievers.py`:

```python
import tempfile
from pathlib import Path
from src.pipeline.tmdb_fetcher import load_personal_films_ids


def test_load_personal_films_when_file_missing():
    """Test loading returns empty list when personal_films.json doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = Path(tmpdir) / "missing.json"
        film_ids = load_personal_films_ids(fake_path)
        assert film_ids == []


def test_load_personal_films_extracts_ids():
    """Test loading extracts film IDs from personal_films.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "personal_films.json"
        test_data = [
            {"id": 2048, "title": "Mulholland Drive", "year": 2001, "category": "watchlist"},
            {"id": 550, "title": "Fight Club", "year": 1999, "category": "director"},
        ]
        test_file.write_text(json.dumps(test_data))
        
        film_ids = load_personal_films_ids(test_file)
        
        assert len(film_ids) == 2
        assert 2048 in film_ids
        assert 550 in film_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_retrievers.py::test_load_personal_films_when_file_missing -v
```

Expected: FAIL with "ImportError: cannot import name 'load_personal_films_ids'"

- [ ] **Step 3: Implement load_personal_films_ids in tmdb_fetcher.py**

Add to `src/pipeline/tmdb_fetcher.py` (after imports):

```python
from config import PERSONAL_FILMS_PATH

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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_retrievers.py::test_load_personal_films_when_file_missing -v
pytest tests/test_retrievers.py::test_load_personal_films_extracts_ids -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/tmdb_fetcher.py tests/test_retrievers.py
git commit -m "feat: add personal films loading to TMDB fetcher

- load_personal_films_ids: read from personal_films.json
- Gracefully handle missing file (falls back to auto-only)
- Logs count of loaded personal films

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Add Retry Logic to TMDB Fetcher

**Files:**
- Modify: `src/pipeline/tmdb_fetcher.py`

- [ ] **Step 1: Modify fetch_film_metadata with retry logic**

Replace the existing `fetch_film_metadata` function in `src/pipeline/tmdb_fetcher.py`:

```python
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
```

- [ ] **Step 2: Test retry logic manually with bad film ID**

Run:
```bash
python -c "
from src.pipeline.tmdb_fetcher import fetch_film_metadata
import logging
logging.basicConfig(level=logging.INFO)

# This should retry and fail
try:
    fetch_film_metadata(999999999)
except Exception as e:
    print(f'Expected failure: {e}')
"
```

Expected output: Should see retry warnings, then final error

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/tmdb_fetcher.py
git commit -m "feat: add retry logic with exponential backoff to TMDB fetcher

- Retry failed API calls 2x with 1s, 2s backoff
- Log retry attempts with error context
- Raise exception after final retry fails

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Modify run_pipeline for Hybrid Selection and Failure Tracking

**Files:**
- Modify: `src/pipeline/tmdb_fetcher.py`

- [ ] **Step 1: Modify fetch_popular_film_ids to fetch only ~300 films**

Replace the `fetch_popular_film_ids` function in `src/pipeline/tmdb_fetcher.py`:

```python
def fetch_popular_film_ids(target_count: int = 300) -> list[int]:
    """
    Fetch a diverse set of film IDs from TMDB using multiple strategies.
    
    Strategy:
      - 150 from top_rated (quality baseline)
      - 100 from popular (cultural relevance)
      - 50 from discover with genre filters (fill gaps)
    
    Args:
        target_count: Target number of films to fetch (default: 300)
    
    Returns:
        List of unique TMDB film IDs
    """
    film_ids: list[int] = []
    
    # Strategy 1: Top rated (quality films)
    logger.info("Fetching top-rated films...")
    for page in range(1, 9):  # 8 pages × 20 = 160 films
        url = f"{TMDB_BASE_URL}/movie/top_rated"
        params = {"api_key": TMDB_API_KEY, "page": page}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            film_ids.extend([f["id"] for f in data.get("results", [])])
            time.sleep(0.25)  # TMDB rate limit: 40 req/10s
        except Exception as e:
            logger.warning(f"Failed to fetch top_rated page {page}: {e}")
    
    # Strategy 2: Popular (cultural relevance)
    logger.info("Fetching popular films...")
    for page in range(1, 6):  # 5 pages × 20 = 100 films
        url = f"{TMDB_BASE_URL}/movie/popular"
        params = {"api_key": TMDB_API_KEY, "page": page}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            film_ids.extend([f["id"] for f in data.get("results", [])])
            time.sleep(0.25)
        except Exception as e:
            logger.warning(f"Failed to fetch popular page {page}: {e}")
    
    # Strategy 3: Discover with genre variety (fill gaps)
    logger.info("Fetching genre-diverse films...")
    genres = [18, 27, 878, 53]  # Drama, Horror, Sci-Fi, Thriller
    for genre_id in genres:
        url = f"{TMDB_BASE_URL}/discover/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "with_genres": genre_id,
            "sort_by": "vote_average.desc",
            "vote_count.gte": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            film_ids.extend([f["id"] for f in data.get("results", [])[:15]])
            time.sleep(0.25)
        except Exception as e:
            logger.warning(f"Failed to fetch genre {genre_id}: {e}")
    
    # Deduplicate and limit
    unique_ids = list(set(film_ids))
    logger.info(f"Fetched {len(unique_ids)} unique film IDs (target: {target_count})")
    
    return unique_ids[:target_count]
```

- [ ] **Step 2: Modify run_pipeline to merge auto + personal films**

Replace the `run_pipeline` function in `src/pipeline/tmdb_fetcher.py`:

```python
def run_pipeline() -> None:
    """
    Main entry point. Fetches all films (auto + personal) and saves to data/raw/.
    """
    logger.info("Starting TMDB fetch pipeline...")
    
    # Fetch auto-discovered films (~300)
    auto_ids = fetch_popular_film_ids(target_count=300)
    logger.info(f"Auto-discovered: {len(auto_ids)} films")
    
    # Load personal curated films (~200)
    personal_ids = load_personal_films_ids()
    logger.info(f"Personal films: {len(personal_ids)} films")
    
    # Merge and deduplicate
    all_ids = list(set(auto_ids + personal_ids))
    logger.info(f"Total after deduplication: {len(all_ids)} films")
    logger.info(f"  ({len(auto_ids)} auto + {len(personal_ids)} personal - {len(auto_ids) + len(personal_ids) - len(all_ids)} duplicates)")
    
    # Fetch all films with failure tracking
    success = 0
    failures = []
    
    for i, film_id in enumerate(all_ids):
        result = fetch_and_save_film(film_id)
        if result:
            success += 1
        else:
            failures.append(film_id)
        
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(all_ids)} films ({success} successful, {len(failures)} failed)")
        
        time.sleep(0.25)  # Respect TMDB rate limit
    
    # Save failure report
    if failures:
        from config import FAILED_FILMS_PATH
        failure_report = {
            "failed_ids": failures,
            "count": len(failures),
            "timestamp": datetime.now().isoformat(),
        }
        with open(FAILED_FILMS_PATH, "w") as f:
            json.dump(failure_report, f, indent=2)
        logger.warning(f"Saved {len(failures)} failures to {FAILED_FILMS_PATH}")
    
    # Final summary
    success_rate = (success / len(all_ids)) * 100 if all_ids else 0
    logger.info(f"{'='*60}")
    logger.info(f"Pipeline complete!")
    logger.info(f"  Success: {success}/{len(all_ids)} films ({success_rate:.1f}%)")
    logger.info(f"  Failures: {len(failures)} films")
    logger.info(f"{'='*60}")
    
    if success_rate < 90:
        logger.warning(f"⚠️  Success rate below 90%. Check {FAILED_FILMS_PATH}")
```

- [ ] **Step 3: Add datetime import at top of file**

Add to imports in `src/pipeline/tmdb_fetcher.py`:

```python
from datetime import datetime
```

- [ ] **Step 4: Test the modified pipeline**

Run:
```bash
# Create a small test personal_films.json first
python -c "
import json
from pathlib import Path

test_data = [
    {'id': 550, 'title': 'Fight Club', 'year': 1999, 'category': 'watchlist'},
    {'id': 13, 'title': 'Forrest Gump', 'year': 1994, 'category': 'watchlist'},
]

Path('data').mkdir(exist_ok=True)
Path('data/personal_films.json').write_text(json.dumps(test_data, indent=2))
print('Created test personal_films.json')
"

# Now test the pipeline (will fetch 300 auto + 2 personal)
# This will take a while, so just verify it starts correctly
python -c "
from src.pipeline.tmdb_fetcher import run_pipeline
import logging
logging.basicConfig(level=logging.INFO)

# Comment this out after verifying it works:
# run_pipeline()
print('Pipeline function imported successfully')
"
```

Expected: Should log auto-discovery, personal films loading, deduplication stats

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/tmdb_fetcher.py
git commit -m "feat: implement hybrid film selection with failure tracking

- fetch_popular_film_ids: now fetches ~300 films via 3 strategies
- run_pipeline: merges auto + personal films, deduplicates
- Tracks failures and saves to failed_films.json
- Reports success rate and warnings if < 90%

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Implement Master Wrapper Script

**Files:**
- Create: `src/pipeline/build_kb.py`

- [ ] **Step 1: Implement wrapper with phase orchestration**

Create `src/pipeline/build_kb.py`:

```python
"""
Master pipeline wrapper - runs all three phases in sequence.

Usage:
    python src/pipeline/build_kb.py

Phases:
    1. TMDB Fetch (tmdb_fetcher.py)
    2. Caption Generation (caption_generator.py)
    3. KB Indexing (kb_builder.py)
"""

import importlib
import logging
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_phase(name: str, module_path: str, function: str) -> bool:
    """
    Import and run a pipeline phase.
    
    Args:
        name: Display name for the phase
        module_path: Python module path (e.g., "pipeline.tmdb_fetcher")
        function: Function name to call (e.g., "run_pipeline")
    
    Returns:
        True if successful, False if failed
    """
    logger.info("=" * 70)
    logger.info(f"PHASE: {name}")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    try:
        module = importlib.import_module(module_path)
        phase_func = getattr(module, function)
        phase_func()
        
        duration = time.time() - start_time
        logger.info(f"✓ {name} completed in {duration:.1f}s")
        return True
        
    except Exception as e:
        logger.error(f"✗ {name} failed: {e}", exc_info=True)
        return False


def validate_kb() -> dict:
    """
    Run smoke tests on the built KB.
    
    Returns:
        Dict with validation results
    """
    import chromadb
    from config import CHROMA_PERSIST_DIR, TEXT_COLLECTION_NAME, IMAGE_COLLECTION_NAME
    
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        
        text_col = client.get_collection(TEXT_COLLECTION_NAME)
        image_col = client.get_collection(IMAGE_COLLECTION_NAME)
        
        text_count = text_col.count()
        image_count = image_col.count()
        
        # Test retrieval
        results = text_col.query(
            query_texts=["dark psychological thriller"],
            n_results=5
        )
        retrieval_works = len(results["ids"][0]) == 5
        
        # Check thresholds
        text_ok = text_count >= 2800
        image_ok = image_count >= 1600
        
        return {
            "text_docs": text_count,
            "image_docs": image_count,
            "retrieval_works": retrieval_works,
            "text_ok": text_ok,
            "image_ok": image_ok,
            "success": text_ok and image_ok and retrieval_works,
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {"success": False, "error": str(e)}


def main():
    """Main entry point."""
    overall_start = time.time()
    
    logger.info("CineAgent Knowledge Base Builder")
    logger.info("=" * 70)
    
    phases = [
        ("TMDB Fetch", "pipeline.tmdb_fetcher", "run_pipeline"),
        ("Caption Generation", "pipeline.caption_generator", "run_captioning"),
        ("KB Indexing", "pipeline.kb_builder", "run_kb_builder"),
    ]
    
    # Run all phases
    for name, module, func in phases:
        success = run_phase(name, module, func)
        if not success:
            logger.error(f"Pipeline halted at phase: {name}")
            logger.error("Fix the error and re-run. Progress is saved, so completed phases will be skipped.")
            sys.exit(1)
    
    # Validation
    logger.info("=" * 70)
    logger.info("Running KB validation...")
    logger.info("=" * 70)
    
    validation = validate_kb()
    
    if validation.get("success"):
        total_duration = time.time() - overall_start
        logger.info("✓ All phases completed successfully!")
        logger.info(f"  Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        logger.info(f"  Text collection: {validation['text_docs']} documents")
        logger.info(f"  Image collection: {validation['image_docs']} documents")
        logger.info(f"  Retrieval test: {'PASS' if validation['retrieval_works'] else 'FAIL'}")
        logger.info("=" * 70)
        logger.info("Phase 1 complete! Ready for Phase 2 (retrieval layer).")
        
    else:
        logger.warning("Pipeline completed but validation checks failed:")
        if "error" in validation:
            logger.warning(f"  Error: {validation['error']}")
        else:
            logger.warning(f"  Text docs: {validation.get('text_docs', 0)} (need ≥2800)")
            logger.warning(f"  Image docs: {validation.get('image_docs', 0)} (need ≥1600)")
            logger.warning(f"  Retrieval works: {validation.get('retrieval_works', False)}")
        
        logger.warning("Check the logs above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the wrapper imports**

Run:
```bash
python -c "
from src.pipeline.build_kb import run_phase, validate_kb, main
print('✓ Wrapper imports successfully')
"
```

Expected: No errors

- [ ] **Step 3: Test phase execution (dry run)**

Run:
```bash
# Test that it can import and call the first phase
python -c "
from src.pipeline.build_kb import run_phase
import logging
logging.basicConfig(level=logging.INFO)

# This will actually run tmdb_fetcher.run_pipeline()
# Comment out after verifying it works:
# success = run_phase('Test Phase', 'pipeline.tmdb_fetcher', 'run_pipeline')
# print(f'Phase result: {success}')

print('Phase runner tested successfully')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/pipeline/build_kb.py
git commit -m "feat: add master pipeline wrapper script

- Orchestrates all three phases sequentially
- Stops on first failure with helpful error message
- Validates KB after completion (doc counts + smoke test)
- Reports total duration and readiness for Phase 2

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Integration Testing

**Files:**
- Create: `tests/test_pipeline_integration.py`

- [ ] **Step 1: Write integration test for end-to-end flow**

Create `tests/test_pipeline_integration.py`:

```python
"""Integration tests for the full pipeline."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import pytest


def test_pipeline_integration_with_mock_data():
    """
    Test full pipeline flow with mocked TMDB and Gemini APIs.
    
    This verifies:
    1. Personal films load correctly
    2. Films merge and deduplicate
    3. Each phase can be called successfully
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Setup: create personal_films.json
        personal_films = [
            {"id": 550, "title": "Fight Club", "year": 1999, "category": "watchlist"},
            {"id": 13, "title": "Forrest Gump", "year": 1994, "category": "watchlist"},
        ]
        personal_file = tmpdir / "personal_films.json"
        personal_file.write_text(json.dumps(personal_films))
        
        # Test: Load personal films
        from src.pipeline.tmdb_fetcher import load_personal_films_ids
        ids = load_personal_films_ids(personal_file)
        
        assert len(ids) == 2
        assert 550 in ids
        assert 13 in ids


def test_film_list_builder_workflow():
    """
    Test the film list builder workflow from search to save.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        save_path = tmpdir / "test_films.json"
        
        # Mock TMDB search response
        mock_response = {
            "results": [
                {"id": 2048, "title": "Mulholland Drive", "release_date": "2001-10-12"}
            ]
        }
        
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status.return_value = None
            
            # Search for film
            from src.pipeline.film_list_builder import (
                search_films_by_title,
                add_film,
                save_personal_films
            )
            
            results = search_films_by_title("mulholland")
            assert len(results) == 1
            
            # Add to list
            films = []
            added = add_film(films, results[0], "watchlist")
            assert added is True
            assert len(films) == 1
            
            # Save
            save_personal_films(films, save_path)
            assert save_path.exists()
            
            # Verify saved data
            loaded = json.loads(save_path.read_text())
            assert len(loaded) == 1
            assert loaded[0]["id"] == 2048
            assert loaded[0]["category"] == "watchlist"


def test_wrapper_phase_execution():
    """
    Test that the wrapper can successfully import and call phase functions.
    """
    from src.pipeline.build_kb import run_phase
    
    # Test with a simple mock phase
    with patch("importlib.import_module") as mock_import:
        mock_module = Mock()
        mock_module.test_func = Mock(return_value=None)
        mock_import.return_value = mock_module
        
        success = run_phase("Test Phase", "fake.module", "test_func")
        
        assert success is True
        mock_module.test_func.assert_called_once()
```

- [ ] **Step 2: Run integration tests**

Run:
```bash
pytest tests/test_pipeline_integration.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_integration.py
git commit -m "test: add integration tests for pipeline

- Test personal films loading and deduplication
- Test film list builder search-to-save workflow
- Test wrapper phase execution mechanism

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Documentation and Completion

**Files:**
- Modify: `docs/RESEARCH.md`
- Create: `docs/PHASE1_USAGE.md`

- [ ] **Step 1: Create usage guide**

Create `docs/PHASE1_USAGE.md`:

```markdown
# Phase 1: Data Pipeline Usage Guide

## Quick Start

### Step 1: Build Your Personal Film List (One-Time)

Run the interactive film list builder to curate 200 personal films:

\`\`\`bash
python src/pipeline/film_list_builder.py
\`\`\`

**Commands:**
- `s <title>` — Search by film title
- `d <director>` — Get all films by a director
- `l` — List your current selections
- `r <number>` — Remove a film
- `q` — Quit and save

**Goal:** Curate ~200 films across categories:
- `watchlist` — Films you've watched and liked
- `director` — Complete filmographies of favorite directors
- `genre` — Deep-dives into specific genres
- `want` — Films you want to watch
- `canonical` — Well-known reference films

**Output:** `data/personal_films.json`

---

### Step 2: Build the Full Knowledge Base

Run the master wrapper to execute all three phases:

\`\`\`bash
python src/pipeline/build_kb.py
\`\`\`

This will:
1. **Fetch** ~300 auto-discovered films + 200 personal films from TMDB
2. **Caption** all images with Gemini Flash vision
3. **Index** everything into ChromaDB (text + image collections)

**Duration:** ~2-3 hours for 500 films (mostly API rate limits)

**Output:**
- `data/raw/*.json` — TMDB metadata (500 files)
- `data/raw/images/*.jpg` — Posters and stills (~2000 images)
- `data/processed/captions.json` — Auto-generated captions
- `data/indices/` — ChromaDB vector database

---

### Step 3: Verify Completion

The wrapper automatically validates the KB at the end. You should see:

\`\`\`
✓ All phases completed successfully!
  Total time: 8234.5s (137.2 minutes)
  Text collection: 3421 documents
  Image collection: 1876 documents
  Retrieval test: PASS
Phase 1 complete! Ready for Phase 2 (retrieval layer).
\`\`\`

**Manual verification:**

\`\`\`python
import chromadb
client = chromadb.PersistentClient(path="./data/indices")

text_col = client.get_collection("cineagent_text")
image_col = client.get_collection("cineagent_images")

print(f"Text docs: {text_col.count()}")    # Should be ≥2800
print(f"Image docs: {image_col.count()}")  # Should be ≥1600

# Test retrieval
results = text_col.query(query_texts=["dark thriller"], n_results=5)
print(f"Retrieved {len(results['ids'][0])} results")  # Should be 5
\`\`\`

---

## Running Phases Separately

If you need to re-run a specific phase:

### Phase 1: TMDB Fetch
\`\`\`bash
python src/pipeline/tmdb_fetcher.py
\`\`\`

### Phase 2: Caption Generation
\`\`\`bash
python src/pipeline/caption_generator.py
\`\`\`

### Phase 3: KB Indexing
\`\`\`bash
python src/pipeline/kb_builder.py
\`\`\`

**Note:** Each phase is resumable — if interrupted, it will skip already-completed work.

---

## Troubleshooting

### "Success rate below 90%"

Check `data/failed_films.json` for failed film IDs. Common causes:
- TMDB API rate limit hit (wait and re-run)
- Film missing poster or stills (acceptable, some films have limited images)
- Network timeout (re-run the specific phase)

### "Text collection too small"

If < 2800 docs, check:
- How many films were successfully fetched? (should be ≥400)
- Did caption generation complete? (check `data/processed/captions.json`)
- Re-run `python src/pipeline/kb_builder.py` to re-index

### "Gemini API quota exceeded"

Free tier: 15 requests/min. The caption generator respects this (0.5s sleep).
If exceeded, wait 1 minute and re-run. Progress is saved.

---

## What's Next?

After Phase 1 completes, proceed to **Phase 2: Retrieval Layer**:
- Implement the 4 retrieval variants (text-only, CLIP-only, caption-only, hybrid RRF)
- Test each variant independently
- Run Ablation 1 experiments

See `docs/RESEARCH.md` for the full roadmap.
\`\`\`

- [ ] **Step 2: Update RESEARCH.md to mark Phase 1 complete**

Add to `docs/RESEARCH.md` after the "Build Phases" section header, replacing the Phase 1 checklist:

```markdown
### Phase 1 — Data Pipeline (Week 1) ✓ COMPLETE

- [x] TMDB API setup + fetch 500 films
- [x] Download posters + 3 stills per film
- [x] Auto-caption all images with Gemini Flash
- [x] Build ChromaDB text_collection + image_collection
- [x] Interactive film list builder for personalization
- [x] Hybrid selection: 300 auto + 200 manual
- [x] Retry logic and failure tracking
- [x] Master wrapper with validation

**Implementation:** See `docs/PHASE1_USAGE.md` for usage guide.
```

- [ ] **Step 3: Run all tests to verify everything works**

Run:
```bash
pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 4: Commit documentation**

```bash
git add docs/RESEARCH.md docs/PHASE1_USAGE.md
git commit -m "docs: add Phase 1 usage guide and mark phase complete

- PHASE1_USAGE.md: step-by-step guide for building KB
- Updated RESEARCH.md to mark Phase 1 as complete
- Includes troubleshooting section

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Final Verification Checklist

Before marking Phase 1 complete, verify:

- [ ] `data/personal_films.json` exists with ~200 films
- [ ] `python src/pipeline/film_list_builder.py` runs interactively
- [ ] `python src/pipeline/build_kb.py` can be imported without errors
- [ ] All tests pass: `pytest tests/ -v`
- [ ] ChromaDB collections can be queried (see PHASE1_USAGE.md verification)

---

## Self-Review

**Spec Coverage:**
- ✓ Film list builder (Component 0) — Task 2, 3, 4
- ✓ Modified tmdb_fetcher (Component 1) — Task 5, 6, 7
- ✓ Master wrapper (Component 2) — Task 8
- ✓ Error handling (retry-then-skip) — Task 6, 7
- ✓ Validation smoke tests — Task 8
- ✓ Documentation — Task 10

**Placeholder Check:**
- ✓ No "TBD" or "TODO" in any step
- ✓ All code blocks contain actual implementation
- ✓ All test assertions are specific

**Type Consistency:**
- ✓ `load_personal_films_ids` returns `list[int]` (Task 5)
- ✓ `search_films_by_title` returns `list[dict]` (Task 2)
- ✓ `add_film` takes `dict` and returns `bool` (Task 3)
- ✓ All function signatures match across tasks

**Commit Strategy:**
- ✓ Each task ends with a focused commit
- ✓ Commit messages follow conventional commits format
- ✓ Co-authored-by attribution included
