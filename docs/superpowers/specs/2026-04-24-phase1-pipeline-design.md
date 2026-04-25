# Phase 1 Data Pipeline Design
**Date:** 2026-04-24  
**Phase:** 1 — Data Pipeline (TMDB Fetch + Captions + KB Build)  
**Status:** Approved

---

## Overview

Build the CineAgent knowledge base with ~500 films across three modalities (text, poster images, scene stills) using a hybrid film selection strategy: 300 auto-discovered films from TMDB + 200 manually curated films from personal watchlist.

## Design Approach

**Selected: Approach 1 — Minimal Modifications**

Build on existing pipeline scaffolding with targeted additions:
- Add interactive film list builder (`film_list_builder.py`)
- Modify TMDB fetcher to merge auto + manual film IDs
- Add master wrapper script (`build_kb.py`) for one-command execution
- Add retry-then-skip error handling with failure tracking

**Why this approach:**
- Fastest to implement — existing scripts already work
- Simple, debuggable architecture
- Meets all rubric requirements (personalisation, reproducibility, modularity)
- Saves engineering complexity for Phase 2-4 where it matters

---

## Architecture

### Pipeline Flow

```
Phase 0 (Manual, run once):
  film_list_builder.py (interactive CLI)
    → Output: data/personal_films.json (200 curated films)

Phase 1 (TMDB Fetch):
  tmdb_fetcher.py (modified)
    → Input: personal_films.json
    → Fetch: 300 auto + 200 manual = 500 films
    → Output: data/raw/*.json, data/raw/images/*.jpg
    → Failures: data/failed_films.json

Phase 2 (Caption Generation):
  caption_generator.py (unchanged)
    → Input: data/raw/images/*.jpg
    → Output: data/processed/captions.json (~2000 captions)

Phase 3 (KB Indexing):
  kb_builder.py (unchanged)
    → Input: data/raw/*.json + captions.json
    → Output: data/indices/ (ChromaDB collections)

Optional:
  build_kb.py (wrapper)
    → Runs all three phases in sequence
```

### Communication Between Phases

Phases communicate via **JSON files on disk**:
- Simple, debuggable, no complex state management
- Each phase is resumable (checks existing output files)
- Easy to inspect intermediate results

---

## Component Specifications

### Component 0: `film_list_builder.py` (New)

**Purpose:** Interactive CLI helper for building personal 200-film list

**Features:**
- Search by title: `s <title>` → returns TMDB search results → pick correct one
- Search by director: `d <name>` → returns all films by director → select multiple
- Category tagging: Tag each film (watchlist/favorite_director/want_to_watch/canonical)
- Progress saving: Auto-saves to `data/personal_films.json` after each addition
- Resume support: Shows current list, allows continuation

**UI Flow:**
```
CineAgent Film List Builder
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current list: 47/200 films

Commands:
  s <title>   - Search by title
  d <name>    - Search by director
  l           - List current selections
  r           - Remove a film
  q           - Quit and save

> s mulholland drive
Found 3 results:
  1. Mulholland Drive (2001) - David Lynch
  2. Mulholland Dr. (1999) - David Lynch (TV)
  3. Mulholland Falls (1996)
Select [1-3]: 1

Category? [watchlist/director/genre/want/canonical]: watchlist
✓ Added: Mulholland Drive (2001)
```

**Output Format** (`data/personal_films.json`):
```json
[
  {
    "id": 2048,
    "title": "Mulholland Drive",
    "year": 2001,
    "category": "watchlist",
    "added_at": "2026-04-24T10:23:15"
  }
]
```

**Implementation Notes:**
- Use `requests` to call TMDB search endpoints
- Simple input validation (prevent duplicates)
- Write to JSON after every successful addition (crash-safe)

---

### Component 1: `tmdb_fetcher.py` (Modified)

**Changes to existing code:**

#### 1. New Function: `load_personal_films()`
```python
def load_personal_films() -> list[int]:
    """Load manually curated film IDs from personal_films.json."""
    personal_file = Path("data/personal_films.json")
    if not personal_file.exists():
        logger.warning("personal_films.json not found, using auto-discovery only")
        return []
    
    with open(personal_file) as f:
        data = json.load(f)
    
    film_ids = [film["id"] for film in data]
    logger.info(f"Loaded {len(film_ids)} personal films")
    return film_ids
```

#### 2. Modified: `fetch_popular_film_ids()`
- **Before:** Fetched 500 films from `top_rated` endpoint
- **After:** Fetch only ~300 films using multiple strategies for diversity:
  - 150 from `top_rated` (quality baseline)
  - 100 from `popular` (cultural relevance)
  - 50 from `discover` with genre filters (fill gaps)

#### 3. Modified: `run_pipeline()`
```python
def run_pipeline() -> None:
    logger.info("Starting TMDB fetch pipeline...")
    
    # Merge auto + personal film IDs
    auto_ids = fetch_popular_film_ids(pages=15)  # ~300 films
    personal_ids = load_personal_films()          # ~200 films
    all_ids = list(set(auto_ids + personal_ids))  # Deduplicate
    
    logger.info(f"Target: {len(all_ids)} films ({len(auto_ids)} auto + {len(personal_ids)} personal)")
    
    success = 0
    failures = []
    
    for i, film_id in enumerate(all_ids):
        result = fetch_and_save_film(film_id)
        if result:
            success += 1
        else:
            failures.append(film_id)
        
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(all_ids)} ({success} successful)")
        time.sleep(0.25)
    
    # Save failure report
    if failures:
        with open("data/failed_films.json", "w") as f:
            json.dump({"failed_ids": failures, "count": len(failures)}, f, indent=2)
    
    logger.info(f"Pipeline complete. {success}/{len(all_ids)} films fetched.")
```

#### 4. Add Retry Logic to `fetch_film_metadata()`
```python
def fetch_film_metadata(film_id: int, max_retries: int = 2) -> dict:
    """Fetch with exponential backoff retry."""
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
                logger.warning(f"Retry {attempt + 1}/{max_retries} for film {film_id} after {wait_time}s")
                time.sleep(wait_time)
                continue
            logger.error(f"Failed to fetch film {film_id} after {max_retries} retries: {e}")
            raise
```

---

### Component 2: `build_kb.py` (New Wrapper)

**Purpose:** Master orchestrator that runs all three phases sequentially

**Implementation:**
```python
"""
Master pipeline wrapper - runs all three phases in sequence.

Usage:
    python src/pipeline/build_kb.py
"""

import importlib
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_phase(name: str, module_path: str, function: str) -> bool:
    """
    Import and run a pipeline phase.
    Returns True if successful, False if failed.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Starting phase: {name}")
    logger.info(f"{'='*60}")
    
    start = time.time()
    
    try:
        module = importlib.import_module(module_path)
        getattr(module, function)()
        duration = time.time() - start
        logger.info(f"✓ {name} complete ({duration:.1f}s)")
        return True
    except Exception as e:
        logger.error(f"✗ {name} failed: {e}", exc_info=True)
        return False

def validate_kb() -> dict:
    """Run smoke tests on the built KB."""
    import chromadb
    
    client = chromadb.PersistentClient(path="./data/indices")
    text_col = client.get_collection("cineagent_text")
    image_col = client.get_collection("cineagent_images")
    
    text_count = text_col.count()
    image_count = image_col.count()
    
    # Test retrieval
    results = text_col.query(query_texts=["dark psychological thriller"], n_results=5)
    retrieval_works = len(results["ids"][0]) == 5
    
    return {
        "text_docs": text_count,
        "image_docs": image_count,
        "retrieval_works": retrieval_works,
        "success": text_count >= 2800 and image_count >= 1600 and retrieval_works
    }

def main():
    start_time = time.time()
    
    phases = [
        ("TMDB Fetch", "src.pipeline.tmdb_fetcher", "run_pipeline"),
        ("Caption Generation", "src.pipeline.caption_generator", "run_captioning"),
        ("KB Indexing", "src.pipeline.kb_builder", "run_kb_builder"),
    ]
    
    for name, module, func in phases:
        if not run_phase(name, module, func):
            logger.error(f"Pipeline halted at {name}")
            sys.exit(1)
    
    # Validation
    logger.info("Running KB validation...")
    validation = validate_kb()
    
    if validation["success"]:
        logger.info(f"✓ Pipeline complete! ({time.time() - start_time:.1f}s total)")
        logger.info(f"  Text collection: {validation['text_docs']} documents")
        logger.info(f"  Image collection: {validation['image_docs']} documents")
    else:
        logger.warning("Pipeline completed but validation checks failed:")
        logger.warning(f"  Text docs: {validation['text_docs']} (need ≥2800)")
        logger.warning(f"  Image docs: {validation['image_docs']} (need ≥1600)")
        logger.warning(f"  Retrieval works: {validation['retrieval_works']}")

if __name__ == "__main__":
    main()
```

---

## Error Handling Strategy

### Retry-Then-Skip Approach

**At each phase:**
- **TMDB Fetch**: 
  - Retry API calls 2× with exponential backoff (1s, 2s)
  - Log failures to `data/failed_films.json`
  - Continue with remaining films
  
- **Caption Generation**: 
  - Try/except per image (already implemented)
  - Log warnings for failures
  - Save progress every 50 images
  
- **KB Builder**: 
  - Skip images that can't be opened
  - Log warnings
  - Continue indexing

### Failure Tracking

**Output:** `data/failed_films.json`
```json
{
  "failed_ids": [12345, 67890],
  "count": 2,
  "timestamp": "2026-04-24T14:32:10"
}
```

**Validation at end:**
- Report success rate: "Built KB with 487/500 films (97.4%)"
- If < 90% success (< 450 films): Print warning to check `failed_films.json`
- Smoke test: Query both collections to ensure retrieval works

---

## File Structure After Completion

```
data/
├── personal_films.json          # 200 curated films (input)
├── failed_films.json            # Failures if any (output)
├── raw/
│   ├── *.json                   # ~500 TMDB metadata files
│   └── images/
│       ├── *_poster.jpg         # ~500 posters
│       └── *_still_*.jpg        # ~1500 stills (3 per film)
├── processed/
│   ├── captions.json            # ~2000 image captions
│   └── films_processed.json     # Debug sample
└── indices/                     # ChromaDB persistent storage
    └── chroma.sqlite3
```

---

## Completion Criteria

Phase 1 is complete when:

✅ `data/indices/` contains ChromaDB collections  
✅ Text collection has ≥ 2800 documents (~400 films × 7 docs/film)  
✅ Image collection has ≥ 1600 documents (~400 films × 4 images/film)  
✅ Success rate ≥ 90% (≥ 450/500 films)  
✅ Smoke test passes: Can query both collections and get results  
✅ `personal_films.json` documents curation for the report  

**Validation command:**
```python
import chromadb
client = chromadb.PersistentClient(path="./data/indices")
text_col = client.get_collection("cineagent_text")
image_col = client.get_collection("cineagent_images")

assert text_col.count() >= 2800, "Text collection too small"
assert image_col.count() >= 1600, "Image collection too small"

results = text_col.query(query_texts=["dark psychological thriller"], n_results=5)
assert len(results["ids"][0]) == 5, "Retrieval failed"

print("✓ Phase 1 complete!")
```

---

## Next Steps

After Phase 1 completion:
1. Update `docs/RESEARCH.md` to mark Phase 1 complete
2. Commit the knowledge base metadata (not the images/indices themselves)
3. Proceed to Phase 2: Retrieval Layer implementation
