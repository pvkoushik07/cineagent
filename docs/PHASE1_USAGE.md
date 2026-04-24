# Phase 1 Usage Guide — Building the Knowledge Base

This guide walks you through building the CineAgent knowledge base from scratch.
Phase 1 creates the complete multimodal KB: plot text, posters, scene stills, and
auto-generated captions — all indexed in ChromaDB.

---

## Quick Start (3 Steps)

### 1. Set up environment variables

```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your API keys:
# TMDB_API_KEY=your_tmdb_key_here
# GEMINI_API_KEY=your_gemini_key_here
```

Get your keys:
- TMDB: https://developer.themoviedb.org (free)
- Gemini: https://aistudio.google.com (free tier available)

### 2. Build your personal film list (optional but recommended)

```bash
python src/pipeline/film_list_builder.py
```

This launches an interactive CLI that lets you:
- Search for films by title
- Search for films by director
- Add films to your personal list
- View your current list
- Save and load your list

The saved list will be merged with popular films to reach the target of 500 films.

**Why this matters:** The research question asks "does a multimodal agent outperform
a plain LLM on personalised recommendation accuracy?" You cannot measure
personalisation without films you have actually watched or want to watch. This
is not just a technical requirement — it is a research validity requirement.

### 3. Run the full pipeline

```bash
python src/pipeline/build_kb.py
```

This runs all three phases sequentially:
1. TMDB Fetcher: downloads metadata, posters, and 3 scene stills per film
2. Caption Generator: auto-captions all images using Gemini Flash vision
3. KB Builder: embeds and indexes everything into ChromaDB

**Expected time:** 20-30 minutes for 500 films (TMDB has rate limits)

**Output:** Two ChromaDB collections in `data/indices/`:
- `cineagent_text`: plot summaries, reviews, captions (MiniLM embeddings)
- `cineagent_images`: posters and stills (CLIP embeddings)

---

## Running Phases Separately (Advanced)

If you need to re-run individual phases (e.g., regenerate captions with different
prompts, or rebuild the index without re-fetching TMDB data), you can run each
phase independently.

### Phase 1: TMDB Fetcher

```bash
python src/pipeline/tmdb_fetcher.py
```

**What it does:**
- Reads `data/personal_films.json` (if it exists)
- Fetches top popular films from TMDB to fill remaining slots up to 500
- For each film:
  - Downloads metadata (title, year, genres, director, cast, plot, runtime, rating)
  - Downloads official poster (JPEG, 500px width)
  - Downloads 3 scene stills (JPEG, 780px width)
- Saves raw JSON to `data/raw/films/`
- Saves images to `data/raw/images/`
- Tracks failures in `data/failed_films.json`

**Exit codes:**
- 0: Success
- 1: TMDB API key missing or invalid
- 2: Network error or rate limit exceeded

**Note:** TMDB allows 40 requests per 10 seconds. The fetcher respects this with
automatic rate limiting and retries.

### Phase 2: Caption Generator

```bash
python src/pipeline/caption_generator.py
```

**What it does:**
- Scans `data/raw/images/` for all posters and stills
- For each image, calls Gemini Flash vision API with the prompt:
  ```
  Describe this film image in 2-3 sentences. Focus on visual mood, color palette,
  composition, and atmosphere. Do not guess plot details.
  ```
- Saves captions as JSON: `data/processed/captions.json`

**Why auto-caption?**
This is one of the ablation experiments. We compare:
- CLIP-only retrieval (direct image embeddings)
- Caption-only retrieval (text search over auto-generated descriptions)
- Hybrid RRF fusion

If caption-only underperforms CLIP-only, that tells us the bottleneck is linguistic
description, not visual encoding.

**Exit codes:**
- 0: Success
- 1: Gemini API key missing or invalid
- 2: Image file not found or corrupted

### Phase 3: KB Builder

```bash
python src/pipeline/kb_builder.py
```

**What it does:**
- Loads raw film metadata from `data/raw/films/`
- Loads captions from `data/processed/captions.json`
- Creates text documents (plot + reviews + captions) for each film
- Embeds text with `sentence-transformers/all-MiniLM-L6-v2`
- Embeds images with `sentence-transformers/clip-ViT-B-32`
- Indexes into two ChromaDB collections:
  - `cineagent_text` (text embeddings)
  - `cineagent_images` (CLIP embeddings)
- Saves persistent index to `data/indices/`

**Exit codes:**
- 0: Success
- 1: Missing input data (run tmdb_fetcher first)
- 2: ChromaDB indexing failed

---

## Troubleshooting

### "TMDB API key not found"
- Check that `.env` exists in the project root
- Check that `TMDB_API_KEY=your_key_here` is set (no quotes, no spaces)
- Verify your key at https://developer.themoviedb.org/settings/api

### "Gemini API quota exceeded"
- Gemini Flash free tier: 15 requests per minute, 1500 per day
- For 500 films × 4 images = 2000 captions, you may hit the daily limit
- Solution: run caption_generator.py over 2 days, or upgrade to paid tier (~$1 total)

### "ChromaDB persistent directory not found"
- The `data/indices/` directory should auto-create on first run
- If it does not, manually create: `mkdir -p data/indices`
- Check `src/config.py` for `CHROMA_PERSIST_DIR` setting

### "Rate limit exceeded (TMDB)"
- TMDB free tier: 40 requests per 10 seconds
- The fetcher auto-retries with exponential backoff (3 attempts)
- If you still hit limits, the fetcher will resume from where it left off

### "Image download failed for film XYZ"
- Some films do not have posters or stills on TMDB
- The fetcher tracks these in `data/failed_films.json`
- These films are excluded from the KB
- Check the log for: "No poster available" or "Insufficient stills"

### "Caption generation failed for image XYZ"
- Check that the image file exists and is a valid JPEG
- Check Gemini API key and quota
- The caption generator skips failed images and logs them
- Re-run caption_generator.py to retry only failed images

### "KB validation failed: collection is empty"
- This means Phase 3 completed but no documents were indexed
- Check that `data/raw/films/` contains JSON files
- Check that `data/processed/captions.json` exists
- Re-run `build_kb.py` to retry all three phases

---

## Verifying Success

After running `build_kb.py`, you should see:

```
=== KB BUILD COMPLETE ===
Text documents indexed: 500
Image documents indexed: 2000 (500 films × 4 images each)
Total films in KB: 500
Collections:
  - cineagent_text
  - cineagent_images
Ready for Phase 2: Retrieval Layer
```

To manually verify:

```bash
# Check that collections exist
ls data/indices/

# Expected output:
# chroma.sqlite3  (ChromaDB storage)
# cineagent_text/ (text collection)
# cineagent_images/ (image collection)

# Check film count
python -c "
import chromadb
client = chromadb.PersistentClient(path='data/indices')
text_coll = client.get_collection('cineagent_text')
print(f'Text docs: {text_coll.count()}')
img_coll = client.get_collection('cineagent_images')
print(f'Image docs: {img_coll.count()}')
"
```

Expected output:
```
Text docs: 500
Image docs: 2000
```

---

## What's Next?

Phase 1 is complete. You now have:
- 500 films with metadata, posters, and scene stills
- Auto-generated captions for all images
- Two ChromaDB collections (text + images) ready for retrieval

Next steps:
- **Phase 2**: Implement retrieval layer (text, CLIP, caption, hybrid RRF)
- **Phase 3**: Build LangGraph agent (5 nodes)
- **Phase 4**: Run evaluation (3 variants, 2 ablations, 4 query families)

See `docs/RESEARCH.md` for the full build plan.

---

## Files Created by Phase 1

```
data/
├── personal_films.json         ← your curated list (if you ran film_list_builder)
├── failed_films.json           ← films that could not be fetched
├── raw/
│   ├── films/                  ← TMDB metadata (500 JSON files)
│   └── images/                 ← posters + stills (2000 JPEGs)
├── processed/
│   └── captions.json           ← auto-generated image descriptions
└── indices/
    ├── chroma.sqlite3          ← ChromaDB storage
    ├── cineagent_text/         ← text collection
    └── cineagent_images/       ← image collection
```

---

## Configuration Reference

All settings live in `src/config.py`. Key settings for Phase 1:

```python
FILMS_TARGET_COUNT = 500        # Total films to fetch
STILLS_PER_FILM = 3             # Scene stills per film
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer
IMAGE_EMBEDDING_MODEL = "clip-ViT-B-32"    # CLIP model
GEMINI_MODEL = "gemini-1.5-flash"          # Caption generation
```

Do not change these unless you are running a controlled ablation experiment.
If you change the embedding models, you must rebuild the entire KB.

---

## Cost Estimate

- TMDB API: Free (40 req/10s limit)
- Gemini Flash: ~2000 caption requests = $0.50 (free tier: 1500/day)
- ChromaDB: Local, no cost
- **Total: $0-1 depending on free tier usage**

---

## Need Help?

- Check logs: all scripts use Python `logging` module at INFO level
- Run tests: `pytest tests/test_pipeline.py -v`
- See `ARCHITECTURE.md` for design decisions
- See `RESEARCH.md` for research rationale
