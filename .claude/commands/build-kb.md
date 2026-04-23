# Build Knowledge Base

Run the full data pipeline to fetch films from TMDB, generate image captions,
and index everything into ChromaDB.

## Steps

1. Check that TMDB_API_KEY and GEMINI_API_KEY are set in .env
2. Run `python src/pipeline/tmdb_fetcher.py` to fetch film data and images
3. Run `python src/pipeline/caption_generator.py` to auto-caption all images
4. Run `python src/pipeline/kb_builder.py` to embed and index into ChromaDB
5. Verify by checking data/indices/ directory exists and has content
6. Report: how many films indexed, how many images, any errors

## Expected Output

- data/raw/*.json — TMDB API responses
- data/processed/films.json — cleaned film documents
- data/processed/captions.json — image captions
- data/indices/cineagent_text/ — ChromaDB text collection
- data/indices/cineagent_images/ — ChromaDB image collection

## If It Fails

- TMDB rate limit: add time.sleep(0.25) between requests
- Gemini quota: caption in batches of 50, save progress to JSON
- ChromaDB: delete data/indices/ and re-run kb_builder.py
