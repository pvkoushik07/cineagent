# Dataset Description
## CineAgent Knowledge Base

This document satisfies the "genuinely personalised" KB requirement by explaining
*what* data is in the KB, *why* those films were chosen, and *how* each modality
is represented.

---

## Film Selection Criteria (Personalisation Evidence)

The 500 films were not randomly selected. They were curated across five
personal categories:

**Category 1 — Personal watchlist (~150 films)**
Films I have watched and have strong opinions about. Forms the ground truth
backbone for evaluation since I know whether a recommendation is good.

**Category 2 — Directors I follow (~100 films)**
Complete (or near-complete) filmographies of directors whose work I actively
seek out:
- David Fincher (Fight Club, Se7en, Zodiac, Gone Girl, The Social Network)
- Denis Villeneuve (Arrival, Blade Runner 2049, Prisoners, Incendies, Dune)
- Park Chan-wook (Oldboy, The Handmaiden, Sympathy for Mr. Vengeance)
- Alfonso Cuarón (Y Tu Mamá También, Children of Men, Gravity, Roma)
- Christopher Nolan (Memento, The Prestige, Inception, Interstellar)

**Category 3 — Genre deep-dives (~100 films)**
Genres I want to explore more deeply:
- Slow-burn psychological thrillers
- Non-English arthouse cinema
- Neo-noir

**Category 4 — Want-to-watch list (~100 films)**
Films I have not seen but intend to. These are important for the conversational
query family — the agent must recommend from this list without knowing I
haven't seen them.

**Category 5 — Canonical reference films (~50 films)**
Well-known films that serve as landmarks for describing taste ("something like
Parasite", "in the style of Blade Runner"). Ensures the KB can ground
comparative queries.

---

## Per-Film Documents

Each film generates the following documents in the KB:

### Text documents (→ text_collection in ChromaDB)

| Document type | Source | Content | Avg length |
|---------------|--------|---------|------------|
| plot_text | TMDB overview + Wikipedia | Full plot summary + genre + director + cast | ~400 words |
| reviews_text | Scraped critic excerpts | 2–3 short critic review excerpts | ~200 words |
| poster_caption | Gemini Flash vision | Auto-generated description of the poster image | ~80 words |
| still_captions (×3) | Gemini Flash vision | Auto-generated description of each scene still | ~60 words each |

### Image documents (→ image_collection in ChromaDB)

| Document type | Source | Size | CLIP model |
|---------------|--------|------|-----------|
| poster_image | TMDB /poster_path | 500px wide JPEG | clip-ViT-B-32 |
| still_image_1 | TMDB /images/backdrops | 780px wide JPEG | clip-ViT-B-32 |
| still_image_2 | TMDB /images/backdrops | 780px wide JPEG | clip-ViT-B-32 |
| still_image_3 | TMDB /images/backdrops | 780px wide JPEG | clip-ViT-B-32 |

### Structured metadata (stored as ChromaDB .metadata fields)

```json
{
  "film_id": "27205",
  "title": "Inception",
  "year": 2010,
  "director": "Christopher Nolan",
  "genres": ["Action", "Science Fiction", "Adventure"],
  "runtime_mins": 148,
  "language": "en",
  "tmdb_rating": 8.4,
  "tmdb_vote_count": 35000,
  "cast_top3": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"]
}
```

---

## KB Statistics (Target)

| Metric | Target |
|--------|--------|
| Total films | 500 |
| Text documents | ~3,500 (7 per film: plot + reviews + 1 poster caption + 3 still captions) |
| Image documents | ~2,000 (4 per film: 1 poster + 3 stills) |
| Total ChromaDB text_collection size | ~3,500 vectors |
| Total ChromaDB image_collection size | ~2,000 vectors |
| Total storage (images) | ~400MB |
| Total storage (indices) | ~50MB |

---

## Why Three Modalities Are Necessary

**Text alone** can answer: "Who directed Inception?" — yes. But it cannot answer:
"Find a film with a cold, clinical, sterile visual environment" — the answer
to this query lives in scene stills, not in a plot synopsis.

**Images alone** can answer visual mood queries but cannot answer: "What is the
plot of Parasite?" — there is no text in a poster.

**Captions** bridge both: they are text descriptions of images, enabling text
search to approximately reach visual content. However, caption quality is
bounded by the captioning model — our ablation tests whether raw CLIP embeddings
outperform caption-based text search for visual queries.

**All three together** cover the full query space across all four required
query families.
