# CineAgent — Personalised Multimodal Film Discovery Agent
## INFS4205/7205 Assignment 3 | UQ | Solo Project

---

## What This Project Is

A multimodal LangGraph agent that answers personalised film queries by retrieving
across three modalities: plot text, poster images, and scene stills. The system
maintains a dynamic taste profile that updates across conversation turns.

**Research Question:**
> When users express film preferences through natural language that evolves across
> a conversation, does a multimodal agent with a dynamic taste profile outperform
> a static RAG pipeline and a plain LLM on personalised recommendation accuracy?
> And across the retrieval design space, which combination of text embeddings,
> CLIP poster embeddings, CLIP scene-still embeddings, and auto-generated image
> captions produces the highest Recall@5 for mood and aesthetic queries?

**This is a research project, not just a chatbot.** Every implementation decision
must be justifiable as a design choice that we can ablate and measure.

---

## Project Structure

```
cineagent/
├── CLAUDE.md                   ← you are here (read every session)
├── docs/
│   ├── RESEARCH.md             ← hypothesis, ablation plan, metrics
│   ├── ARCHITECTURE.md         ← system design decisions + rationale
│   └── DATASET.md              ← KB description, what data, why
├── src/
│   ├── pipeline/               ← data ingestion, KB building
│   │   ├── tmdb_fetcher.py     ← fetch movies, posters, stills from TMDB
│   │   ├── caption_generator.py← auto-caption images with Gemini Flash
│   │   └── kb_builder.py       ← assemble and index into ChromaDB
│   ├── retrieval/              ← all retrieval strategies
│   │   ├── text_retriever.py   ← MiniLM dense text search
│   │   ├── clip_retriever.py   ← CLIP image search (posters + stills)
│   │   ├── caption_retriever.py← text search over auto-captions
│   │   └── hybrid_retriever.py ← RRF fusion across all retrievers
│   ├── agent/                  ← LangGraph workflow
│   │   ├── graph.py            ← main StateGraph definition
│   │   ├── nodes.py            ← all 5 node implementations
│   │   ├── state.py            ← AgentState TypedDict definition
│   │   └── tools.py            ← LangChain tool wrappers
│   └── evaluation/             ← evaluation harness
│       ├── test_suite.py       ← 4 query families, ground truth
│       ├── metrics.py          ← Recall@k, MRR, latency, tool calls
│       └── run_eval.py         ← runs all variants, saves results
├── notebooks/
│   ├── 01_data_pipeline.ipynb  ← build KB interactively
│   ├── 02_retrieval_ablation.ipynb ← compare retrieval variants
│   ├── 03_agent_demo.ipynb     ← demo the full agent
│   └── 04_evaluation.ipynb     ← final results and charts
├── data/
│   ├── raw/                    ← TMDB API responses (JSON)
│   ├── processed/              ← cleaned docs, image paths, captions
│   └── indices/                ← ChromaDB persistent storage
├── tests/
│   ├── test_retrievers.py
│   ├── test_agent_nodes.py
│   └── test_evaluation.py
└── .claude/
    ├── settings.json
    └── commands/               ← custom slash commands
        ├── build-kb.md
        ├── run-eval.md
        ├── test.md
        └── status.md
```

---

## Tech Stack (Never Change Without Updating ARCHITECTURE.md)

| Component | Tool | Why |
|-----------|------|-----|
| Data source | TMDB API (free) | Posters + stills + metadata + plot text |
| Text embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free, CPU-only, fast |
| Image embeddings | CLIP (clip-ViT-B-32 via sentence-transformers) | Text-image shared space |
| Vector DB | ChromaDB (local) | No cloud account, fully reproducible |
| Hybrid fusion | Reciprocal Rank Fusion (custom) | Standard, explainable |
| Agent framework | LangGraph | Assignment requirement |
| LLM | Google Gemini 1.5 Flash | Cheap (~$1-3 total), natively multimodal |
| LLM fallback | Ollama + llava | Free, local, for development |
| Evaluation | ragas + custom harness | LLM-as-judge + Recall@k |
| Captioning | Gemini Flash vision | Auto-caption posters and stills |

---

## The 5 LangGraph Nodes (Agent Architecture)

```
User Query
    │
    ▼
[1] QueryRouter          → classifies: factual / visual / hybrid / multi-hop
    │
    ▼
[2] RetrievalPlanner     → selects tools: text_search / clip_search / hybrid
    │
    ▼
[3] TasteProfileUpdater  → extracts preference signals, updates profile in state
    │
    ▼
[4] AnswerSynthesiser    → calls Gemini Flash with retrieved context + images
    │
    ▼
[5] Verifier             → checks: already watched? contradicts preferences?
    │                       re-routes to RetrievalPlanner if fails check
    ▼
Response
```

---

## Knowledge Base Design

**Target size:** ~500 films (personalised selection: must include films you have
watched or want to watch — this satisfies the "genuinely personalised" requirement)

**Per film documents:**
- `plot_text`: plot summary + genre + director + cast (text modality)
- `reviews_text`: 2-3 critic reviews excerpts (text modality)
- `poster_image`: official poster JPEG (image modality — stylised art)
- `still_images`: 3 scene stills JPEGs (image modality — real photographs)
- `poster_caption`: auto-generated by Gemini Flash vision
- `still_captions`: auto-generated by Gemini Flash vision
- `metadata`: year, runtime, language, rating, genres (structured)

**Two ChromaDB collections:**
- `text_collection`: plot + reviews + captions (MiniLM embeddings)
- `image_collection`: posters + stills (CLIP embeddings)

---

## Evaluation Plan (Do Not Deviate)

### 4 Required Query Families + Test Cases

**Family 1 — Factual Retrieval**
- Q: "Who directed Mulholland Drive and what year was it released?"
- Expected: Retrieved plot_text doc for Mulholland Drive
- Metric: Recall@5 (correct doc in top 5?)

**Family 2 — Cross-Modal (image-dependent)**
- Q: "Find me a film with a cold, desaturated, rain-soaked visual atmosphere"
- Expected: CLIP retrieval on stills surfaces films like Blade Runner, Se7en, Prisoners
- Metric: Recall@5 — text-only system MUST fail this; hybrid must succeed
- This is the KEY experiment proving the research hypothesis

**Family 3 — Multi-Hop Synthesis**
- Q: "I want something like Parasite — dark social commentary, non-English, recent"
- Expected: Agent decomposes into: retrieve by mood/tone + filter by language + filter by year
- Metric: Task success rate (does final answer satisfy all 3 constraints?)

**Family 4 — Conversational Follow-Up / Memory**
- Turn 1: "I love slow-burn psychological thrillers"
- Turn 2: "Preferably non-English, pre-2010"
- Turn 3: "I've already seen Oldboy and Cache, suggest something else"
- Expected: Each turn narrows correctly using memory; Turn 3 filters watched list
- Metric: Recall@5 per turn + preference adherence score

### 3 System Variants to Compare
- **Variant A**: Plain Gemini Flash (no retrieval, no memory) — baseline
- **Variant B**: Fixed RAG pipeline (text+CLIP hybrid, no router, no memory)
- **Variant C**: Full CineAgent (router + memory + taste updater + verifier)

### 2 Ablations
- **Ablation 1** (retrieval design): text-only vs caption-only vs CLIP-only vs hybrid RRF
- **Ablation 2** (memory design): no-memory vs static-memory vs dynamic-taste-updater

### Required Metrics
- Recall@5 (retrieval quality)
- ragas faithfulness score (answer groundedness)
- End-to-end latency in ms (efficiency)
- Tool call count per query (efficiency)

---

## Coding Standards

- Python 3.11+
- Type hints on all function signatures — no bare `def func(x)`
- Docstrings on every class and public method (one line minimum)
- No hardcoded API keys — use `.env` + `python-dotenv`
- No hardcoded file paths — use `pathlib.Path` and `config.py`
- All ChromaDB operations go through `src/retrieval/` — never inline
- All LLM calls go through `src/agent/` — never inline in pipeline code
- Tests for every retriever and every agent node
- Log with Python `logging` module, not `print()`

---

## Environment Variables (.env — never commit this file)

```
TMDB_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
CHROMA_PERSIST_DIR=./data/indices
OLLAMA_BASE_URL=http://localhost:11434
LOG_LEVEL=INFO
```

---

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build the knowledge base (run once)
python src/pipeline/kb_builder.py

# Run the agent interactively
python src/agent/graph.py

# Run evaluation suite
python src/evaluation/run_eval.py

# Run tests
pytest tests/ -v

# Launch notebooks
jupyter lab notebooks/
```

---

## Current Status

See `docs/RESEARCH.md` for what has been built and what remains.
Update this section as phases complete:

- [ ] Phase 1: Data pipeline + KB building
- [ ] Phase 2: Retrieval layer (all 4 variants)
- [ ] Phase 3: LangGraph agent (all 5 nodes)
- [ ] Phase 4: Evaluation harness + results
- [ ] Phase 5: Report

---

## Hard Rules (Read Before Every Session)

1. **Never implement without checking RESEARCH.md first** — every feature must
   map to a rubric criterion or evaluation requirement
2. **Never change the tech stack** without updating ARCHITECTURE.md
3. **Every retrieval variant must stay independently runnable** — ablations require
   each variant to work in isolation, not just as part of the full system
4. **Evaluation ground truth is fixed** — do not change test queries after
   evaluation has started or results become invalid
5. **Keep CLAUDE.md under 200 lines of actual instructions** — detail lives in
   docs/, not here
