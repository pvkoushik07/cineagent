# Architecture Decisions
## CineAgent — Every Design Choice With Its Rationale

When implementing, consult this file before choosing any library, pattern, or
data structure. If a decision is recorded here, do not override it without
updating this file first.

---

## Data Flow (End to End)

```
TMDB API
    │
    ├── plot text + reviews ──────────────────────────┐
    ├── poster JPEG ──→ Gemini caption ───────────────┤
    └── still JPEGs ──→ Gemini captions ──────────────┤
                                                       ▼
                                              ChromaDB text_collection
                                              (MiniLM embeddings)
                   poster JPEG ──→ CLIP ──────────────┐
                   still JPEGs ──→ CLIP ──────────────┤
                                                       ▼
                                              ChromaDB image_collection
                                              (CLIP embeddings)

User Query
    │
    ▼
LangGraph Agent
    ├── QueryRouter (classifies query type)
    ├── RetrievalPlanner (selects retrieval tools)
    │       ├── text_search_tool → text_collection
    │       ├── clip_search_tool → image_collection
    │       └── hybrid_tool → RRF(text_results, clip_results)
    ├── TasteProfileUpdater (updates AgentState)
    ├── AnswerSynthesiser (Gemini Flash multimodal)
    └── Verifier (checks watched list + preferences)
```

---

## Decision Log

### ADR-001: ChromaDB over Pinecone/Weaviate
**Decision:** Use ChromaDB local persistent storage
**Rationale:** Fully reproducible — reviewer can clone and run with zero cloud
accounts. Supports metadata filtering (.where()) natively. Handles two
separate collections cleanly. Fast enough for 500 films × 4 docs each = ~2000
vectors. Cost: free.
**Trade-off accepted:** No managed cloud hosting, no horizontal scale. Fine for
this project size.

### ADR-002: Two Separate Collections Over One
**Decision:** text_collection (MiniLM) and image_collection (CLIP) are separate
**Rationale:** Enables clean ablation — text-only retrieval uses only
text_collection, CLIP-only uses only image_collection, hybrid fuses both.
If we store everything in one collection we cannot isolate the modalities.
**Trade-off accepted:** Two embed operations per document at index time. Fine.

### ADR-003: CLIP via sentence-transformers, Not OpenAI CLIP API
**Decision:** Use clip-ViT-B-32 locally via sentence-transformers library
**Rationale:** Free, no API key, runs on CPU (slow but acceptable for ~2000
images indexed once). Embedding inference at query time is fast (< 200ms on
CPU for a single image). No ongoing cost.
**Trade-off accepted:** ViT-B-32 is smaller than ViT-L-14. Acceptable quality
for this domain.

### ADR-004: Gemini Flash for LLM (Not GPT-4o, Not Claude)
**Decision:** google-generativeai SDK, gemini-1.5-flash model
**Rationale:** Natively multimodal (text + images in same prompt). Free tier
covers all development. Paid tier costs ~$0.075/1M tokens — full project under
$3. Supports function calling for LangChain tool use.
**Trade-off accepted:** Not the most capable model. For a KB of 500 curated
films, Flash is more than capable.

### ADR-005: RRF Fusion (Not Learned Fusion)
**Decision:** Reciprocal Rank Fusion with k=60 (standard default)
**Rationale:** RRF is parameter-free, well-understood, and standard in the
retrieval literature. Learned fusion would require labelled relevance data we
don't have. RRF is explainable — important for the report.
**Formula:** RRF_score(d) = Σ 1/(k + rank_i(d)) for each ranked list i
**Trade-off accepted:** Not optimal — learned fusion with enough labels would
outperform. Out of scope.

### ADR-006: AgentState as TypedDict (Not Pydantic)
**Decision:** LangGraph state defined as TypedDict
**Rationale:** Native LangGraph pattern. Pydantic adds validation overhead not
needed for this project. TypedDict provides type hints for IDE support.
**State schema:**
```python
class AgentState(TypedDict):
    query: str
    query_type: str          # factual | visual | hybrid | multi_hop
    retrieved_docs: list
    retrieved_images: list
    taste_profile: dict      # {genres, directors, actors, avoid, watched}
    conversation_history: list
    tool_calls_count: int
    response: str
    verified: bool
```

### ADR-007: Taste Profile as Structured JSON in State
**Decision:** Taste profile is a dict with fixed keys stored in AgentState
**Rationale:** Structured profile enables the Verifier node to do deterministic
checks (is film in watched list?). Freeform text profile would require LLM to
parse it again at verification time.
**Schema:**
```json
{
  "preferred_genres": ["thriller", "drama"],
  "preferred_directors": ["David Fincher"],
  "preferred_languages": ["non-English"],
  "year_range": {"min": null, "max": 2010},
  "avoid_genres": ["romantic comedy"],
  "watched": ["oldboy", "cache", "parasite"],
  "mood_keywords": ["slow-burn", "psychological", "bleak"],
  "confidence": 0.8
}
```

### ADR-008: Notebooks for Pipeline + Evaluation, Scripts for Production
**Decision:** Use Jupyter notebooks for KB building and evaluation; Python
scripts for the agent itself
**Rationale:** Notebooks show step-by-step process visually — good for the
report and for reproducibility evidence. The agent itself (graph.py) is a
script because it needs to be importable by the evaluation harness.
**Rule:** Notebooks are for exploration and reporting. Scripts are for
production code. Never put business logic only in a notebook.

### ADR-009: ragas Configured to Use Gemini Flash (Not OpenAI)
**Decision:** Override ragas default LLM from OpenAI to Gemini Flash
**Rationale:** ragas defaults to OpenAI gpt-4 which costs more and requires
a separate API key. Gemini Flash is already in the stack.
**Implementation:**
```python
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
ragas_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-1.5-flash"))
```

### ADR-010: 500 Films, Curated Not Random
**Decision:** Manually select 500 films (not top-500-by-popularity)
**Rationale:** The assignment requires "genuinely personalised" KB. A random
top-500 from TMDB is not personalised. The selection must reflect personal taste
— films watched, want to watch, directors followed, genres preferred.
**Personalisation evidence:** The report must explain the curation criteria.
Example: "Films were selected to cover: my watchlist, films by directors I
follow (Fincher, Villeneuve, Park Chan-wook, etc.), and gaps in genres I want
to explore."

---

## Module Responsibilities (No Cross-Module Coupling)

```
src/pipeline/    → only writes to data/ directories, never imports from agent/
src/retrieval/   → only reads from data/indices/, never imports from agent/
src/agent/       → imports from retrieval/, never imports from pipeline/
src/evaluation/  → imports from agent/ and retrieval/, writes to data/results/
```

**Why:** This decoupling means each module can be tested independently and
each retrieval variant can be swapped without touching agent code.

---

## File Naming Conventions

- Python files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- ChromaDB collection names: `cineagent_text`, `cineagent_images`
- Film IDs: TMDB integer ID as string, e.g. `"27205"` for Inception

---

## What NOT to Build (Scope Boundaries)

- No web UI or frontend — CLI and notebooks only
- No user authentication — single-user system
- No deployed API — runs locally only
- No fine-tuning of any model — inference only
- No real-time TMDB queries during agent runtime — KB is pre-built offline
- No streaming responses — batch completion only
