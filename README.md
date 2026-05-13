# CineAgent
### A Personalised Multimodal Film Discovery Agent
**INFS4205/7205 Assignment 3 — University of Queensland**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Workflow-1c3c3c?style=flat-square)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-7b61ff?style=flat-square)
![Gemini](https://img.shields.io/badge/LLM-Gemini%20Flash-4285f4?style=flat-square&logo=google)
![Multimodal RAG](https://img.shields.io/badge/Retrieval-Multimodal%20RAG-green?style=flat-square)

---

## Overview

A multimodal LangGraph agent for personalised film recommendations, combining text embeddings, CLIP image search, BM25 keyword matching, and dynamic taste profiling. The system evaluates 3 architectural variants across 13 test queries in 4 query families.

**Quick Start:** `python verify_reproducibility.py` → `python demo.py --demo`

**Key Finding:** Simpler Fixed RAG architecture (61.5% Recall@5) outperforms the complex agent system (53.8%) — an honest negative result demonstrating that added architectural complexity doesn't always improve recommendation accuracy. Analysis available in evaluation notebooks.

---

## Research Question

> When users express film preferences through natural language that evolves
> across a conversation, does a multimodal agent with a dynamic taste profile
> outperform a static RAG pipeline and a plain LLM on personalised
> recommendation accuracy? And across the retrieval design space, which
> combination of text, CLIP poster embeddings, CLIP scene-still embeddings,
> and auto-generated image captions produces the highest Recall@5 for mood
> and aesthetic queries?

**Result:** The agent hypothesis was partially rejected. Text embeddings + BM25 hybrid retrieval dominated, CLIP failed on abstract mood queries, and routing/verification added latency without improving accuracy. See evaluation notebooks for detailed analysis.

---

## Getting Started

### Quick Test (2 minutes)

If the knowledge base is already built, you can test the agent immediately:

```bash
# 1. Verify setup
python verify_reproducibility.py

# 2. Run demo with pre-written queries
python demo.py --demo

# 3. Or try your own questions interactively
python demo.py
```

**Example queries to try:**
- "I want a psychological thriller with a twist ending"
- "Find me something visually stunning with minimal dialogue"
- "Dark social commentary film, non-English, after 2010"
- "Something like Parasite but set in America"

### Evaluation Test Suite

**Test queries location:** `src/evaluation/test_suite.py`

The evaluation suite contains 13+ test cases across 4 query families, each with:
- Full query text
- Ground truth film IDs (TMDB IDs)
- Expected film titles
- Design rationale

**4 Query Families:**

1. **Factual Retrieval** (5 queries)
   - Tests: Direct fact lookup (director, year, cast, genre)
   - Example: "Who directed Mulholland Drive and when was it released?"
   - Expected behavior: Text-only retrieval from plot documents
   - Metric: Recall@5

2. **Visual/Cross-Modal** (5 queries)
   - Tests: Aesthetic/mood queries requiring image understanding
   - Example: "cold desaturated rain-soaked urban visual atmosphere"
   - Expected behavior: CLIP image search on scene stills
   - **KEY TEST:** Text-only retrieval expected to fail (hypothesis validation)
   - Metric: Recall@5, comparing text-only vs CLIP vs hybrid

3. **Multi-Hop Synthesis** (3 queries)
   - Tests: Multiple independent constraints + metadata filtering
   - Example: "Dark social commentary, non-English, after 2010"
   - Expected behavior: Hybrid retrieval + BM25 + metadata filtering
   - Metric: Task success rate (all constraints satisfied?)

4. **Conversational/Memory** (2 sequences, 3-5 turns each)
   - Tests: Multi-turn preference evolution and memory
   - Example Turn 1: "I love slow-burn psychological thrillers"
   - Example Turn 2: "Preferably non-English, pre-2010"
   - Example Turn 3: "I've already seen Oldboy, suggest something else"
   - Expected behavior: Taste profile updates across turns, filters watched list
   - Metric: Recall@5 per turn + preference adherence score

**Sample test queries from each family:**

```python
# Family 1: Factual Retrieval
"Who directed Mulholland Drive and when was it released?"
"What is the plot of Parasite by Bong Joon-ho?"
"What genre is Oldboy and who stars in it?"

# Family 2: Visual/Cross-Modal (KEY HYPOTHESIS TEST)
"cold desaturated rain-soaked urban visual atmosphere"
"vibrant saturated color palette film"
"neon-lit urban cyberpunk aesthetic"

# Family 3: Multi-Hop Synthesis
"Dark social commentary, non-English, after 2010"
"True crime drama, 2000s, ensemble cast"
"Contemplative, Terrence Malick style, before 1980"

# Family 4: Conversational (multi-turn sequences)
Turn 1: "I love slow-burn psychological thrillers"
Turn 2: "Preferably non-English, pre-2010"
Turn 3: "I've already seen Oldboy and Cache, suggest something else"
```

**To view all 13 test queries with ground truth:**
```bash
# View raw test cases with TMDB IDs and design notes
cat src/evaluation/test_suite.py

# Or run the evaluation to see queries in action
python src/evaluation/run_eval.py --all
```

### Evaluation Results & Analysis

**Results available in:** `notebooks/04_evaluation.ipynb` and `data/results/eval_results.json`

**Quick summary:**
- **3 system variants compared:**
  - Variant A: Plain Gemini Flash (no retrieval, baseline)
  - Variant B: Fixed RAG pipeline (no routing, no memory) — **61.5% Recall@5**
  - Variant C: Full CineAgent (routing + memory + verification) — **53.8% Recall@5**

- **Key finding (honest negative result):**
  - Simpler Fixed RAG (Variant B) outperforms complex agent (Variant C)
  - BM25 sparse retrieval integration improved both variants significantly
  - CLIP embeddings failed on abstract mood queries (20% on visual queries)
  - Text embeddings (MiniLM) + BM25 keyword matching = best performance

**Performance summary table:**

| Variant | Description | Recall@5 | Latency (ms) | Tool Calls |
|---------|-------------|----------|--------------|------------|
| **A** | Plain Gemini (no retrieval) | 23.1% | 450 | 0 |
| **B** | Fixed RAG (text + BM25 + CLIP) | **61.5%** | 1,200 | 3.2 |
| **C** | Full Agent (routing + memory) | 53.8% | 1,800 | 5.1 |

**By query family:**

| Query Family | Variant A | Variant B | Variant C |
|--------------|-----------|-----------|-----------|
| Factual | 60% | **80%** | 80% |
| Visual | 0% | 40% | 20% |
| Multi-hop | 0% | **100%** | 66.7% |
| Conversational | 25% | 50% | **75%** |

**Visualizations:** All figures in `figures/` directory (10 PNG files):
- `fig1_variant_comparison.png` — Overall Recall@5 by variant
- `fig2_query_family_performance.png` — Performance breakdown by query type
- `fig3_latency_analysis.png` — End-to-end latency comparison
- `fig4_multihop_breakdown.png` — Multi-hop query analysis
- And 6 more detailed figures...

### Documentation & Code Overview

**Core Documentation:**
- ✅ `README.md` — Complete setup, usage, and architecture guide (this file)
- ✅ `notebooks/04_evaluation.ipynb` — Research question, hypothesis, results, and analysis

**Source Code:**
- ✅ `src/pipeline/` — Data ingestion, KB building (TMDB fetcher, caption generator)
- ✅ `src/retrieval/` — 5 retrieval strategies (text, CLIP, caption, BM25, hybrid RRF)
- ✅ `src/agent/` — LangGraph workflow (5 nodes: router, planner, updater, synthesiser, verifier)
- ✅ `src/evaluation/` — Test suite, metrics (Recall@k, MRR, latency), evaluation harness

**Testing & Demo:**
- ✅ `tests/` — pytest unit tests (retrievers, agent nodes, integration)
- ✅ `demo.py` — Interactive demo with pre-written queries
- ✅ `verify_reproducibility.py` — Automated setup verification
- ✅ `presentation_demo.py` — Live presentation demo script

**Evaluation Artifacts:**
- ✅ `src/evaluation/test_suite.py` — All test queries + ground truth
- ✅ `figures/` — 10 visualization figures (PNG)
- ✅ `notebooks/04_evaluation.ipynb` — Interactive results exploration

**Knowledge Base:**
- ✅ `data/indices/` — ChromaDB persistent storage (if provided)
- ✅ 531 films, 2,627 text docs, 1,914 image docs, 2,124 auto-generated captions

### How to Reproduce Results

**Option 1: Use pre-built knowledge base** (if available)
```bash
# KB already in data/indices/
python verify_reproducibility.py
python src/evaluation/run_eval.py --all
```

**Option 2: Build KB from scratch** (~30 minutes, requires API keys)
```bash
# Set up .env with TMDB_API_KEY and GEMINI_API_KEY
python src/pipeline/kb_builder.py
python src/evaluation/run_eval.py --all
```

**Expected output:**
- `data/results/eval_results.json` — Raw evaluation metrics
- Console output showing Recall@5 for each variant
- Notebook `notebooks/04_evaluation.ipynb` for interactive analysis

### Documentation Map

| Question | Location |
|----------|----------|
| Research question & hypothesis | `README.md` (Research Question section) |
| System architecture | `README.md` (System Architecture section) |
| Knowledge base design | `README.md` (Project Structure section), `src/pipeline/kb_builder.py` |
| All test queries + ground truth | `src/evaluation/test_suite.py` |
| Evaluation results | `notebooks/04_evaluation.ipynb`, `data/results/eval_results.json` |
| Performance visualizations | `figures/*.png` (10 files) |
| How agent nodes work | `src/agent/nodes.py` (with docstrings) |
| Retrieval implementations | `src/retrieval/*.py` (5 files) |
| Unit tests | `tests/test_*.py` (6 files) |

### Personalised Knowledge Base

This knowledge base is genuinely personalised, curated from actual viewing history rather than generic "top 500" lists.

**Collection metadata:** `data/personal_collection.json` (if included)
- 387 watched films with personal ratings (1.0-5.0)
- Watch dates spanning 2016-2026
- Rewatch counts (143 films rewatched)
- Personal notes on favorites
- 67 films owned on physical media
- Watchlist priorities and selection reasons

---

## Quick Start (Reproduce Everything in 5 Steps)

### Prerequisites
- Python 3.11+
- Free TMDB API key → https://developer.themoviedb.org
- Free Google Gemini API key → https://aistudio.google.com

**Note:** The project uses `google-generativeai` package which is deprecated. A deprecation warning will appear but functionality is unaffected. Migration to `google.genai` is planned for future work.

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your TMDB_API_KEY and GEMINI_API_KEY
```

### 3. Verify setup
```bash
python verify_reproducibility.py
```

### 4. Build the knowledge base (if not provided)
```bash
python src/pipeline/kb_builder.py
# Takes ~20-30 minutes for 500 films (TMDB rate limit)
# Saves to data/indices/
# Note: A pre-built KB may be available in data/indices/
```

### 5. Run the agent interactively
```bash
python src/agent/graph.py
# Or use the demo script:
python demo.py --demo
```

### 6. Reproduce evaluation results
```bash
python src/evaluation/run_eval.py --all
# Results saved to data/results/eval_results.json
jupyter lab notebooks/04_evaluation.ipynb
```

### Alternative: Ollama (fully free, no API keys)
```bash
# Install Ollama: https://ollama.ai
ollama pull llava
OLLAMA_BASE_URL=http://localhost:11434 python src/agent/graph.py
```

---

## System Architecture

```
Three modalities in the knowledge base:
  Text    → plot summaries + reviews (MiniLM embeddings → ChromaDB)
  Images  → posters + scene stills  (CLIP embeddings → ChromaDB)
  Captions→ auto-generated image descriptions (MiniLM embeddings → ChromaDB)

Five-node LangGraph agent:
  QueryRouter → RetrievalPlanner → TasteProfileUpdater → AnswerSynthesiser → Verifier
```

---

## Evaluation Summary

Three system variants compared across four query families:
- **Variant A**: Plain Gemini Flash (no retrieval)
- **Variant B**: Fixed RAG pipeline (no routing, no memory)
- **Variant C**: Full CineAgent (routing + dynamic taste memory)

Two ablations:
- **Ablation 1**: text-only vs caption-only vs CLIP-only vs hybrid RRF
- **Ablation 2**: no-memory vs static-memory vs dynamic-taste-updater

See `notebooks/04_evaluation.ipynb` for full results and failure analysis.

---

## Project Structure (Complete File Tree)

```
cineagent/
├── README.md                          ← Start here (setup guide)
├── requirements.txt                   ← Python dependencies (pinned versions)
├── .env.example                       ← Template for API keys
│
├── demo.py                            ← Interactive demo (5 pre-written queries)
├── verify_reproducibility.py         ← Setup verification script
├── presentation_demo.py               ← Live presentation demo
│
├── figures/                           📊 Evaluation visualizations (10 PNG files)
│   ├── fig1_variant_comparison.png
│   ├── fig2_query_family_performance.png
│   ├── fig3_latency_analysis.png
│   └── ... (7 more figures)
│
├── src/                               💻 Source code
│   ├── config.py                      ← All paths, API keys, settings
│   │
│   ├── pipeline/                      🔧 Data ingestion & KB building
│   │   ├── tmdb_fetcher.py            ← Fetch films, posters, stills from TMDB
│   │   ├── caption_generator.py       ← Auto-caption images with Gemini Flash
│   │   └── kb_builder.py              ← Assemble ChromaDB collections
│   │
│   ├── retrieval/                     🔍 5 Retrieval strategies
│   │   ├── text_retriever.py          ← MiniLM dense text embeddings
│   │   ├── clip_retriever.py          ← CLIP image embeddings (posters + stills)
│   │   ├── caption_retriever.py       ← Text search over auto-captions
│   │   ├── bm25_retriever.py          ← BM25 keyword/sparse retrieval
│   │   └── hybrid_retriever.py        ← RRF fusion across all retrievers
│   │
│   ├── agent/                         🤖 LangGraph workflow (5 nodes)
│   │   ├── graph.py                   ← StateGraph definition, compilation, runner
│   │   ├── nodes.py                   ← All 5 node implementations (300+ lines)
│   │   ├── state.py                   ← AgentState TypedDict definition
│   │   └── tools.py                   ← LangChain tool wrappers
│   │
│   └── evaluation/                    📈 Evaluation harness
│       ├── test_suite.py              ← **All 13 test queries + ground truth**
│       ├── metrics.py                 ← Recall@k, MRR, latency, tool count
│       └── run_eval.py                ← Runs all variants, saves results
│
├── tests/                             ✅ pytest unit tests
│   ├── test_retrievers.py             ← All 5 retrieval strategies
│   ├── test_agent_nodes.py            ← All 5 LangGraph nodes
│   ├── test_retrieval_integration.py  ← End-to-end retrieval workflows
│   ├── test_agent_integration.py      ← Full agent conversation flows
│   └── test_pipeline_integration.py   ← KB building pipeline
│
├── notebooks/                         📓 Jupyter notebooks
│   ├── 01_data_pipeline.ipynb         ← KB building walkthrough
│   ├── 02_retrieval_ablation.ipynb    ← Compare retrieval variants
│   ├── 03_agent_demo.ipynb            ← Interactive agent demo
│   └── 04_evaluation.ipynb            ← **Results analysis + visualizations**
│
├── data/                              💾 Data storage
│   ├── indices/                       ← ChromaDB persistent storage (2 collections)
│   │   ├── text_collection/           ← 2,627 text documents (MiniLM embeddings)
│   │   └── image_collection/          ← 1,914 image documents (CLIP embeddings)
│   │
│   ├── processed/                     ← Cleaned docs, image paths, captions
│   └── results/                       ← Evaluation outputs (eval_results.json)
│
└── scripts/                           🛠️ Utility scripts
    ├── test_conversational.py         ← Multi-turn conversation testing
    └── test_multihop.py               ← Multi-hop query testing
```

**Essential files:**
1. **`src/evaluation/test_suite.py`** — All test queries and ground truth
2. **`src/agent/nodes.py`** — LangGraph implementation (heavily documented)
3. **`notebooks/04_evaluation.ipynb`** — Complete evaluation results and analysis
4. **`figures/*.png`** — 10 evaluation visualizations
5. **`demo.py`** — Interactive demonstration

---

## Running Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_retrievers.py -v          # Retrieval layer tests
pytest tests/test_agent_nodes.py -v         # LangGraph node tests
pytest tests/test_retrieval_integration.py -v  # Integration tests
```

**Test coverage:**
- `test_retrievers.py` — All 5 retrieval strategies (text, CLIP, caption, BM25, hybrid)
- `test_agent_nodes.py` — All 5 LangGraph nodes (router, planner, updater, synthesiser, verifier)
- `test_retrieval_integration.py` — End-to-end retrieval workflows
- `test_agent_integration.py` — Full agent conversation flows
- `test_pipeline_integration.py` — KB building pipeline

---

## Tech Stack Details

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Text embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Free, CPU-only, 384-dim, fast inference |
| **Image embeddings** | CLIP-ViT-B-32 | Text-image shared embedding space, 512-dim |
| **Vector DB** | ChromaDB (persistent local) | No cloud account required, fully reproducible |
| **Keyword search** | BM25 (rank_bm25) | Exact keyword matching for thematic queries |
| **Hybrid fusion** | Reciprocal Rank Fusion (RRF) | Standard, explainable, no learned weights |
| **Agent framework** | LangGraph 0.1 | Assignment requirement, stateful workflows |
| **LLM** | Google Gemini 2.5 Flash | Cheap (~$1-3 total), natively multimodal |
| **Data source** | TMDB API (free tier) | Posters + stills + metadata, 40 req/10sec limit |
| **Captioning** | Gemini Flash vision API | Auto-generate image descriptions |
| **Evaluation** | Custom harness + Recall@k | Reproducible, no LLM-as-judge bias |

**No paid services required** — Everything runs locally except LLM API calls (Gemini Flash costs ~$0.002 per query).

**Alternative (fully free):** Use Ollama + llava locally:
```bash
ollama pull llava
export OLLAMA_BASE_URL=http://localhost:11434
python demo.py
```

---

## Understanding the Agent Workflow

**5-Node LangGraph Architecture:**

```
User Query
    │
    ▼
┌─────────────────┐
│  QueryRouter    │  ← Classifies: factual | visual | hybrid | multi_hop
│  (Gemini Flash) │     Extracts constraints (year, language, genre)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RetrievalPlanner│  ← Selects tools based on query type:
│                 │     • factual → text_search
└────────┬────────┘     • visual → clip_search + hybrid
         │              • multi_hop → hybrid + BM25 + metadata filtering
         ▼
┌─────────────────┐
│TasteProfileUpd. │  ← Extracts preferences from query
│                 │     Updates dynamic taste profile with confidence scores
└────────┬────────┘     Accumulates across conversation turns
         │
         ▼
┌─────────────────┐
│AnswerSynthesiser│  ← Calls Gemini Flash with:
│  (Gemini Flash) │     • Retrieved text context (plots, reviews)
└────────┬────────┘     • Retrieved images (posters, stills)
         │              • Current taste profile
         ▼              • Conversation history
┌─────────────────┐
│    Verifier     │  ← Validates output:
│                 │     • Already watched? (contradicts user history)
└────────┬────────┘     • Contradicts preferences?
         │              • Generic/hallucinated?
         ├─────────────► Re-route to RetrievalPlanner if fails (max 2 retries)
         │
         ▼
    Response
```

**Key implementation files:**
- `src/agent/graph.py` — StateGraph definition and compilation
- `src/agent/nodes.py` — All 5 node implementations (300+ lines with docstrings)
- `src/agent/state.py` — AgentState TypedDict definition
- `src/agent/tools.py` — LangChain tool wrappers for retrievers

---

## Repository Contents

**This repository includes:**

1. **Source code** — Complete `src/` directory with all modules
2. **Tests** — `tests/` directory with pytest unit tests
3. **Demo scripts** — `demo.py`, `verify_reproducibility.py`, `presentation_demo.py`
4. **Visualizations** — `figures/` with 10 PNG evaluation charts
5. **Evaluation notebooks** — `notebooks/04_evaluation.ipynb` with complete results analysis
6. **Knowledge base** — `data/indices/` (pre-built ChromaDB, if available)
7. **Requirements** — `requirements.txt` with pinned versions
8. **Configuration** — `.env.example` template

**Building the knowledge base:**
- If not pre-built, follow "Build KB from scratch" instructions
- Requires TMDB API key (free) and Gemini API key (free tier)
- Build time: ~30 minutes (TMDB rate limit: 40 requests per 10 seconds)

---

## Frequently Asked Questions

**Q: How do I know if the setup worked?**  
A: Run `python verify_reproducibility.py` — it checks all dependencies, API keys, and KB existence.

**Q: Where are the actual test queries used in evaluation?**  
A: `src/evaluation/test_suite.py` — lines 43-250 contain all queries with ground truth.

**Q: Can I see the raw evaluation results?**  
A: Yes, after running evaluation: `data/results/eval_results.json` (JSON format) or open `notebooks/04_evaluation.ipynb` for interactive exploration.

**Q: How is the knowledge base personalised?**  
A: See `data/personal_collection.json` (if available) for watch history, ratings, and personal notes spanning 10 years (2016-2026) with 387 watched films.

**Q: What if I don't have API keys?**  
A: Use Ollama (free, local): `ollama pull llava && export OLLAMA_BASE_URL=http://localhost:11434`

**Q: How long does evaluation take?**  
A: ~5-10 minutes for full evaluation suite (13 test cases × 3 variants = 39 queries).

**Q: Is there a quick way to test a single query?**  
A: Yes: `python demo.py` (interactive mode) or `python demo.py --quick` (3 pre-written queries).

**Q: How do I see what the agent is actually doing?**  
A: Check console output during demo — shows retrieval strategy, tool calls, latency, and taste profile updates. Or read `src/agent/nodes.py` for implementation details with extensive docstrings.

---

## Project Components Reference

**Key features and implementation locations:**

| Component | Implementation Location |
|-----------|------------------------|
| **Research question** | `README.md` (above), `notebooks/04_evaluation.ipynb` |
| **Hypothesis** | `README.md`, `notebooks/04_evaluation.ipynb` |
| **LangGraph agent** | `src/agent/graph.py` (StateGraph), `src/agent/nodes.py` (5 nodes) |
| **Multimodal knowledge base** | 3 modalities: text (MiniLM), images (CLIP), captions — `src/pipeline/kb_builder.py` |
| **Personalisation** | `data/personal_collection.json` (387 watched films, 10-year history) |
| **Retrieval tools** | `text_search`, `clip_search`, `hybrid_search` — `src/agent/tools.py` |
| **Agent workflow** | 5-node graph: router → planner → updater → synthesiser → verifier |
| **Memory & state** | Dynamic taste profile in `AgentState` — `src/agent/nodes.py:TasteProfileUpdater` |
| **Evaluation suite** | 4 query families, 13 test cases, 3 variants — `src/evaluation/test_suite.py` |
| **Ablation studies** | Text-only vs CLIP-only vs caption-only vs hybrid — `notebooks/04_evaluation.ipynb` |
| **Metrics** | Recall@5, MRR, latency, tool calls — `notebooks/04_evaluation.ipynb` |
| **Results** | Fixed RAG (61.5%) outperforms agent (53.8%) — see visualizations |
| **Visualizations** | 10 figures in `figures/` directory |
| **Code quality** | Type hints, docstrings, unit tests — `src/**/*.py`, `tests/` |
| **Reproducibility** | `verify_reproducibility.py`, `requirements.txt`, `.env.example` |

**Verification steps:**
- Run `python demo.py --demo` to test the agent interactively
- Run `pytest tests/ -v` to execute all unit tests
- Open `notebooks/04_evaluation.ipynb` for complete results analysis
- View `src/evaluation/test_suite.py` for all test queries
- Explore `figures/` for 10 PNG visualization files
- Review `src/agent/graph.py` for LangGraph implementation

---
