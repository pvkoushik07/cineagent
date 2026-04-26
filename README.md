# CineAgent
### A Personalised Multimodal Film Discovery Agent
**INFS4205/7205 Assignment 3 — University of Queensland**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Workflow-1c3c3c?style=flat-square)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-7b61ff?style=flat-square)
![Gemini](https://img.shields.io/badge/LLM-Gemini%20Flash-4285f4?style=flat-square&logo=google)
![Multimodal RAG](https://img.shields.io/badge/Retrieval-Multimodal%20RAG-green?style=flat-square)

---

## Research Question

> When users express film preferences through natural language that evolves
> across a conversation, does a multimodal agent with a dynamic taste profile
> outperform a static RAG pipeline and a plain LLM on personalised
> recommendation accuracy? And across the retrieval design space, which
> combination of text, CLIP poster embeddings, CLIP scene-still embeddings,
> and auto-generated image captions produces the highest Recall@5 for mood
> and aesthetic queries?

---

## Quick Start (Reproduce Everything in 5 Steps)

### Prerequisites
- Python 3.11+
- Free TMDB API key → https://developer.themoviedb.org
- Free Google Gemini API key → https://aistudio.google.com

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your TMDB_API_KEY and GEMINI_API_KEY
```

### 3. Build the knowledge base
```bash
python src/pipeline/kb_builder.py
# Takes ~20-30 minutes for 500 films (TMDB rate limit)
# Saves to data/indices/
```

### 4. Run the agent interactively
```bash
python src/agent/graph.py
```

### 5. Reproduce evaluation results
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

See `docs/ARCHITECTURE.md` for full design rationale.

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

## Project Structure

```
cineagent/
├── CLAUDE.md               ← Claude Code instructions (read first)
├── docs/
│   ├── RESEARCH.md         ← hypothesis, ablation plan, ground truth
│   └── ARCHITECTURE.md     ← all design decisions with rationale
├── src/
│   ├── config.py           ← all paths and settings
│   ├── pipeline/           ← data fetching and KB building
│   ├── retrieval/          ← 4 retrieval variants
│   ├── agent/              ← LangGraph workflow
│   └── evaluation/         ← metrics and harness
├── notebooks/              ← step-by-step reproducible analysis
├── tests/                  ← pytest unit tests
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v
```
