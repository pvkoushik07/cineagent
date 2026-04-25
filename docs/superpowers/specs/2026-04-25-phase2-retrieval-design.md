# Phase 2 Retrieval Layer Testing & Ablation Design
**Date:** 2026-04-25  
**Phase:** 2 — Retrieval Layer Validation & Ablation Analysis  
**Status:** Approved

---

## Overview

Validate the existing retrieval implementations against the real knowledge base (469 films) and perform Ablation 1 to compare all 4 retrieval variants on ground truth queries.

## Context

Phase 1 is complete:
- ✅ 469 films in KB (212 personal + 300 auto-discovered, deduplicated)
- ✅ 2361 image captions generated (Gemini Flash)
- ✅ ChromaDB collections built (text_collection, image_collection)
- ✅ All 18 ground truth films present in KB

Retrieval code exists from initial scaffolding:
- `text_retriever.py` (MiniLM on plots/reviews)
- `caption_retriever.py` (MiniLM on Gemini captions)
- `clip_retriever.py` (CLIP on posters/stills)
- `hybrid_retriever.py` (RRF fusion of all 3)

All unit tests passing (12/12). Need integration testing and ablation analysis.

---

## Design Approach

**Selected: Integration Testing + Ablation Notebook**

Since retrieval code already exists and unit tests pass, Phase 2 focuses on:
1. **Integration testing** - verify retrievers work with real KB
2. **Ablation analysis** - compare 4 variants on ground truth queries
3. **Documentation** - record results, update RESEARCH.md

**Why this approach:**
- Retrieval code is mature (from scaffolding)
- Real KB just built, need validation
- Ablation 1 is critical for research question
- Fastest path to proving/disproving hypothesis

---

## Architecture

### Retrieval Layer Components

```
User Query (text)
    │
    ├──> TextRetriever
    │      ├─> Embed query with MiniLM (384-dim)
    │      └─> Search text_collection (plots + captions)
    │
    ├──> CaptionRetriever  
    │      ├─> Embed query with MiniLM (384-dim)
    │      └─> Search text_collection (captions only)
    │
    ├──> CLIPRetriever
    │      ├─> Embed query with CLIP (512-dim)
    │      └─> Search image_collection (posters + stills)
    │
    └──> HybridRetriever
           ├─> Call all 3 retrievers
           ├─> Apply RRF fusion (k=60)
           └─> Return top-5 fused results
```

### Key Design Principles

1. **Independent executability** - Each retriever works standalone (not coupled to agent)
2. **Standardized output format** - All return list of dicts with same schema
3. **ChromaDB abstraction** - Retrievers hide ChromaDB implementation details
4. **Performance target** - < 2 seconds per query per retriever

---

## Component Specifications

### Component 1: Integration Test Suite

**File:** `tests/test_retrieval_integration.py` (new)

**Purpose:** Verify retrievers work with real KB, not just mock data.

**Test cases:**

1. **test_text_retriever_factual_query**
   - Query: "Who directed Mulholland Drive?"
   - Expected: Mulholland Drive plot doc in top-5 results
   - Validates: Text retriever connects to real KB

2. **test_clip_retriever_visual_query**
   - Query: "cold desaturated rain-soaked atmosphere"
   - Expected: Blade Runner 2049 still in top-5
   - Validates: CLIP retriever + text-to-image embedding

3. **test_caption_retriever_visual_query**
   - Query: "neon-lit urban nightscape"
   - Expected: Captioned image doc in top-5
   - Validates: Gemini captions are searchable

4. **test_hybrid_retriever_combines_sources**
   - Query: "psychological thriller with twist ending"
   - Expected: Results from multiple modalities in top-5
   - Validates: RRF fusion works correctly

5. **test_retriever_latency**
   - Run 10 queries through each retriever
   - Expected: Mean latency < 2 seconds
   - Validates: Performance is acceptable

**Success criteria:** All 5 tests pass

---

### Component 2: Ablation Notebook

**File:** `notebooks/02_retrieval_ablation.ipynb` (new)

**Purpose:** Compare all 4 retrieval variants on ground truth queries to test hypothesis.

**Hypothesis (from RESEARCH.md):**
> Hybrid RRF will outperform all single-modality variants on Family 2 (visual) queries. CLIP-only will outperform text-only on Family 2. Text-only will outperform CLIP-only on Family 1 (factual).

**Notebook structure:**

#### Cell 1: Setup
```python
import sys
sys.path.insert(0, "../src")

from retrieval.text_retriever import TextRetriever
from retrieval.caption_retriever import CaptionRetriever
from retrieval.clip_retriever import CLIPRetriever
from retrieval.hybrid_retriever import HybridRetriever

text_ret = TextRetriever()
caption_ret = CaptionRetriever()
clip_ret = CLIPRetriever()
hybrid_ret = HybridRetriever()
```

#### Cell 2: Ground Truth Queries
```python
# Family 1: Factual queries (5 queries)
family1 = {
    "F1_01": {
        "query": "Who directed Mulholland Drive?",
        "correct_doc_id": "1018_plot",  # Mulholland Drive plot
        "correct_film_id": "1018"
    },
    "F1_02": {
        "query": "What year was Parasite released?",
        "correct_doc_id": "496243_plot",
        "correct_film_id": "496243"
    },
    # ... 3 more factual queries
}

# Family 2: Visual/mood queries (5 queries)
family2 = {
    "F2_01": {
        "query": "cold, desaturated, rain-soaked visual atmosphere",
        "correct_doc_id": "335984_still_1",  # Blade Runner 2049
        "correct_film_id": "335984"
    },
    "F2_02": {
        "query": "warm, golden, dusty western landscape",
        "correct_doc_id": "7345_still_0",  # There Will Be Blood
        "correct_film_id": "7345"
    },
    # ... 3 more visual queries
}

all_queries = {**family1, **family2}
```

#### Cell 3: Recall@5 Metric
```python
def recall_at_k(results: list[dict], correct_film_id: str, k: int = 5) -> bool:
    """Check if correct film appears in top-k results."""
    top_k_film_ids = [r["film_id"] for r in results[:k]]
    return correct_film_id in top_k_film_ids

def evaluate_retriever(retriever, queries: dict, k: int = 5) -> dict:
    """Run all queries, compute Recall@k."""
    hits = 0
    results_log = []
    
    for qid, qdata in queries.items():
        results = retriever.retrieve(qdata["query"], top_k=k)
        hit = recall_at_k(results, qdata["correct_film_id"], k)
        hits += hit
        results_log.append({
            "query_id": qid,
            "query": qdata["query"],
            "hit": hit,
            "top_result": results[0]["title"] if results else None
        })
    
    recall = hits / len(queries)
    return {"recall_at_k": recall, "hits": hits, "total": len(queries), "log": results_log}
```

#### Cell 4: Run Ablation
```python
import pandas as pd

# Run all 4 variants
results = {
    "Text-only": {
        "Family 1": evaluate_retriever(text_ret, family1),
        "Family 2": evaluate_retriever(text_ret, family2),
    },
    "Caption-only": {
        "Family 1": evaluate_retriever(caption_ret, family1),
        "Family 2": evaluate_retriever(caption_ret, family2),
    },
    "CLIP-only": {
        "Family 1": evaluate_retriever(clip_ret, family1),
        "Family 2": evaluate_retriever(clip_ret, family2),
    },
    "Hybrid RRF": {
        "Family 1": evaluate_retriever(hybrid_ret, family1),
        "Family 2": evaluate_retriever(hybrid_ret, family2),
    },
}

# Build comparison table
table_data = []
for variant, families in results.items():
    f1_recall = families["Family 1"]["recall_at_k"]
    f2_recall = families["Family 2"]["recall_at_k"]
    overall = (f1_recall + f2_recall) / 2
    table_data.append({
        "Variant": variant,
        "Family 1 (Factual)": f"{f1_recall:.2f}",
        "Family 2 (Visual)": f"{f2_recall:.2f}",
        "Overall": f"{overall:.2f}"
    })

df = pd.DataFrame(table_data)
print(df.to_markdown(index=False))
```

#### Cell 5: Analysis (Markdown)
```markdown
## Ablation 1 Results

**Hypothesis validation:**
- ✅/❌ Hybrid > single-modality on Family 2?
- ✅/❌ CLIP > Text on Family 2?
- ✅/❌ Text > CLIP on Family 1?

**Key findings:**
- [Describe what the results show]
- [Any unexpected patterns?]
- [Implications for final system design]

**Next steps:**
- Use Hybrid RRF for final agent (Phase 3)
- Document failure modes in RESEARCH.md
```

**Expected runtime:** ~2 minutes (10 queries × 4 variants × ~3s)

---

## Documentation Updates

### 1. Retriever Docstring Examples

Add usage examples to each retriever file:

```python
# Example for text_retriever.py
"""
Usage:
    from retrieval.text_retriever import TextRetriever
    
    retriever = TextRetriever(top_k=5)
    results = retriever.retrieve("Who directed Mulholland Drive?")
    
    for result in results:
        print(f"{result['title']}: {result['score']:.3f}")
        print(f"  {result['content'][:100]}...")
"""
```

### 2. RESEARCH.md Updates

Mark Phase 2 complete:

```markdown
### Phase 2 — Retrieval Layer ✓ COMPLETE
- [x] Integration tests with real KB (469 films)
- [x] Ablation 1: text vs caption vs CLIP vs hybrid
- [x] Notebook 02: Recall@5 comparison table
- [x] Hypothesis validation: Hybrid RRF outperforms single-modality
- [x] Performance verified: < 2s latency per query
- [x] All 4 variants independently testable

**Ablation 1 Results:** See `notebooks/02_retrieval_ablation.ipynb`
```

---

## Success Criteria

Phase 2 is complete when:

✅ Integration test suite exists and passes (5 tests)  
✅ All 4 retrievers connect to real KB  
✅ Ablation notebook runs end-to-end without errors  
✅ Recall@5 comparison table generated  
✅ Hypothesis validated or refuted (document both outcomes)  
✅ RESEARCH.md Phase 2 marked complete  
✅ Git commit with "Phase 2 complete" message  

**Validation command:**
```bash
# Run integration tests
pytest tests/test_retrieval_integration.py -v

# Run ablation notebook
jupyter nbconvert --to notebook --execute notebooks/02_retrieval_ablation.ipynb

# Check Phase 2 checkbox in RESEARCH.md
grep "Phase 2.*COMPLETE" docs/RESEARCH.md
```

---

## Known Limitations & Future Work

1. **Ground truth may need refinement** - First run of ablation may reveal that some "correct" docs are not actually the best matches. Document and adjust.

2. **CLIP text embeddings may be weaker than image embeddings** - If CLIP-only underperforms caption-only on Family 2, this is a valid finding (CLIP's text encoder is less refined than its vision encoder).

3. **RRF constant k=60 is not tuned** - Using standard value. Could grid-search k ∈ [20, 100] for optimal fusion, but not required for assignment.

4. **Latency not optimized** - Embedding models load fresh each query. For production, would cache embeddings or use a vector DB with pre-computed embeddings. Fine for research purposes.

---

## Next Steps

After Phase 2 completion:
1. Commit Phase 2 code and results
2. Review ablation findings with user
3. Proceed to Phase 3: LangGraph Agent implementation
4. Use Hybrid RRF as the retrieval tool in agent's RetrievalPlanner node
