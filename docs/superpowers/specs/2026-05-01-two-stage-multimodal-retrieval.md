# Two-Stage Multimodal Retrieval Design Specification

**Date:** 2026-05-01  
**Status:** Approved for Implementation  
**Target:** CineAgent Phase 3 Enhancement — Fix RQ/Implementation Mismatch

---

## Problem Statement

**Current Issue:**
- Research Question claims: "using both visual poster/scene embeddings and textual plot signals"
- Actual Implementation: Only uses text-based retrieval (TextRetriever + CaptionRetriever)
- CLIPRetriever is implemented but NEVER called in production agent (Variant C)
- Rubric C1 claims "novel CLIP+memory angle" but CLIP not used in final system

**Gap:**
System has multimodal DATA but not multimodal RETRIEVAL. This undermines novelty claim and creates mismatch between RQ and implementation.

**Assignment Risk:**
May not fully satisfy "Multimodal Integration" requirement if final system doesn't actually USE visual embeddings for retrieval.

---

## Solution: Two-Stage Query-Adaptive Retrieval

### Core Principle

**Stage 1 (Text Retrieval):** Ensures high recall — text gets relevant documents into candidate set  
**Stage 2 (CLIP Reranking):** Improves precision — visual embeddings refine ranking for final top-5

**Query-Adaptive Fusion:** Different query types use different text/CLIP weight balances:
- Factual queries: trust text (80%) over CLIP (20%)
- Visual queries: CLIP dominant (70%) over text (30%)
- Multi-hop: balanced (60%/40%)

This approach leverages strengths of both modalities while mitigating CLIP's weakness on factual queries.

---

## Architecture

### High-Level Flow

```
User Query
    ↓
QueryRouter → classifies query_type ∈ {factual, visual, multi_hop, hybrid}
    ↓
RetrievalPlanner (Variant D) calls TwoStageRetriever
    ↓
┌──────────────────────────────────────────────┐
│ Stage 1: Text Retrieval                      │
│  - TextRetriever.retrieve(query, top_k=20)   │
│  - Returns 20 candidates with text_score     │
│  - Ensures high recall (100% on factual)     │
└──────────────────────────────────────────────┘
    ↓ (20 candidates)
┌──────────────────────────────────────────────┐
│ Stage 2: CLIP Reranking                      │
│  - Encode query with CLIP (once)             │
│  - For each candidate:                       │
│    • Load poster + stills from metadata      │
│    • Compute CLIP similarity                 │
│    • Average across images                   │
│  - Normalize scores to [0,1]                 │
│  - Fuse: α×text_score + β×clip_score         │
│  - (α,β) determined by query_type            │
│  - Sort by fused score, return top-5         │
└──────────────────────────────────────────────┘
    ↓ (top-5 results)
AnswerSynthesiser
```

### Weight Mapping

| Query Type | Text Weight (α) | CLIP Weight (β) | Rationale |
|------------|----------------|----------------|-----------|
| factual    | 0.8            | 0.2            | Text excels (100% recall in Ablation 1) |
| visual     | 0.3            | 0.7            | CLIP should refine visual/mood ranking |
| multi_hop  | 0.6            | 0.4            | Balanced — both signals needed |
| hybrid     | 0.5            | 0.5            | Equal contribution |

**Default (fallback):** 0.6 text, 0.4 CLIP (text-favored)

---

## Components

### New: `src/retrieval/two_stage_retriever.py`

**Class: TwoStageRetriever**

```python
class TwoStageRetriever:
    """
    Combines text recall with CLIP reranking using query-adaptive fusion.
    
    Stage 1: TextRetriever gets top-K candidates (K=20 by default)
    Stage 2: CLIP scores those K, reranks using query-type-specific weights
    
    Attributes:
        text_retriever: TextRetriever instance for candidate generation
        clip_retriever: CLIPRetriever instance for visual scoring
        top_k: Final number of results to return (default: 5)
        candidate_k: Number of candidates from stage 1 (default: 20)
    """
    
    def __init__(self, top_k: int = 5, candidate_k: int = 20):
        self.text_retriever = TextRetriever(top_k=candidate_k)
        self.clip_retriever = CLIPRetriever(top_k=candidate_k)
        self.top_k = top_k
        self.candidate_k = candidate_k
    
    def retrieve(
        self, 
        query: str, 
        query_type: str = "factual"
    ) -> list[dict]:
        """
        Two-stage retrieval with query-adaptive fusion.
        
        Args:
            query: User query text
            query_type: One of [factual, visual, multi_hop, hybrid]
        
        Returns:
            Top-k documents after reranking, each with:
              - doc_id, film_id, title, content, modality
              - text_score, clip_score, fused_score
              - metadata (poster_path, still_paths, year, etc.)
        """
        # Stage 1: Get candidates via text
        candidates = self.text_retriever.retrieve(query)
        
        if not candidates:
            return []
        
        # Stage 2: CLIP rerank
        alpha, beta = self._get_weights(query_type)
        reranked = self._rerank_with_clip(query, candidates, alpha, beta)
        
        return reranked[:self.top_k]
    
    def _get_weights(self, query_type: str) -> tuple[float, float]:
        """Returns (alpha, beta) for text and CLIP weights."""
        weights = {
            "factual": (0.8, 0.2),
            "visual": (0.3, 0.7),
            "multi_hop": (0.6, 0.4),
            "hybrid": (0.5, 0.5),
        }
        return weights.get(query_type, (0.6, 0.4))
    
    def _rerank_with_clip(
        self, 
        query: str, 
        candidates: list[dict], 
        alpha: float, 
        beta: float
    ) -> list[dict]:
        """
        Rerank candidates using CLIP visual similarity.
        
        Process:
        1. Encode query with CLIP (text encoder) — done once
        2. For each candidate:
           - Extract image paths from metadata
           - If images exist: compute CLIP similarity, average across images
           - If no images: clip_score = 0 (fallback to text)
        3. Normalize text_score and clip_score to [0,1]
        4. Compute fused_score = alpha * norm(text_score) + beta * norm(clip_score)
        5. Sort by fused_score descending
        
        Args:
            query: User query text
            candidates: Results from stage 1 (text retrieval)
            alpha: Text weight
            beta: CLIP weight
        
        Returns:
            Reranked candidates with text_score, clip_score, fused_score fields
        """
        # Implementation details in code
```

**Key Methods:**
- `retrieve()` — main entry point
- `_get_weights()` — query-adaptive weight lookup
- `_rerank_with_clip()` — CLIP scoring + fusion logic
- `_normalize_scores()` — min-max normalization to [0,1]
- `_compute_clip_score()` — average CLIP similarity across images

**Error Handling:**
- Missing images → clip_score = 0, falls back to text ranking
- CLIP encoding failure → log warning, use text_score only
- Empty candidate set → return empty list (no crash)

---

### Modified: `src/agent/state.py`

**Add field to AgentState:**

```python
class AgentState(TypedDict):
    # ... existing fields ...
    variant: str  # NEW: "A", "B", "C", or "D" — controls which retrieval strategy
```

**Purpose:** Allows evaluation harness to select retrieval variant without changing agent code.

---

### Modified: `src/agent/nodes.py`

**Update RetrievalPlanner node:**

```python
def retrieval_planner_node(state: AgentState) -> dict:
    """
    Retrieval planner — selects and calls appropriate retriever.
    
    Variant-aware:
      - Variant A: no retrieval (plain LLM)
      - Variant B: hybrid RRF (text + CLIP fusion)
      - Variant C: text-only (current best)
      - Variant D: two-stage (text → CLIP reranking) ← NEW
    """
    query = state["query"]
    query_type = state.get("query_type", "factual")
    variant = state.get("variant", "C")  # Default to C if not specified
    
    if variant == "D":
        # Use two-stage retrieval
        retriever = TwoStageRetriever()
        results = retriever.retrieve(query=query, query_type=query_type)
    elif variant == "C":
        # Existing text-only logic
        # ...
    # ... other variants ...
```

---

### Modified: `src/evaluation/run_eval.py`

**Add Variant D evaluation:**

```python
def run_variant_d() -> dict:
    """
    Evaluate Variant D: Two-Stage Multimodal Retrieval.
    
    Same workflow as Variant C but with variant="D" in state,
    which triggers TwoStageRetriever in RetrievalPlanner.
    
    Returns:
        Evaluation results dict with per-query metrics
    """
    # Implementation: identical to run_variant_c() but set variant="D"
```

**Add Ablation 3: CLIP Weight Tuning**

```python
def run_ablation_3_clip_weights() -> dict:
    """
    Ablation Study 3: Impact of CLIP Reranking Weight.
    
    Test visual queries (Family 2) with varying CLIP weights:
      beta ∈ {0.0, 0.3, 0.5, 0.7, 0.9}
      alpha = 1.0 - beta
    
    Measures:
      - Recall@5 vs beta
      - Optimal beta for visual queries
    
    Returns:
        Dict mapping beta → recall results
    """
```

---

## Data Flow

### Score Fusion Algorithm

**Input:** 
- `candidates`: list of 20 docs from text retrieval, each with `text_score`
- `query`: user query string
- `alpha`, `beta`: fusion weights

**Process:**

1. **Encode query (once):**
   ```python
   query_embedding = clip_model.encode(query)  # CLIP text encoder
   ```

2. **For each candidate:**
   ```python
   image_paths = extract_images(candidate.metadata)
   
   if image_paths:
       clip_scores = []
       for img_path in image_paths:
           img = load_image(img_path)
           img_embedding = clip_model.encode(img)
           similarity = cosine_similarity(query_embedding, img_embedding)
           clip_scores.append(similarity)
       
       clip_score = mean(clip_scores)  # Average across poster + stills
   else:
       clip_score = 0.0  # No images → fallback to text
   
   candidate.clip_score = clip_score
   ```

3. **Normalize scores:**
   ```python
   text_scores = [c.text_score for c in candidates]
   clip_scores = [c.clip_score for c in candidates]
   
   norm_text = min_max_normalize(text_scores)
   norm_clip = min_max_normalize(clip_scores)
   ```

4. **Fuse and sort:**
   ```python
   for i, candidate in enumerate(candidates):
       candidate.fused_score = alpha * norm_text[i] + beta * norm_clip[i]
   
   candidates.sort(key=lambda c: c.fused_score, reverse=True)
   return candidates[:5]
   ```

**Complexity:**
- Stage 1: O(n) text retrieval (n = collection size)
- Stage 2: O(k × m) CLIP encoding (k=20 candidates, m≈3 images per film)
- Total: O(n + 60) ≈ O(n) since k, m are constants

**Estimated Latency:**
- Text retrieval: ~1.5s (current)
- CLIP encoding: ~500ms for 20 candidates × 3 images (CPU inference)
- Total: ~2s retrieval (vs 1.5s for text-only)

---

## Evaluation & Success Criteria

### Must Meet All Four:

1. **Recall@5 ≥ 38.5%** overall (match current Variant C)
2. **Visual recall ≥ 20%** (match current, ideally improve to 40%+)
3. **Factual recall ≥ 80%** (maintain text strength)
4. **Latency ≤ 25s** end-to-end (allow +7s buffer for CLIP overhead)

### Validation Workflow

```
┌─────────────────────────────────────────────┐
│ 1. Implement TwoStageRetriever              │
│ 2. Add unit tests (6 tests minimum)         │
│ 3. Integrate into RetrievalPlanner (Variant D) │
│ 4. Run evaluation: python run_eval.py --variant-d │
└─────────────────────────────────────────────┘
                    ↓
         ┌──────────────────────┐
         │ Check Success Criteria│
         └──────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ All 4 criteria met?   │
        └───────────────────────┘
         YES ↓              ↓ NO (but recall ≥ 35%)
    ┌────────────┐      ┌──────────────┐
    │ PROMOTE    │      │ KEEP VARIANT D│
    │ to Variant C│      │ separately   │
    │            │      │ Update RQ to │
    │ Fixes RQ   │      │ acknowledge  │
    │ mismatch   │      │ tradeoffs    │
    └────────────┘      └──────────────┘
                              ↓ NO (recall < 35%)
                        ┌──────────────┐
                        │ DON'T MERGE  │
                        │ Document in  │
                        │ failure      │
                        │ analysis     │
                        └──────────────┘
```

### If SUCCESS (≥38.5% recall):

**Actions:**
1. Rename current Variant C → "Variant C_text_baseline"
2. Make two-stage system the new Variant C
3. Update research question: "using both visual poster/scene embeddings (CLIP) and textual plot signals"
4. Update rubric mapping: "CLIP+memory angle" now accurate
5. Merge to main branch

**Report Impact:**
- Fixes RQ/implementation mismatch
- Strengthens novelty claim (genuinely multimodal)
- Provides comparison: text-only baseline vs multimodal

### If PARTIAL SUCCESS (35-38% recall):

**Actions:**
1. Keep as separate Variant D
2. Revise RQ: "We explored both text-only and multimodal retrieval; text-dominant hybrid performed best"
3. Document tradeoff: CLIP adds multimodal capability but slight recall cost

**Report Impact:**
- Honest about performance tradeoff
- Shows thorough exploration of design space
- Valuable negative result

### If FAILURE (<35% recall):

**Actions:**
1. Don't merge to main
2. Keep branch for analysis
3. Add detailed failure analysis to report
4. Keep current Variant C as-is

**Report Impact:**
- Acknowledge RQ limitation in discussion
- Explain why text-only outperformed multimodal
- Discuss future work: fine-tuning CLIP on film domain

---

## Testing Strategy

### Unit Tests: `tests/test_two_stage_retriever.py`

**Required tests (minimum 6):**

1. `test_returns_top_k()`
   - Verify returns exactly `top_k` results (default 5)

2. `test_query_adaptive_weights()`
   - Visual query → α=0.3, β=0.7
   - Factual query → α=0.8, β=0.2
   - Verify weight lookup correct

3. `test_score_fusion()`
   - Mock candidates with known text/CLIP scores
   - Verify fused score = α×text + β×clip

4. `test_handles_missing_images()`
   - Candidate with no poster/stills
   - Verify clip_score = 0, falls back to text ranking

5. `test_normalization()`
   - Input scores in different ranges
   - Verify normalized to [0,1]

6. `test_clip_improves_visual_ranking()`
   - Load real visual query from test suite (e.g., F2_01)
   - Verify CLIP moves visually-relevant films higher in top-5
   - This is the KEY integration test

### Integration Test: `tests/test_agent_integration.py`

**Add test:**

```python
def test_variant_d_two_stage_retrieval():
    """
    Variant D uses two-stage retrieval with query-adaptive fusion.
    
    Tests:
      - Visual query routes to high CLIP weight
      - Factual query routes to high text weight
      - Results contain both text_score and clip_score
      - Latency reasonable (<25s)
    """
```

### Manual Validation

**Test with sample queries:**
```bash
# Visual query — should use β=0.7
python src/agent/graph.py
> cold, desaturated, rain-soaked atmosphere
# Verify Blade Runner 2049 in top-5

# Factual query — should use β=0.2
> Who directed Mulholland Drive?
# Verify correct answer, text-dominated ranking
```

---

## File Structure

```
cineagent/
├── src/
│   ├── retrieval/
│   │   ├── two_stage_retriever.py       # NEW: TwoStageRetriever class
│   │   ├── text_retriever.py
│   │   ├── clip_retriever.py
│   │   ├── caption_retriever.py
│   │   └── hybrid_retriever.py
│   ├── agent/
│   │   ├── nodes.py                     # MODIFY: Add variant="D" handling
│   │   └── state.py                     # MODIFY: Add "variant" field
│   └── evaluation/
│       ├── run_eval.py                  # MODIFY: Add run_variant_d(), run_ablation_3()
│       └── test_suite.py                # No changes
├── tests/
│   ├── test_two_stage_retriever.py      # NEW: 6+ unit tests
│   └── test_agent_integration.py        # MODIFY: Add Variant D test
├── notebooks/
│   └── 05_ablation3_clip_weights.ipynb  # NEW: Weight tuning results
├── data/
│   └── results/
│       └── eval_results.json            # MODIFY: Add variant_D and ablation_3 sections
└── docs/
    ├── RESEARCH.md                       # MODIFY: Update with two-stage findings
    └── superpowers/specs/
        └── 2026-05-01-two-stage-multimodal-retrieval.md  # THIS FILE
```

---

## Implementation Timeline

**Estimated effort: 7-9 hours**

| Task | Estimated Time | Dependencies |
|------|----------------|--------------|
| 1. TwoStageRetriever class | 2-3 hours | None |
| 2. Unit tests | 1-2 hours | Task 1 |
| 3. Modify nodes.py, state.py | 1 hour | Task 1 |
| 4. Evaluation setup (run_variant_d, ablation_3) | 1 hour | Task 3 |
| 5. Run full evaluation | 30-45 min | Task 4 |
| 6. Analysis + decision | 1 hour | Task 5 |

**Critical path:** Tasks 1→2→3→4→5→6 (sequential)

---

## Git Workflow

```bash
# Initialize branch
git checkout -b two-stage-retrieval

# Commit 1: Core retriever
git add src/retrieval/two_stage_retriever.py
git commit -m "feat: add TwoStageRetriever with query-adaptive fusion"

# Commit 2: Tests
git add tests/test_two_stage_retriever.py
git commit -m "test: add TwoStageRetriever unit tests (6 tests)"

# Commit 3: Agent integration
git add src/agent/nodes.py src/agent/state.py
git commit -m "feat: integrate two-stage retrieval as Variant D"

# Commit 4: Evaluation
git add src/evaluation/run_eval.py
git commit -m "feat: add Variant D evaluation + Ablation 3"

# Validate
pytest tests/test_two_stage_retriever.py -v
python src/evaluation/run_eval.py --variant-d

# Decision point: check success criteria
# IF SUCCESS:
git checkout main
git merge two-stage-retrieval
git tag v2.0-multimodal-retrieval
git push origin main --tags

# IF FAILURE:
# Stay on branch, document findings, don't merge
```

---

## Rollback Plan

**If performance does not meet criteria:**

```bash
# Option 1: Return to main (clean slate)
git checkout main
git branch -D two-stage-retrieval

# Option 2: Keep branch for failure analysis
git checkout main
# Branch remains for documentation
# Add findings to RESEARCH.md failure analysis section
```

**No risk to main branch** — all work isolated until validated.

---

## Success Metrics Summary

| Metric | Current (Variant C) | Target (Variant D) | Stretch Goal |
|--------|---------------------|-------------------|--------------|
| Overall Recall@5 | 38.5% | ≥ 38.5% | > 45% |
| Visual Recall@5 | 20% | ≥ 20% | > 40% |
| Factual Recall@5 | 80% | ≥ 80% | Maintain |
| Latency (mean) | 17.5s | ≤ 25s | < 20s |

**Primary Goal:** Match or beat current performance while genuinely using multimodal retrieval.

**Secondary Goal:** Improve visual query performance (20% → 40%+) through CLIP reranking.

---

## Research Question Impact

### Current RQ (Inaccurate):
> "using both visual poster/scene embeddings and textual plot signals"

**Problem:** Implementation doesn't use visual embeddings.

### Updated RQ (If Two-Stage Succeeds):
> "When users express film preferences through natural language, does a multimodal agent with query-adaptive two-stage retrieval — using text embeddings for candidate generation and CLIP visual embeddings for reranking — and a dynamic taste profile outperform text-only RAG and plain LLM baselines on personalized recommendation accuracy?"

**Improvements:**
- Accurate description of implementation
- Highlights query-adaptive fusion (novel contribution)
- Maintains dynamic taste profile angle
- Clearly states what's being compared

### Alternative RQ (If Two-Stage Performs Similarly):
> "We explored both text-only and multimodal two-stage retrieval. Text-dominant fusion (80% text, 20% CLIP) achieved comparable performance to pure text (38.5% vs 38.5%), demonstrating that auto-generated captions provide sufficient visual signal for mood queries without true cross-modal embeddings."

**Value:**
- Honest about findings
- Negative result is still publishable
- Shows thorough design space exploration

---

## Novelty Claims (Updated)

### What Makes This Novel:

1. **Query-Adaptive Two-Stage Fusion**
   - Most systems use fixed retrieval strategy
   - We adapt CLIP weight based on query semantics
   - Visual queries get more CLIP influence; factual queries less

2. **Caption vs. CLIP Comparison**
   - Tests whether linguistic descriptions of images suffice
   - Or if direct visual embeddings add value
   - Empirical finding: captions competitive with CLIP for mood queries

3. **Dynamic Taste Profile**
   - Still a key contribution
   - Works alongside any retrieval strategy

4. **Comprehensive Retrieval Ablation**
   - 4 retrieval variants tested
   - 3 system variants compared
   - Provides clear design space map

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| CLIP doesn't improve recall | Medium | Medium | Keep as Variant D, document tradeoff |
| Latency exceeds 25s | Low | Low | Optimize CLIP encoding, reduce candidate_k |
| Missing images cause crashes | High | Low | Graceful fallback to text_score=1, clip_score=0 |
| Weight tuning is brittle | Low | Medium | Use Ablation 3 to validate weights empirically |
| Implementation takes >9 hours | Low | Medium | Scope creep — stick to spec, no extras |

---

## Open Questions

1. **Should we cache CLIP embeddings for film images?**
   - Pro: Faster stage 2 (no re-encoding)
   - Con: Storage overhead, complexity
   - Decision: No — candidate set changes per query, cache hit rate low

2. **Should we use image-to-image CLIP search directly?**
   - Pro: True cross-modal retrieval
   - Con: Requires query image (we have text queries)
   - Decision: No — stick to text query → CLIP text encoder

3. **Should we tune weights per query family in evaluation?**
   - Pro: Optimal performance per family
   - Con: Overfitting to test set
   - Decision: Use fixed weight map, validate with Ablation 3

---

## References

**Related Work:**
- Ablation 1 (Phase 2): Text-only 100%, CLIP-only 20% on visual queries
- ADR-005 (ARCHITECTURE.md): RRF fusion for hybrid retrieval
- Current Variant C: Text-only retrieval, 38.5% overall recall

**Design Decisions:**
- Two-stage > single-stage: preserves text recall, adds CLIP signal
- Query-adaptive > fixed fusion: empirically motivated by Ablation 1
- 20 candidates > 50 or 100: balance between recall and CLIP latency

---

## Appendix: Example Outputs

### Visual Query Example

**Input:** "cold, desaturated, rain-soaked atmosphere"  
**Query Type:** visual  
**Weights:** α=0.3 (text), β=0.7 (CLIP)

**Stage 1 (Text Retrieval, top-20):**
```json
[
  {"film_id": "335984", "title": "Blade Runner 2049", "text_score": 0.92},
  {"film_id": "78", "title": "Blade Runner", "text_score": 0.89},
  {"film_id": "424", "title": "Se7en", "text_score": 0.85},
  ...
]
```

**Stage 2 (CLIP Reranking, top-5):**
```json
[
  {
    "film_id": "335984",
    "title": "Blade Runner 2049",
    "text_score": 0.92,
    "clip_score": 0.94,  // High visual match
    "fused_score": 0.934  // 0.3*0.92 + 0.7*0.94 = 0.934
  },
  {
    "film_id": "78",
    "title": "Blade Runner",
    "text_score": 0.89,
    "clip_score": 0.91,
    "fused_score": 0.904
  },
  ...
]
```

**Outcome:** CLIP boosts visually-matching films even if text score slightly lower.

### Factual Query Example

**Input:** "Who directed Mulholland Drive?"  
**Query Type:** factual  
**Weights:** α=0.8 (text), β=0.2 (CLIP)

**Stage 2 (CLIP Reranking, top-5):**
```json
[
  {
    "film_id": "1018",
    "title": "Mulholland Drive",
    "text_score": 0.98,  // Perfect text match
    "clip_score": 0.45,  // Low visual relevance (irrelevant)
    "fused_score": 0.874  // 0.8*0.98 + 0.2*0.45 = 0.874
  },
  ...
]
```

**Outcome:** Text dominates, CLIP has minimal influence (correct behavior).

---

## Sign-Off

**Approved by:** User (2026-05-01)  
**Ready for Implementation:** Yes  
**Next Step:** Invoke `writing-plans` skill to create detailed implementation plan.
