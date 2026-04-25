# Phase 2 Retrieval Layer Testing & Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate 4 retrieval variants against real KB (523 films) and prove hybrid RRF outperforms single-modality retrieval on visual queries.

**Architecture:** Integration tests verify retrievers work with real ChromaDB collections. Ablation notebook compares all 4 variants on ground truth queries (Families 1 & 2) using Recall@5 metric.

**Tech Stack:** pytest (testing), Jupyter (ablation analysis), ChromaDB (vector DB), sentence-transformers (embeddings)

---

## File Structure

**Files to create:**
- `tests/test_retrieval_integration.py` - Integration tests for all 4 retrievers
- `notebooks/02_retrieval_ablation.ipynb` - Ablation analysis notebook

**Files to modify:**
- `src/retrieval/text_retriever.py` - Add usage example in docstring
- `src/retrieval/caption_retriever.py` - Add usage example in docstring
- `src/retrieval/clip_retriever.py` - Add usage example in docstring
- `src/retrieval/hybrid_retriever.py` - Add usage example in docstring
- `docs/RESEARCH.md` - Mark Phase 2 complete

---

## Task 1: Integration Test - Text Retriever

**Files:**
- Create: `tests/test_retrieval_integration.py`

- [ ] **Step 1: Write failing test for text retriever factual query**

```python
"""
Integration tests for retrieval layer with real KB.

Verifies all 4 retrievers work with the actual ChromaDB collections (523 films).
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.text_retriever import TextRetriever


class TestTextRetrieverIntegration:
    """Integration tests for TextRetriever with real KB."""
    
    @pytest.fixture(scope="class")
    def text_retriever(self):
        """Initialize text retriever once for all tests."""
        return TextRetriever(top_k=5)
    
    def test_factual_query_retrieves_correct_film(self, text_retriever):
        """Test: Factual query returns correct film in top-5."""
        # Mulholland Drive (film_id: 1018) should be in top-5 for director query
        query = "Who directed Mulholland Drive?"
        results = text_retriever.retrieve(query)
        
        # Check we got results
        assert len(results) > 0, "Text retriever returned no results"
        assert len(results) <= 5, "Text retriever returned more than top-5"
        
        # Check Mulholland Drive is in top-5
        film_ids = [r["film_id"] for r in results]
        assert "1018" in film_ids, (
            f"Mulholland Drive (1018) not in top-5 results. "
            f"Got film_ids: {film_ids}"
        )
        
        # Check result format
        first_result = results[0]
        assert "doc_id" in first_result
        assert "film_id" in first_result
        assert "title" in first_result
        assert "score" in first_result
        assert "content" in first_result
        assert isinstance(first_result["score"], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval_integration.py::TestTextRetrieverIntegration::test_factual_query_retrieves_correct_film -v`

Expected: FAIL with "No module named 'retrieval'" or passes if KB already works

- [ ] **Step 3: Fix imports if needed, run test**

If test fails due to imports, the fixture and imports are correct. Test should PASS if KB is properly built.

Run: `pytest tests/test_retrieval_integration.py::TestTextRetrieverIntegration::test_factual_query_retrieves_correct_film -v`

Expected: PASS (retrieval code already exists and works)

- [ ] **Step 4: Commit**

```bash
git add tests/test_retrieval_integration.py
git commit -m "test: add integration test for text retriever factual queries

Verifies text retriever returns Mulholland Drive for director query.
Tests against real KB (523 films)."
```

---

## Task 2: Integration Test - CLIP Retriever

**Files:**
- Modify: `tests/test_retrieval_integration.py`

- [ ] **Step 1: Write failing test for CLIP retriever visual query**

```python
from retrieval.clip_retriever import CLIPRetriever


class TestCLIPRetrieverIntegration:
    """Integration tests for CLIPRetriever with real KB."""
    
    @pytest.fixture(scope="class")
    def clip_retriever(self):
        """Initialize CLIP retriever once for all tests."""
        return CLIPRetriever(top_k=5)
    
    def test_visual_query_retrieves_correct_film(self, clip_retriever):
        """Test: Visual/mood query returns correct film in top-5."""
        # Blade Runner 2049 (film_id: 335984) should be in top-5 for cold rainy atmosphere
        query = "cold, desaturated, rain-soaked visual atmosphere"
        results = clip_retriever.retrieve(query)
        
        # Check we got results
        assert len(results) > 0, "CLIP retriever returned no results"
        assert len(results) <= 5, "CLIP retriever returned more than top-5"
        
        # Check Blade Runner 2049 is in top-5
        film_ids = [r["film_id"] for r in results]
        assert "335984" in film_ids, (
            f"Blade Runner 2049 (335984) not in top-5 results. "
            f"Got film_ids: {film_ids}, titles: {[r['title'] for r in results]}"
        )
        
        # Check result format includes image_path
        first_result = results[0]
        assert "doc_id" in first_result
        assert "film_id" in first_result
        assert "title" in first_result
        assert "score" in first_result
        assert isinstance(first_result["score"], float)
```

- [ ] **Step 2: Run test to verify behavior**

Run: `pytest tests/test_retrieval_integration.py::TestCLIPRetrieverIntegration::test_visual_query_retrieves_correct_film -v`

Expected: PASS if CLIP retriever works correctly with KB

Note: If test fails (Blade Runner 2049 not in top-5), this is a valid finding. Document in test output and continue - the test is correct, retrieval may need tuning.

- [ ] **Step 3: Commit**

```bash
git add tests/test_retrieval_integration.py
git commit -m "test: add integration test for CLIP retriever visual queries

Verifies CLIP retriever returns Blade Runner 2049 for mood query.
Tests cross-modal retrieval (text query -> image results)."
```

---

## Task 3: Integration Test - Caption Retriever

**Files:**
- Modify: `tests/test_retrieval_integration.py`

- [ ] **Step 1: Write test for caption retriever**

```python
from retrieval.caption_retriever import CaptionRetriever


class TestCaptionRetrieverIntegration:
    """Integration tests for CaptionRetriever with real KB."""
    
    @pytest.fixture(scope="class")
    def caption_retriever(self):
        """Initialize caption retriever once for all tests."""
        return CaptionRetriever(top_k=5)
    
    def test_visual_query_via_captions(self, caption_retriever):
        """Test: Visual query retrieves films via auto-generated captions."""
        # Should retrieve films with neon-lit nightscape imagery
        query = "neon-lit urban nightscape, purple and green palette"
        results = caption_retriever.retrieve(query)
        
        # Check we got results
        assert len(results) > 0, "Caption retriever returned no results"
        assert len(results) <= 5, "Caption retriever returned more than top-5"
        
        # Check result format and that we got caption docs
        first_result = results[0]
        assert "doc_id" in first_result
        assert "film_id" in first_result
        assert "title" in first_result
        assert "score" in first_result
        assert "content" in first_result
        assert isinstance(first_result["score"], float)
        
        # Verify we got caption-type documents (not plot docs)
        metadata = first_result.get("metadata", {})
        doc_type = metadata.get("doc_type", "")
        assert "caption" in doc_type, (
            f"Expected caption document, got doc_type: {doc_type}"
        )
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_retrieval_integration.py::TestCaptionRetrieverIntegration::test_visual_query_via_captions -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_retrieval_integration.py
git commit -m "test: add integration test for caption retriever

Verifies caption retriever searches Gemini-generated captions.
Tests secondary visual retrieval pathway (text -> captions)."
```

---

## Task 4: Integration Test - Hybrid Retriever

**Files:**
- Modify: `tests/test_retrieval_integration.py`

- [ ] **Step 1: Write test for hybrid RRF fusion**

```python
from retrieval.hybrid_retriever import HybridRetriever


class TestHybridRetrieverIntegration:
    """Integration tests for HybridRetriever with real KB."""
    
    @pytest.fixture(scope="class")
    def hybrid_retriever(self):
        """Initialize hybrid retriever once for all tests."""
        return HybridRetriever(top_k=5)
    
    def test_hybrid_combines_multiple_sources(self, hybrid_retriever):
        """Test: Hybrid retriever fuses results from multiple modalities."""
        # Query that benefits from both text and visual retrieval
        query = "psychological thriller with twist ending"
        results = hybrid_retriever.retrieve(query)
        
        # Check we got results
        assert len(results) > 0, "Hybrid retriever returned no results"
        assert len(results) <= 5, "Hybrid retriever returned more than top-5"
        
        # Check result format includes RRF score
        first_result = results[0]
        assert "doc_id" in first_result
        assert "film_id" in first_result
        assert "title" in first_result
        assert "score" in first_result  # This should be RRF score
        assert isinstance(first_result["score"], float)
        
        # Hybrid should return film-level results (deduplicated across sources)
        # Check that film_ids are unique (no duplicate films in top-5)
        film_ids = [r["film_id"] for r in results]
        unique_film_ids = list(set(film_ids))
        assert len(film_ids) == len(unique_film_ids), (
            f"Hybrid retriever returned duplicate films: {film_ids}"
        )
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_retrieval_integration.py::TestHybridRetrieverIntegration::test_hybrid_combines_multiple_sources -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_retrieval_integration.py
git commit -m "test: add integration test for hybrid RRF retriever

Verifies hybrid retriever fuses text + CLIP + caption results.
Tests RRF fusion and film-level deduplication."
```

---

## Task 5: Integration Test - Latency Check

**Files:**
- Modify: `tests/test_retrieval_integration.py`

- [ ] **Step 1: Write latency test for all retrievers**

```python
import time


class TestRetrieverPerformance:
    """Performance tests for all retrievers."""
    
    def test_retriever_latency_under_2_seconds(self):
        """Test: All retrievers complete queries in < 2 seconds."""
        retrievers = {
            "text": TextRetriever(top_k=5),
            "caption": CaptionRetriever(top_k=5),
            "clip": CLIPRetriever(top_k=5),
            "hybrid": HybridRetriever(top_k=5),
        }
        
        test_queries = [
            "Who directed Mulholland Drive?",
            "cold rainy atmosphere",
            "neon-lit nightscape",
        ]
        
        for retriever_name, retriever in retrievers.items():
            latencies = []
            
            for query in test_queries:
                start = time.perf_counter()
                results = retriever.retrieve(query)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                
                assert len(results) > 0, (
                    f"{retriever_name} returned no results for: {query}"
                )
            
            mean_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            assert max_latency < 2.0, (
                f"{retriever_name} retriever too slow: "
                f"max={max_latency:.3f}s, mean={mean_latency:.3f}s"
            )
            
            print(f"{retriever_name}: mean={mean_latency:.3f}s, max={max_latency:.3f}s")
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_retrieval_integration.py::TestRetrieverPerformance::test_retriever_latency_under_2_seconds -v -s`

Expected: PASS with printed latency stats

- [ ] **Step 3: Commit**

```bash
git add tests/test_retrieval_integration.py
git commit -m "test: add latency performance tests for all retrievers

Verifies all 4 retrievers complete queries in < 2 seconds.
Measures mean and max latency across 3 test queries."
```

---

## Task 6: Run Integration Test Suite

**Files:**
- None (verification step)

- [ ] **Step 1: Run all integration tests**

Run: `pytest tests/test_retrieval_integration.py -v`

Expected: All tests PASS

Output should show:
```
test_retrieval_integration.py::TestTextRetrieverIntegration::test_factual_query_retrieves_correct_film PASSED
test_retrieval_integration.py::TestCLIPRetrieverIntegration::test_visual_query_retrieves_correct_film PASSED
test_retrieval_integration.py::TestCaptionRetrieverIntegration::test_visual_query_via_captions PASSED
test_retrieval_integration.py::TestHybridRetrieverIntegration::test_hybrid_combines_multiple_sources PASSED
test_retrieval_integration.py::TestRetrieverPerformance::test_retriever_latency_under_2_seconds PASSED
```

- [ ] **Step 2: Document any failures**

If any tests fail, note which ones and why. Common failure modes:
- Ground truth film not in top-5 (valid finding - document for ablation)
- Latency > 2s (may need optimization or hardware-dependent)
- Connection errors (check ChromaDB path, collections exist)

No code changes needed for this step - just verification.

---

## Task 7: Create Ablation Notebook - Setup

**Files:**
- Create: `notebooks/02_retrieval_ablation.ipynb`

- [ ] **Step 1: Create notebook with imports and setup cell**

Create `notebooks/02_retrieval_ablation.ipynb` with this first cell:

```python
# Cell 1: Setup and Imports

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd().parent / "src"))

from retrieval.text_retriever import TextRetriever
from retrieval.caption_retriever import CaptionRetriever
from retrieval.clip_retriever import CLIPRetriever
from retrieval.hybrid_retriever import HybridRetriever

import pandas as pd
from typing import Dict, List

# Initialize all 4 retrievers
print("Initializing retrievers...")
text_ret = TextRetriever(top_k=5)
caption_ret = CaptionRetriever(top_k=5)
clip_ret = CLIPRetriever(top_k=5)
hybrid_ret = HybridRetriever(top_k=5)
print("✓ All retrievers loaded")
```

- [ ] **Step 2: Add markdown cell with notebook purpose**

Add markdown cell at top:

```markdown
# Retrieval Ablation Analysis - Phase 2

**Goal:** Compare all 4 retrieval variants on ground truth queries to validate hypothesis.

**Hypothesis:** Hybrid RRF > CLIP-only > Caption-only > Text-only on visual queries (Family 2)

**Method:** Run Families 1 & 2 queries (10 total), compute Recall@5 for each variant.

**Expected runtime:** ~2 minutes
```

- [ ] **Step 3: Run setup cell to verify imports work**

Run the setup cell in Jupyter. Expected output:
```
Initializing retrievers...
✓ All retrievers loaded
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/02_retrieval_ablation.ipynb
git commit -m "feat: create ablation notebook with setup cell

Added notebook 02_retrieval_ablation.ipynb with:
- Imports for all 4 retrievers
- Initialization code
- Purpose documentation"
```

---

## Task 8: Ablation Notebook - Ground Truth Queries

**Files:**
- Modify: `notebooks/02_retrieval_ablation.ipynb`

- [ ] **Step 1: Add ground truth queries cell**

Add new code cell:

```python
# Cell 2: Ground Truth Queries

# Family 1: Factual queries (5 queries)
# These should work well with text retrieval (plot/metadata search)
family1 = {
    "F1_01": {
        "query": "Who directed Mulholland Drive?",
        "correct_film_id": "1018",  # Mulholland Drive
        "category": "director"
    },
    "F1_02": {
        "query": "What year was Parasite released?",
        "correct_film_id": "496243",  # Parasite
        "category": "year"
    },
    "F1_03": {
        "query": "What genre is Oldboy?",
        "correct_film_id": "670",  # Oldboy
        "category": "genre"
    },
    "F1_04": {
        "query": "Who stars in No Country for Old Men?",
        "correct_film_id": "6977",  # No Country for Old Men
        "category": "cast"
    },
    "F1_05": {
        "query": "How long is 2001: A Space Odyssey?",
        "correct_film_id": "62",  # 2001: A Space Odyssey
        "category": "runtime"
    }
}

# Family 2: Visual/mood queries (5 queries)
# These should work better with CLIP/caption retrieval (visual content search)
family2 = {
    "F2_01": {
        "query": "cold, desaturated, rain-soaked visual atmosphere",
        "correct_film_id": "335984",  # Blade Runner 2049
        "category": "mood"
    },
    "F2_02": {
        "query": "warm, golden, dusty western landscape",
        "correct_film_id": "7345",  # There Will Be Blood
        "category": "mood"
    },
    "F2_03": {
        "query": "clinical white sterile environments, institutional feel",
        "correct_film_id": "510",  # One Flew Over the Cuckoo's Nest
        "category": "mood"
    },
    "F2_04": {
        "query": "neon-lit urban nightscape, purple and green palette",
        "correct_film_id": "1538",  # Collateral
        "category": "color"
    },
    "F2_05": {
        "query": "foggy grey post-industrial wasteland",
        "correct_film_id": "9693",  # Children of Men
        "category": "mood"
    }
}

all_queries = {**family1, **family2}

print(f"Family 1 (Factual): {len(family1)} queries")
print(f"Family 2 (Visual): {len(family2)} queries")
print(f"Total queries: {len(all_queries)}")
```

- [ ] **Step 2: Run cell to verify ground truth is defined**

Run the cell. Expected output:
```
Family 1 (Factual): 5 queries
Family 2 (Visual): 5 queries
Total queries: 10
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_retrieval_ablation.ipynb
git commit -m "feat: add ground truth queries to ablation notebook

Added 10 ground truth queries:
- Family 1 (factual): 5 queries for metadata/plot search
- Family 2 (visual): 5 queries for mood/aesthetic search"
```

---

## Task 9: Ablation Notebook - Evaluation Functions

**Files:**
- Modify: `notebooks/02_retrieval_ablation.ipynb`

- [ ] **Step 1: Add Recall@5 evaluation function**

Add new code cell:

```python
# Cell 3: Evaluation Functions

def recall_at_k(results: List[Dict], correct_film_id: str, k: int = 5) -> bool:
    """
    Check if correct film appears in top-k results.
    
    Args:
        results: List of retrieval results (dicts with 'film_id')
        correct_film_id: Expected film ID
        k: Top-k cutoff (default 5)
    
    Returns:
        True if correct film in top-k, False otherwise
    """
    top_k_film_ids = [r["film_id"] for r in results[:k]]
    return correct_film_id in top_k_film_ids


def evaluate_retriever(retriever, queries: Dict, k: int = 5) -> Dict:
    """
    Run all queries through a retriever and compute Recall@k.
    
    Args:
        retriever: Retriever instance (must have .retrieve() method)
        queries: Dict of {query_id: {query, correct_film_id, ...}}
        k: Top-k for recall calculation
    
    Returns:
        Dict with recall score, hits, total, and per-query log
    """
    hits = 0
    results_log = []
    
    for qid, qdata in queries.items():
        # Run retrieval
        results = retriever.retrieve(qdata["query"])
        
        # Check if correct film in top-k
        hit = recall_at_k(results, qdata["correct_film_id"], k)
        hits += hit
        
        # Log result
        results_log.append({
            "query_id": qid,
            "query": qdata["query"],
            "correct_film_id": qdata["correct_film_id"],
            "hit": hit,
            "top_result_title": results[0]["title"] if results else None,
            "top_result_film_id": results[0]["film_id"] if results else None,
            "top_result_score": results[0]["score"] if results else None
        })
    
    recall = hits / len(queries) if len(queries) > 0 else 0.0
    
    return {
        "recall_at_k": recall,
        "hits": hits,
        "total": len(queries),
        "log": results_log
    }


print("✓ Evaluation functions defined")
print(f"  - recall_at_k: Check if correct film in top-{5}")
print(f"  - evaluate_retriever: Run all queries, compute Recall@{5}")
```

- [ ] **Step 2: Run cell to verify functions are defined**

Run the cell. Expected output:
```
✓ Evaluation functions defined
  - recall_at_k: Check if correct film in top-5
  - evaluate_retriever: Run all queries, compute Recall@5
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_retrieval_ablation.ipynb
git commit -m "feat: add Recall@5 evaluation functions to ablation notebook

Added evaluation logic:
- recall_at_k: check if correct film in top-k
- evaluate_retriever: run all queries, compute aggregate Recall@5"
```

---

## Task 10: Ablation Notebook - Run Ablation

**Files:**
- Modify: `notebooks/02_retrieval_ablation.ipynb`

- [ ] **Step 1: Add ablation execution cell**

Add new code cell:

```python
# Cell 4: Run Ablation

print("Running ablation on all 4 retrieval variants...")
print("=" * 60)

# Run all 4 variants on both query families
results = {}

for variant_name, retriever in [
    ("Text-only", text_ret),
    ("Caption-only", caption_ret),
    ("CLIP-only", clip_ret),
    ("Hybrid RRF", hybrid_ret)
]:
    print(f"\n{variant_name}:")
    
    # Evaluate on Family 1 (factual)
    f1_results = evaluate_retriever(retriever, family1, k=5)
    print(f"  Family 1 (Factual): {f1_results['recall_at_k']:.2f} ({f1_results['hits']}/{f1_results['total']})")
    
    # Evaluate on Family 2 (visual)
    f2_results = evaluate_retriever(retriever, family2, k=5)
    print(f"  Family 2 (Visual):  {f2_results['recall_at_k']:.2f} ({f2_results['hits']}/{f2_results['total']})")
    
    # Store results
    results[variant_name] = {
        "Family 1": f1_results,
        "Family 2": f2_results
    }

print("\n" + "=" * 60)
print("✓ Ablation complete")
```

- [ ] **Step 2: Run cell and observe results**

Run the cell. Expected output format:
```
Running ablation on all 4 retrieval variants...
============================================================

Text-only:
  Family 1 (Factual): 0.80 (4/5)
  Family 2 (Visual):  0.20 (1/5)

Caption-only:
  Family 1 (Factual): 0.60 (3/5)
  Family 2 (Visual):  0.60 (3/5)

CLIP-only:
  Family 1 (Factual): 0.40 (2/5)
  Family 2 (Visual):  0.80 (4/5)

Hybrid RRF:
  Family 1 (Factual): 0.80 (4/5)
  Family 2 (Visual):  1.00 (5/5)

============================================================
✓ Ablation complete
```

Note: Actual numbers will vary based on real KB. This is example output.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_retrieval_ablation.ipynb
git commit -m "feat: add ablation execution cell to notebook

Runs all 4 retrieval variants on Families 1 & 2.
Prints Recall@5 results for each variant."
```

---

## Task 11: Ablation Notebook - Results Table & Analysis

**Files:**
- Modify: `notebooks/02_retrieval_ablation.ipynb`

- [ ] **Step 1: Add results table generation cell**

Add new code cell:

```python
# Cell 5: Generate Results Table

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
print("\nAblation 1 Results - Recall@5")
print("=" * 60)
print(df.to_string(index=False))
print("=" * 60)

# Export for later use
df.to_csv("../data/results/ablation1_recall_at_5.csv", index=False)
print("\n✓ Saved results to data/results/ablation1_recall_at_5.csv")
```

- [ ] **Step 2: Add analysis markdown cell**

Add markdown cell after the results table:

```markdown
## Ablation 1 Results - Analysis

### Hypothesis Validation

**Hypothesis:** Hybrid RRF > CLIP-only > Caption-only > Text-only on Family 2 (visual queries)

**Results:**
- ✅/❌ Hybrid RRF achieved highest recall on Family 2: [X.XX]
- ✅/❌ CLIP-only outperformed Text-only on Family 2: [X.XX vs Y.YY]
- ✅/❌ Text-only outperformed CLIP-only on Family 1: [X.XX vs Y.YY]

### Key Findings

1. **Text-only retrieval fails on visual queries** (Family 2)
   - Recall@5: [X.XX] on Family 2
   - Why: Plot summaries don't describe visual mood/atmosphere

2. **CLIP-only retrieval excels on visual queries** (Family 2)
   - Recall@5: [X.XX] on Family 2
   - Why: CLIP embeddings encode visual content directly

3. **Hybrid RRF combines strengths of both modalities**
   - Recall@5: [X.XX] on Family 1, [X.XX] on Family 2
   - Why: RRF fusion rewards films ranking well across multiple sources

4. **Caption-only retrieval provides secondary visual signal**
   - Performance between Text-only and CLIP-only
   - Captions describe visual content but miss nuance

### Implications for Final System

- **Use Hybrid RRF in agent** (Phase 3) for best overall performance
- **Visual queries are the key differentiation** - this proves multimodal retrieval is necessary
- **Text + CLIP fusion is validated** by higher hybrid recall on both families

### Failure Modes Observed

[Document any unexpected results here]
- Which specific queries failed?
- Which films were incorrectly retrieved?
- Any patterns in failures? (e.g., minimalist posters, ambiguous captions)
```

- [ ] **Step 3: Run cells and fill in actual numbers**

Run the results table cell. Then update the markdown cell with actual numbers from the table.

- [ ] **Step 4: Commit**

```bash
git add notebooks/02_retrieval_ablation.ipynb
git commit -m "feat: add results table and analysis to ablation notebook

Added:
- Pandas DataFrame results table
- Export to CSV for report
- Hypothesis validation analysis
- Key findings documentation"
```

---

## Task 12: Update Retriever Docstrings

**Files:**
- Modify: `src/retrieval/text_retriever.py`
- Modify: `src/retrieval/caption_retriever.py`
- Modify: `src/retrieval/clip_retriever.py`
- Modify: `src/retrieval/hybrid_retriever.py`

- [ ] **Step 1: Add usage example to text_retriever.py docstring**

At the top of `src/retrieval/text_retriever.py`, update the module docstring:

```python
"""
Text Retriever — Ablation 1 Variant: text-only

Searches the ChromaDB text collection using MiniLM dense embeddings.
This is the baseline retrieval variant — no image modality, no captions.

Used in:
  - Ablation 1: text-only vs caption-only vs CLIP-only vs hybrid
  - Variant B (fixed RAG): as the text component
  - Variant C (full agent): as one tool option for the RetrievalPlanner

Usage:
    from retrieval.text_retriever import TextRetriever
    
    retriever = TextRetriever(top_k=5)
    results = retriever.retrieve("Who directed Mulholland Drive?")
    
    for result in results:
        print(f"{result['title']}: {result['score']:.3f}")
        print(f"  {result['content'][:100]}...")

Output format:
    [
        {
            "doc_id": "1018_plot",
            "film_id": "1018",
            "title": "Mulholland Drive",
            "modality": "text",
            "content": "Mulholland Drive (2001). Directed by David Lynch...",
            "score": 0.8234,
            "metadata": {"year": 2001, "directors": "David Lynch", ...}
        },
        ...
    ]
"""
```

- [ ] **Step 2: Add usage example to caption_retriever.py docstring**

Update module docstring in `src/retrieval/caption_retriever.py`:

```python
"""
Caption Retriever — Ablation 1 Variant: caption-only

Searches auto-generated image captions using MiniLM dense embeddings.
This is a secondary visual retrieval pathway — text descriptions of images.

Usage:
    from retrieval.caption_retriever import CaptionRetriever
    
    retriever = CaptionRetriever(top_k=5)
    results = retriever.retrieve("neon-lit urban nightscape")
    
    for result in results:
        print(f"{result['title']}: {result['score']:.3f}")
        print(f"  Caption: {result['content']}")
"""
```

- [ ] **Step 3: Add usage example to clip_retriever.py docstring**

Update module docstring in `src/retrieval/clip_retriever.py`:

```python
"""
CLIP Retriever — Ablation 1 Variant: CLIP-only

Searches posters and scene stills using CLIP text-image embeddings.
This is the primary visual retrieval pathway — direct image search.

Usage:
    from retrieval.clip_retriever import CLIPRetriever
    
    retriever = CLIPRetriever(top_k=5)
    results = retriever.retrieve("cold, rain-soaked atmosphere")
    
    for result in results:
        print(f"{result['title']}: {result['score']:.3f}")
        print(f"  Image: {result['doc_id']}")  # e.g., "335984_still_2"
"""
```

- [ ] **Step 4: Add usage example to hybrid_retriever.py docstring**

Update module docstring in `src/retrieval/hybrid_retriever.py`:

```python
"""
Hybrid Retriever — Ablation 1 Variant: hybrid RRF (Final System)

Fuses results from text, caption, and CLIP retrievers using
Reciprocal Rank Fusion (RRF). This is the expected top-performing
retrieval variant.

RRF Formula:
  RRF_score(d) = Σ_i 1 / (k + rank_i(d))
  where k=60 (standard constant), rank_i(d) = position of doc d in list i

Films that rank well across multiple modalities get boosted.
Films that only rank well in one list get partial credit.

Usage:
    from retrieval.hybrid_retriever import HybridRetriever
    
    retriever = HybridRetriever(top_k=5)
    results = retriever.retrieve("psychological thriller cold atmosphere")
    
    for result in results:
        print(f"{result['title']}: RRF score={result['score']:.3f}")

See ARCHITECTURE.md ADR-005 for design rationale.
"""
```

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/text_retriever.py src/retrieval/caption_retriever.py src/retrieval/clip_retriever.py src/retrieval/hybrid_retriever.py
git commit -m "docs: add usage examples to all retriever docstrings

Added code examples and output format documentation for:
- TextRetriever
- CaptionRetriever
- CLIPRetriever
- HybridRetriever"
```

---

## Task 13: Update RESEARCH.md

**Files:**
- Modify: `docs/RESEARCH.md`

- [ ] **Step 1: Mark Phase 2 tasks complete**

In `docs/RESEARCH.md`, find the "Phase 2 — Retrieval Layer" section and update it:

```markdown
### Phase 2 — Retrieval Layer ✓ COMPLETE
- [x] Integration tests with real KB (523 films)
- [x] text_retriever.py verified (factual queries)
- [x] clip_retriever.py verified (visual queries)
- [x] caption_retriever.py verified (caption search)
- [x] hybrid_retriever.py verified (RRF fusion)
- [x] Latency tests (all < 2s per query)
- [x] Ablation 1: text vs caption vs CLIP vs hybrid
- [x] Notebook 02: Recall@5 comparison table
- [x] All 4 variants independently testable
- [x] Hypothesis validation documented
- [x] Usage examples in docstrings

**Ablation 1 Results:** See `notebooks/02_retrieval_ablation.ipynb`
**Integration Tests:** `pytest tests/test_retrieval_integration.py -v`
```

- [ ] **Step 2: Add ablation results summary**

After the Phase 2 section, add a results summary section:

```markdown
#### Ablation 1 Results Summary

**Hypothesis:** Hybrid RRF > CLIP-only > Caption-only > Text-only on visual queries

| Variant | Family 1 (Factual) | Family 2 (Visual) | Overall |
|---------|-------------------|------------------|---------|
| Text-only | [X.XX] | [X.XX] | [X.XX] |
| Caption-only | [X.XX] | [X.XX] | [X.XX] |
| CLIP-only | [X.XX] | [X.XX] | [X.XX] |
| Hybrid RRF | [X.XX] | [X.XX] | [X.XX] |

**Key Finding:** [Hybrid/CLIP/Caption/Text] achieved highest Recall@5 on Family 2 (visual queries), confirming [hypothesis was validated / refuted].

**Failure Modes Identified:**
- [List any systematic failures observed]
- [E.g., "Minimalist posters underperform in CLIP retrieval"]

**Next Steps:** Use Hybrid RRF in Phase 3 agent implementation.
```

- [ ] **Step 3: Fill in actual ablation results**

Copy the Recall@5 numbers from the notebook into the RESEARCH.md table.

- [ ] **Step 4: Commit**

```bash
git add docs/RESEARCH.md
git commit -m "docs: mark Phase 2 complete in RESEARCH.md

Updated Phase 2 checklist with all completed tasks.
Added Ablation 1 results summary table.
Documented hypothesis validation outcome."
```

---

## Task 14: Final Verification & Documentation

**Files:**
- None (verification only)

- [ ] **Step 1: Run full integration test suite**

Run: `pytest tests/test_retrieval_integration.py -v`

Expected: All 5 test classes pass (Text, CLIP, Caption, Hybrid, Performance)

- [ ] **Step 2: Run ablation notebook end-to-end**

Run: `jupyter nbconvert --to notebook --execute notebooks/02_retrieval_ablation.ipynb --output 02_retrieval_ablation_executed.ipynb`

Expected: Notebook executes without errors, generates results table

- [ ] **Step 3: Verify results files exist**

Check:
```bash
ls -lh data/results/ablation1_recall_at_5.csv
ls -lh notebooks/02_retrieval_ablation.ipynb
ls -lh tests/test_retrieval_integration.py
```

All files should exist.

- [ ] **Step 4: Create final commit**

```bash
git add -A
git commit -m "feat: Phase 2 complete - retrieval layer validated

Completed:
- 5 integration tests (text, CLIP, caption, hybrid, latency)
- Ablation notebook with Recall@5 comparison
- Ground truth queries (Families 1 & 2)
- Usage documentation for all retrievers
- RESEARCH.md Phase 2 marked complete

Results:
- All integration tests passing
- Ablation 1 hypothesis [validated/refuted]
- Ready for Phase 3 (LangGraph agent)"
```

---

## Success Criteria

Phase 2 is complete when:

✅ Integration test suite exists (`tests/test_retrieval_integration.py`)  
✅ All 5 integration tests pass  
✅ All 4 retrievers work with real KB (523 films)  
✅ Ablation notebook runs end-to-end (`notebooks/02_retrieval_ablation.ipynb`)  
✅ Recall@5 comparison table generated  
✅ Hypothesis validated or refuted (documented in notebook + RESEARCH.md)  
✅ Usage examples in all retriever docstrings  
✅ RESEARCH.md Phase 2 marked complete  
✅ Results saved to `data/results/ablation1_recall_at_5.csv`  

**Validation commands:**
```bash
# Run integration tests
pytest tests/test_retrieval_integration.py -v

# Execute ablation notebook
jupyter nbconvert --to notebook --execute notebooks/02_retrieval_ablation.ipynb

# Check Phase 2 marked complete
grep "Phase 2.*COMPLETE" docs/RESEARCH.md
```

---

## Notes for Implementation

**TDD Approach:**
- Tests written first, then verified to fail
- Existing retrieval code should make tests pass immediately
- Focus is on integration testing, not unit testing

**Notebook Execution:**
- Run cells in order during development
- Actual Recall@5 numbers will vary based on KB content
- Document any unexpected results in analysis section

**Ground Truth Accuracy:**
- Film IDs confirmed present in KB (18 ground truth films added earlier)
- If test fails, check film_id is correct for that film in your KB
- Use `grep -r "film_id" data/raw/*.json | grep <title>` to find correct ID

**Commit Frequency:**
- Commit after each task (14 commits total)
- Clear, descriptive commit messages
- Each commit should leave code in working state
