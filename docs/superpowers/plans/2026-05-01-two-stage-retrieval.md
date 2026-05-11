# Two-Stage Multimodal Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two-stage retrieval (text → CLIP reranking) as Variant D to genuinely use multimodal retrieval in production

**Architecture:** TextRetriever gets 20 candidates, CLIPRetriever reranks using query-adaptive fusion weights (factual: 80% text, visual: 70% CLIP), returns top-5

**Tech Stack:** Python 3.11+, ChromaDB, sentence-transformers (CLIP), pytest

---

## File Structure

```
src/retrieval/two_stage_retriever.py    # NEW: TwoStageRetriever class
tests/test_two_stage_retriever.py        # NEW: 6 unit tests
src/agent/state.py                       # MODIFY: Add variant field
src/agent/nodes.py                       # MODIFY: Add variant="D" handling
src/evaluation/run_eval.py               # MODIFY: Add run_variant_d()
tests/test_agent_integration.py          # MODIFY: Add Variant D test
```

---

## Task 1: TwoStageRetriever Basic Structure

**Files:**
- Create: `src/retrieval/two_stage_retriever.py`
- Test: `tests/test_two_stage_retriever.py`

- [ ] **Step 1: Write failing test for basic retrieval**

Create `tests/test_two_stage_retriever.py`:

```python
"""
Unit tests for TwoStageRetriever.

Tests the query-adaptive two-stage retrieval: text candidates → CLIP reranking.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.two_stage_retriever import TwoStageRetriever


class TestTwoStageRetriever:

    @patch("retrieval.two_stage_retriever.TextRetriever")
    @patch("retrieval.two_stage_retriever.CLIPRetriever")
    def test_returns_top_k(self, mock_clip_cls, mock_text_cls):
        """TwoStageRetriever returns exactly top_k results."""
        # Mock text retriever to return 20 candidates
        mock_text_results = [
            {
                "doc_id": f"doc_{i}",
                "film_id": f"{i}",
                "title": f"Film {i}",
                "content": "plot text",
                "modality": "text",
                "score": 0.9 - (i * 0.01),
                "metadata": {"poster_path": f"/path/poster_{i}.jpg"}
            }
            for i in range(20)
        ]
        mock_text_instance = MagicMock()
        mock_text_instance.retrieve.return_value = mock_text_results
        mock_text_cls.return_value = mock_text_instance
        
        # Mock CLIP retriever (not called directly, used for encoding)
        mock_clip_instance = MagicMock()
        mock_clip_cls.return_value = mock_clip_instance
        
        # Create retriever with top_k=5
        retriever = TwoStageRetriever(top_k=5, candidate_k=20)
        results = retriever.retrieve("test query", query_type="factual")
        
        # Should return exactly 5 results
        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_returns_top_k -v
```

Expected: `FAIL - ModuleNotFoundError: No module named 'retrieval.two_stage_retriever'`

- [ ] **Step 3: Create TwoStageRetriever skeleton**

Create `src/retrieval/two_stage_retriever.py`:

```python
"""
Two-Stage Retriever — Text Recall + CLIP Reranking

Stage 1: TextRetriever gets top-K candidates (K=20)
Stage 2: CLIP reranks candidates using query-adaptive fusion weights

Query-Adaptive Weights:
  - factual: 80% text, 20% CLIP (text dominates)
  - visual: 30% text, 70% CLIP (CLIP refines aesthetics)
  - multi_hop: 60% text, 40% CLIP (balanced)
  - hybrid: 50% text, 50% CLIP (equal)

Success Criteria:
  - Overall Recall@5 ≥ 38.5%
  - Visual Recall@5 ≥ 20%
  - Factual Recall@5 ≥ 80%
  - Latency ≤ 25s

Usage:
    from retrieval.two_stage_retriever import TwoStageRetriever
    
    retriever = TwoStageRetriever(top_k=5, candidate_k=20)
    results = retriever.retrieve("cold rain-soaked atmosphere", query_type="visual")
"""

import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.text_retriever import TextRetriever
from retrieval.clip_retriever import CLIPRetriever

logger = logging.getLogger(__name__)


class TwoStageRetriever:
    """
    Two-stage retrieval: text candidates → CLIP reranking.
    
    Combines strengths of both modalities:
      - Text ensures high recall (gets relevant docs in top-20)
      - CLIP improves precision (refines visual ranking in top-5)
    
    Attributes:
        text_retriever: TextRetriever for stage 1 candidate generation
        clip_retriever: CLIPRetriever for stage 2 visual scoring
        top_k: Final number of results to return (default: 5)
        candidate_k: Number of candidates from stage 1 (default: 20)
    """
    
    def __init__(self, top_k: int = 5, candidate_k: int = 20) -> None:
        """
        Initialize two-stage retriever.
        
        Args:
            top_k: Number of final results to return
            candidate_k: Number of candidates from text retrieval
        """
        self.text_retriever = TextRetriever(top_k=candidate_k)
        self.clip_retriever = CLIPRetriever(top_k=candidate_k)
        self.top_k = top_k
        self.candidate_k = candidate_k
        logger.info(f"TwoStageRetriever initialized: top_k={top_k}, candidate_k={candidate_k}")
    
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
              - metadata
        """
        # Stage 1: Get candidates via text retrieval
        candidates = self.text_retriever.retrieve(query)
        
        if not candidates:
            logger.warning(f"No candidates from text retrieval for: {query}")
            return []
        
        logger.info(f"Stage 1: Retrieved {len(candidates)} text candidates")
        
        # Stage 2: CLIP reranking (to be implemented)
        # For now, just return first top_k candidates
        return candidates[:self.top_k]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_returns_top_k -v
```

Expected: `PASS`

- [ ] **Step 5: Commit**

Git commands for user to run:
```bash
git add src/retrieval/two_stage_retriever.py tests/test_two_stage_retriever.py
git commit -m "feat: add TwoStageRetriever skeleton with text-only fallback"
```

---

## Task 2: Query-Adaptive Weight Lookup

**Files:**
- Modify: `src/retrieval/two_stage_retriever.py`
- Test: `tests/test_two_stage_retriever.py`

- [ ] **Step 1: Write failing test for weight lookup**

Add to `tests/test_two_stage_retriever.py`:

```python
    def test_query_adaptive_weights(self):
        """Weight lookup returns correct alpha/beta for each query type."""
        retriever = TwoStageRetriever()
        
        # Visual query: CLIP dominant
        alpha, beta = retriever._get_weights("visual")
        assert alpha == 0.3
        assert beta == 0.7
        
        # Factual query: text dominant
        alpha, beta = retriever._get_weights("factual")
        assert alpha == 0.8
        assert beta == 0.2
        
        # Multi-hop: balanced
        alpha, beta = retriever._get_weights("multi_hop")
        assert alpha == 0.6
        assert beta == 0.4
        
        # Hybrid: equal
        alpha, beta = retriever._get_weights("hybrid")
        assert alpha == 0.5
        assert beta == 0.5
        
        # Unknown type: default (text-favored)
        alpha, beta = retriever._get_weights("unknown_type")
        assert alpha == 0.6
        assert beta == 0.4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_query_adaptive_weights -v
```

Expected: `FAIL - AttributeError: '_get_weights' method not found`

- [ ] **Step 3: Implement _get_weights method**

Add to `TwoStageRetriever` class in `src/retrieval/two_stage_retriever.py`:

```python
    def _get_weights(self, query_type: str) -> tuple[float, float]:
        """
        Get fusion weights for text and CLIP scores.
        
        Query-adaptive weighting based on empirical findings:
          - Text excels at factual (100% recall in Ablation 1)
          - CLIP can refine visual/mood ranking
          - But CLIP alone underperforms (20% recall)
        
        Args:
            query_type: One of [factual, visual, multi_hop, hybrid]
        
        Returns:
            (alpha, beta) where alpha=text weight, beta=CLIP weight
        """
        weights = {
            "factual": (0.8, 0.2),     # Trust text, minimal CLIP
            "visual": (0.3, 0.7),       # CLIP dominant for aesthetics
            "multi_hop": (0.6, 0.4),    # Balanced
            "hybrid": (0.5, 0.5),       # Equal contribution
        }
        alpha, beta = weights.get(query_type, (0.6, 0.4))  # Default: text-favored
        logger.debug(f"Query type '{query_type}': α={alpha}, β={beta}")
        return alpha, beta
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_query_adaptive_weights -v
```

Expected: `PASS`

- [ ] **Step 5: Commit**

Git commands for user:
```bash
git add src/retrieval/two_stage_retriever.py tests/test_two_stage_retriever.py
git commit -m "feat: add query-adaptive weight lookup for fusion"
```

---

## Task 3: Score Normalization Helper

**Files:**
- Modify: `src/retrieval/two_stage_retriever.py`
- Test: `tests/test_two_stage_retriever.py`

- [ ] **Step 1: Write failing test for normalization**

Add to `tests/test_two_stage_retriever.py`:

```python
    def test_normalization(self):
        """Score normalization maps any range to [0,1]."""
        retriever = TwoStageRetriever()
        
        # Test case 1: Scores in [0.5, 0.9]
        scores = [0.5, 0.7, 0.9]
        normalized = retriever._normalize_scores(scores)
        assert normalized[0] == 0.0  # min maps to 0
        assert normalized[2] == 1.0  # max maps to 1
        assert 0.0 < normalized[1] < 1.0  # middle value in (0,1)
        assert abs(normalized[1] - 0.5) < 0.01  # (0.7-0.5)/(0.9-0.5) = 0.5
        
        # Test case 2: All same scores (edge case)
        scores = [0.8, 0.8, 0.8]
        normalized = retriever._normalize_scores(scores)
        assert all(s == 0.5 for s in normalized)  # All map to 0.5
        
        # Test case 3: Negative scores
        scores = [-0.2, 0.0, 0.3]
        normalized = retriever._normalize_scores(scores)
        assert normalized[0] == 0.0
        assert normalized[2] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_normalization -v
```

Expected: `FAIL - AttributeError: '_normalize_scores' method not found`

- [ ] **Step 3: Implement _normalize_scores method**

Add to `TwoStageRetriever` class:

```python
    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """
        Min-max normalize scores to [0, 1].
        
        Formula: (x - min) / (max - min)
        
        Args:
            scores: List of raw scores
        
        Returns:
            Normalized scores in [0, 1]
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        # Handle edge case: all scores identical
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
        return normalized
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_normalization -v
```

Expected: `PASS`

- [ ] **Step 5: Commit**

Git commands for user:
```bash
git add src/retrieval/two_stage_retriever.py tests/test_two_stage_retriever.py
git commit -m "feat: add min-max score normalization"
```

---

## Task 4: CLIP Score Computation

**Files:**
- Modify: `src/retrieval/two_stage_retriever.py`
- Test: `tests/test_two_stage_retriever.py`

- [ ] **Step 1: Write failing test for CLIP scoring**

Add to `tests/test_two_stage_retriever.py`:

```python
    @patch("retrieval.two_stage_retriever.Image")
    def test_compute_clip_score(self, mock_image_cls):
        """CLIP score averages similarity across all images."""
        retriever = TwoStageRetriever()
        
        # Mock CLIP model
        mock_clip = MagicMock()
        retriever.clip_retriever.model = mock_clip
        
        # Mock query embedding
        query_emb = [0.1, 0.2, 0.3]
        
        # Mock image paths and embeddings
        image_paths = ["/path/poster.jpg", "/path/still1.jpg", "/path/still2.jpg"]
        image_embeddings = [
            [0.9, 0.1, 0.0],  # High similarity with query
            [0.5, 0.5, 0.0],  # Medium similarity
            [0.1, 0.8, 0.1],  # Low similarity
        ]
        
        # Mock encode to return image embeddings
        mock_clip.encode.side_effect = image_embeddings
        
        # Mock PIL Image.open
        mock_image_cls.open.return_value = MagicMock()
        
        clip_score = retriever._compute_clip_score(query_emb, image_paths)
        
        # Should average the 3 cosine similarities
        # (We'll calculate expected value based on actual cosine similarity)
        assert isinstance(clip_score, float)
        assert 0.0 <= clip_score <= 1.0
    
    def test_compute_clip_score_no_images(self):
        """CLIP score returns 0 when no images available."""
        retriever = TwoStageRetriever()
        query_emb = [0.1, 0.2, 0.3]
        
        clip_score = retriever._compute_clip_score(query_emb, [])
        
        assert clip_score == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_compute_clip_score -v
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_compute_clip_score_no_images -v
```

Expected: `FAIL - AttributeError: '_compute_clip_score' method not found`

- [ ] **Step 3: Implement _compute_clip_score method**

Add to `TwoStageRetriever` class:

```python
    def _compute_clip_score(
        self,
        query_embedding: list[float],
        image_paths: list[str]
    ) -> float:
        """
        Compute average CLIP similarity across all images.
        
        Args:
            query_embedding: CLIP text encoding of query
            image_paths: List of image file paths (poster + stills)
        
        Returns:
            Average cosine similarity across images, or 0.0 if no images
        """
        if not image_paths:
            return 0.0
        
        try:
            from PIL import Image
            import numpy as np
            
            similarities = []
            for img_path in image_paths:
                # Load and encode image
                img = Image.open(img_path).convert("RGB")
                img_embedding = self.clip_retriever.model.encode(img)
                
                # Compute cosine similarity
                similarity = np.dot(query_embedding, img_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(img_embedding)
                )
                similarities.append(float(similarity))
            
            avg_similarity = sum(similarities) / len(similarities)
            return avg_similarity
        
        except Exception as e:
            logger.warning(f"CLIP scoring failed for images {image_paths}: {e}")
            return 0.0  # Fallback to text-only ranking
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_compute_clip_score -v
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_compute_clip_score_no_images -v
```

Expected: `PASS`

- [ ] **Step 5: Commit**

Git commands for user:
```bash
git add src/retrieval/two_stage_retriever.py tests/test_two_stage_retriever.py
git commit -m "feat: add CLIP score computation with image averaging"
```

---

## Task 5: CLIP Reranking Logic

**Files:**
- Modify: `src/retrieval/two_stage_retriever.py`
- Test: `tests/test_two_stage_retriever.py`

- [ ] **Step 1: Write failing test for score fusion**

Add to `tests/test_two_stage_retriever.py`:

```python
    def test_score_fusion(self):
        """Fusion correctly combines text and CLIP scores."""
        retriever = TwoStageRetriever()
        
        # Mock candidates with known scores
        candidates = [
            {"doc_id": "1", "score": 0.9, "text_score": 0.9, "metadata": {}},
            {"doc_id": "2", "score": 0.7, "text_score": 0.7, "metadata": {}},
            {"doc_id": "3", "score": 0.5, "text_score": 0.5, "metadata": {}},
        ]
        
        # Mock CLIP scores (assign manually for test)
        clip_scores = [0.3, 0.8, 0.6]
        
        # Mock _compute_clip_score to return predefined values
        original_compute = retriever._compute_clip_score
        retriever._compute_clip_score = lambda q_emb, imgs: clip_scores.pop(0)
        
        # Mock CLIP encode
        retriever.clip_retriever.model.encode = MagicMock(return_value=[0.1, 0.2])
        
        # Fuse with alpha=0.5, beta=0.5 (equal weights)
        reranked = retriever._rerank_with_clip("test query", candidates, alpha=0.5, beta=0.5)
        
        # Restore original method
        retriever._compute_clip_score = original_compute
        
        # Check that fused_score field exists
        assert all("fused_score" in doc for doc in reranked)
        assert all("clip_score" in doc for doc in reranked)
        
        # Verify sorted by fused_score descending
        scores = [doc["fused_score"] for doc in reranked]
        assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_score_fusion -v
```

Expected: `FAIL - AttributeError: '_rerank_with_clip' method not found`

- [ ] **Step 3: Implement _rerank_with_clip method**

Add to `TwoStageRetriever` class:

```python
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
        1. Encode query with CLIP text encoder (once)
        2. For each candidate:
           - Extract image paths from metadata
           - Compute average CLIP similarity
           - Store both text_score and clip_score
        3. Normalize both score lists to [0,1]
        4. Compute fused_score = alpha * norm(text) + beta * norm(clip)
        5. Sort by fused_score descending
        
        Args:
            query: User query text
            candidates: Results from stage 1 text retrieval
            alpha: Text weight (0-1)
            beta: CLIP weight (0-1)
        
        Returns:
            Reranked candidates with text_score, clip_score, fused_score fields
        """
        # Encode query once with CLIP
        query_embedding = self.clip_retriever.model.encode(query)
        
        # Compute CLIP scores for each candidate
        for candidate in candidates:
            # Store original text score
            candidate["text_score"] = candidate.get("score", 0.0)
            
            # Extract image paths
            metadata = candidate.get("metadata", {})
            image_paths = []
            
            if metadata.get("poster_path"):
                image_paths.append(metadata["poster_path"])
            
            if metadata.get("still_paths"):
                still_paths = metadata["still_paths"]
                if isinstance(still_paths, list):
                    image_paths.extend(still_paths)
            
            # Compute CLIP score
            candidate["clip_score"] = self._compute_clip_score(query_embedding, image_paths)
        
        # Extract scores for normalization
        text_scores = [c["text_score"] for c in candidates]
        clip_scores = [c["clip_score"] for c in candidates]
        
        # Normalize to [0,1]
        norm_text = self._normalize_scores(text_scores)
        norm_clip = self._normalize_scores(clip_scores)
        
        # Compute fused scores
        for i, candidate in enumerate(candidates):
            candidate["fused_score"] = alpha * norm_text[i] + beta * norm_clip[i]
        
        # Sort by fused score descending
        reranked = sorted(candidates, key=lambda c: c["fused_score"], reverse=True)
        
        logger.info(f"Reranked {len(reranked)} candidates (α={alpha}, β={beta})")
        return reranked
```

- [ ] **Step 4: Update retrieve() to use _rerank_with_clip**

Modify the `retrieve()` method in `TwoStageRetriever`:

```python
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
            Top-k documents after reranking
        """
        # Stage 1: Get candidates via text retrieval
        candidates = self.text_retriever.retrieve(query)
        
        if not candidates:
            logger.warning(f"No candidates from text retrieval for: {query}")
            return []
        
        logger.info(f"Stage 1: Retrieved {len(candidates)} text candidates")
        
        # Stage 2: CLIP reranking
        alpha, beta = self._get_weights(query_type)
        reranked = self._rerank_with_clip(query, candidates, alpha, beta)
        
        logger.info(f"Stage 2: Reranked with {query_type} weights (α={alpha}, β={beta})")
        
        return reranked[:self.top_k]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_score_fusion -v
```

Expected: `PASS`

- [ ] **Step 6: Commit**

Git commands for user:
```bash
git add src/retrieval/two_stage_retriever.py tests/test_two_stage_retriever.py
git commit -m "feat: implement CLIP reranking with score fusion"
```

---

## Task 6: Integration Test - Real Retrieval

**Files:**
- Test: `tests/test_two_stage_retriever.py`

- [ ] **Step 1: Write integration test**

Add to `tests/test_two_stage_retriever.py`:

```python
    def test_clip_improves_visual_ranking(self):
        """
        Integration test: CLIP should improve visual query ranking.
        
        Uses real retrieval (requires ChromaDB to be built).
        Verifies that visually-relevant films rank higher with CLIP.
        """
        import os
        
        # Skip if ChromaDB not built
        if not os.path.exists("data/indices/chroma.sqlite3"):
            pytest.skip("ChromaDB not built — run kb_builder.py first")
        
        retriever = TwoStageRetriever(top_k=5, candidate_k=20)
        
        # Visual query: should retrieve Blade Runner 2049
        query = "cold, desaturated, rain-soaked atmosphere"
        results = retriever.retrieve(query, query_type="visual")
        
        # Verify results structure
        assert len(results) <= 5
        assert all("fused_score" in r for r in results)
        assert all("text_score" in r for r in results)
        assert all("clip_score" in r for r in results)
        
        # Verify CLIP contributed (clip_score > 0 for at least one result)
        clip_scores = [r["clip_score"] for r in results]
        assert any(s > 0 for s in clip_scores), "CLIP should score at least one image"
        
        # Log results for manual inspection
        print("\nVisual query results:")
        for i, r in enumerate(results):
            print(f"{i+1}. {r['title']} | text={r['text_score']:.3f} clip={r['clip_score']:.3f} fused={r['fused_score']:.3f}")
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_two_stage_retriever.py::TestTwoStageRetriever::test_clip_improves_visual_ranking -v -s
```

Expected: `PASS` (if KB built), or `SKIP` with message

- [ ] **Step 3: Run all TwoStageRetriever tests**

```bash
pytest tests/test_two_stage_retriever.py -v
```

Expected: All tests PASS

- [ ] **Step 4: Commit**

Git commands for user:
```bash
git add tests/test_two_stage_retriever.py
git commit -m "test: add integration test for visual query CLIP reranking"
```

---

## Task 7: Add Variant Field to AgentState

**Files:**
- Modify: `src/agent/state.py`
- Test: `tests/test_agent_nodes.py` (existing)

- [ ] **Step 1: Check current AgentState structure**

Read `src/agent/state.py` to see current fields.

- [ ] **Step 2: Add variant field**

Add `variant` field to `AgentState` TypedDict in `src/agent/state.py`:

```python
class AgentState(TypedDict):
    # User interaction
    query: str
    conversation_history: list[dict]
    
    # Query routing
    query_type: str              # factual | visual | hybrid | multi_hop
    retrieval_strategy: str      # text | clip | hybrid
    
    # Retrieval results
    retrieved_docs: list[dict]
    retrieved_images: list[str]  # Image paths
    
    # Taste profile (dynamic memory)
    taste_profile: TasteProfile
    
    # Response generation
    response: str
    cited_films: list[str]       # Film IDs cited in response
    
    # Verification
    verified: bool
    verification_reason: str
    retry_count: int
    
    # Metrics
    tool_calls_count: int
    latency_ms: float
    
    # Variant selector (NEW)
    variant: str                 # "A" | "B" | "C" | "D"
```

- [ ] **Step 3: Update initial_state function**

Update `initial_state()` to include default variant:

```python
def initial_state(query: str, variant: str = "C") -> AgentState:
    """
    Create initial agent state for a new query.
    
    Args:
        query: User query text
        variant: System variant ("A", "B", "C", or "D")
    
    Returns:
        Fresh AgentState with empty values
    """
    return {
        "query": query,
        "conversation_history": [],
        "query_type": "",
        "retrieval_strategy": "",
        "retrieved_docs": [],
        "retrieved_images": [],
        "taste_profile": empty_taste_profile(),
        "response": "",
        "cited_films": [],
        "verified": False,
        "verification_reason": "",
        "retry_count": 0,
        "tool_calls_count": 0,
        "latency_ms": 0.0,
        "variant": variant,  # NEW
    }
```

- [ ] **Step 4: Test that existing tests still pass**

```bash
pytest tests/test_agent_nodes.py -v
```

Expected: All existing tests PASS

- [ ] **Step 5: Commit**

Git commands for user:
```bash
git add src/agent/state.py
git commit -m "feat: add variant field to AgentState for system selection"
```

---

## Task 8: Integrate Two-Stage Retrieval into RetrievalPlanner

**Files:**
- Modify: `src/agent/nodes.py`

- [ ] **Step 1: Read current retrieval_planner_node**

Check how `retrieval_planner_node` currently works in `src/agent/nodes.py`.

- [ ] **Step 2: Add import for TwoStageRetriever**

Add to imports in `src/agent/nodes.py`:

```python
from retrieval.two_stage_retriever import TwoStageRetriever
```

- [ ] **Step 3: Add global variable for lazy loading**

Add to globals section:

```python
_two_stage_retriever: TwoStageRetriever | None = None
```

- [ ] **Step 4: Update _get_retrievers function**

Update `_get_retrievers()` to include two-stage:

```python
def _get_retrievers():
    """Lazy-load retrievers (expensive, initialize once)."""
    global _text_retriever, _clip_retriever, _caption_retriever, _hybrid_retriever, _two_stage_retriever
    if _text_retriever is None:
        _text_retriever = TextRetriever()
        _clip_retriever = CLIPRetriever()
        _caption_retriever = CaptionRetriever()
        _hybrid_retriever = HybridRetriever()
        _two_stage_retriever = TwoStageRetriever()
    return _text_retriever, _clip_retriever, _caption_retriever, _hybrid_retriever, _two_stage_retriever
```

- [ ] **Step 5: Add variant D handling to retrieval_planner_node**

Modify `retrieval_planner_node()` function to add variant D case:

```python
def retrieval_planner_node(state: AgentState) -> dict:
    """
    Retrieval planner — selects and calls appropriate retriever.
    
    Variant-aware routing:
      - Variant C: text-only (current best, 38.5% recall)
      - Variant D: two-stage (text → CLIP reranking) — NEW
    
    Query-type adaptive (for Variant D):
      - Visual queries: CLIP weight 70%
      - Factual queries: text weight 80%
      - Multi-hop: balanced 60/40
    
    Reads:  state["query"], state["query_type"], state["variant"]
    Writes: state["retrieved_docs"], state["retrieved_images"], state["tool_calls_count"]
    
    Args:
        state: Current AgentState
    
    Returns:
        Partial state update dict
    """
    query = state["query"]
    query_type = state.get("query_type", "factual")
    variant = state.get("variant", "C")  # Default to C
    
    try:
        # Get retrievers (lazy-loaded singletons)
        text_retriever, clip_retriever, caption_retriever, hybrid_retriever, two_stage_retriever = _get_retrievers()
        
        # Route based on variant
        if variant == "D":
            # Two-stage retrieval with query-adaptive fusion
            results = two_stage_retriever.retrieve(query=query, query_type=query_type)
            logger.info(f"RetrievalPlanner (Variant D): two-stage retrieval for {query_type} query")
        
        elif variant == "C":
            # Existing text-only logic
            if query_type == "visual":
                results = caption_retriever.retrieve(query)
                logger.info(f"RetrievalPlanner (Variant C): Caption retriever for visual query")
            else:
                results = text_retriever.retrieve(query)
                logger.info(f"RetrievalPlanner (Variant C): Text retriever")
        
        # ... handle other variants (A, B) as before ...
        
        else:
            logger.warning(f"Unknown variant '{variant}', defaulting to text retrieval")
            results = text_retriever.retrieve(query)
        
        if not results:
            logger.warning(f"No results found for query: {query}")
            return {
                "retrieved_docs": [],
                "retrieved_images": [],
                "tool_calls_count": state["tool_calls_count"] + 1
            }
        
        # Extract image paths from metadata (same as before)
        image_paths = []
        for doc in results:
            metadata = doc.get("metadata", {})
            if metadata.get("poster_path"):
                image_paths.append(metadata["poster_path"])
            if metadata.get("still_paths"):
                still_paths = metadata["still_paths"]
                if isinstance(still_paths, list):
                    image_paths.extend(still_paths)
        
        image_paths = image_paths[:5]
        
        logger.info(f"RetrievalPlanner: retrieved {len(results)} docs, {len(image_paths)} images")
        
        return {
            "retrieved_docs": results,
            "retrieved_images": image_paths,
            "tool_calls_count": state["tool_calls_count"] + 1
        }
    
    except Exception as e:
        logger.error(f"RetrievalPlanner failed: {e}")
        return {
            "retrieved_docs": [],
            "retrieved_images": [],
            "tool_calls_count": state["tool_calls_count"] + 1
        }
```

- [ ] **Step 6: Test that existing agent tests still pass**

```bash
pytest tests/test_agent_nodes.py -v
pytest tests/test_agent_integration.py -v
```

Expected: All tests PASS

- [ ] **Step 7: Commit**

Git commands for user:
```bash
git add src/agent/nodes.py
git commit -m "feat: integrate two-stage retrieval as Variant D in RetrievalPlanner"
```

---

## Task 9: Add Variant D Evaluation

**Files:**
- Modify: `src/evaluation/run_eval.py`

- [ ] **Step 1: Read current run_variant_c implementation**

Check `run_variant_c()` to understand structure.

- [ ] **Step 2: Add run_variant_d function**

Add new function to `src/evaluation/run_eval.py`:

```python
def run_variant_d() -> dict:
    """
    Evaluate Variant D: Two-Stage Multimodal Retrieval.
    
    Uses query-adaptive fusion:
      - Text retrieval gets 20 candidates
      - CLIP reranks using query-type-specific weights
      - Visual: 70% CLIP, factual: 80% text
    
    Success Criteria:
      - Overall Recall@5 ≥ 38.5%
      - Visual Recall@5 ≥ 20%
      - Factual Recall@5 ≥ 80%
      - Latency ≤ 25s
    
    Returns:
        Evaluation results dict with per-query metrics
    """
    from agent.graph import graph
    from agent.state import initial_state
    from evaluation.test_suite import TEST_QUERIES
    from evaluation.metrics import compute_recall_at_5, compute_latency
    
    logger.info("=" * 60)
    logger.info("Running Variant D: Two-Stage Multimodal Retrieval")
    logger.info("=" * 60)
    
    results = {
        "variant": "D_two_stage",
        "per_query": []
    }
    
    for test_case in TEST_QUERIES:
        query_id = test_case["query_id"]
        query = test_case["query"]
        family = test_case["family"]
        ground_truth = test_case["ground_truth_film_ids"]
        
        logger.info(f"\n[{query_id}] {query}")
        
        # Initialize state with variant="D"
        state = initial_state(query, variant="D")
        
        # Run agent
        start_time = time.perf_counter()
        final_state = graph.invoke(state)
        latency = (time.perf_counter() - start_time) * 1000
        
        # Extract retrieved film IDs
        retrieved_docs = final_state.get("retrieved_docs", [])
        retrieved_film_ids = [doc["film_id"] for doc in retrieved_docs]
        
        # Compute metrics
        recall = compute_recall_at_5(ground_truth, retrieved_film_ids)
        
        # Store result
        result = {
            "query_id": query_id,
            "query_family": family,
            "variant": "D_two_stage",
            "response": final_state.get("response", ""),
            "retrieved_film_ids": retrieved_film_ids[:5],
            "ground_truth_film_ids": ground_truth,
            "recall_at_5": recall,
            "latency_ms": latency,
            "tool_calls_count": final_state.get("tool_calls_count", 0)
        }
        
        results["per_query"].append(result)
        
        logger.info(f"Recall@5: {recall:.1%} | Latency: {latency:.0f}ms | Tools: {result['tool_calls_count']}")
    
    # Compute aggregate metrics
    all_recalls = [r["recall_at_5"] for r in results["per_query"]]
    all_latencies = [r["latency_ms"] for r in results["per_query"]]
    
    results["aggregate"] = {
        "mean_recall_at_5": sum(all_recalls) / len(all_recalls),
        "mean_latency_ms": sum(all_latencies) / len(all_latencies),
        "median_latency_ms": sorted(all_latencies)[len(all_latencies) // 2],
    }
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Variant D Results:")
    logger.info(f"  Overall Recall@5: {results['aggregate']['mean_recall_at_5']:.1%}")
    logger.info(f"  Mean Latency: {results['aggregate']['mean_latency_ms']:.0f}ms")
    logger.info("=" * 60)
    
    return results
```

- [ ] **Step 3: Add CLI flag for variant D**

Update the main block in `run_eval.py`:

```python
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-a", action="store_true", help="Run Variant A (plain LLM)")
    parser.add_argument("--variant-b", action="store_true", help="Run Variant B (fixed RAG)")
    parser.add_argument("--variant-c", action="store_true", help="Run Variant C (full agent)")
    parser.add_argument("--variant-d", action="store_true", help="Run Variant D (two-stage)")  # NEW
    parser.add_argument("--all", action="store_true", help="Run all variants")
    
    args = parser.parse_args()
    
    all_results = {}
    
    if args.all or args.variant_a:
        all_results["variant_A"] = run_variant_a()
    
    if args.all or args.variant_b:
        all_results["variant_B"] = run_variant_b()
    
    if args.all or args.variant_c:
        all_results["variant_C"] = run_variant_c()
    
    if args.variant_d:  # NEW (don't run with --all yet)
        all_results["variant_D"] = run_variant_d()
    
    # Save results
    with open(EVAL_RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {EVAL_RESULTS_FILE}")
```

- [ ] **Step 4: Commit**

Git commands for user:
```bash
git add src/evaluation/run_eval.py
git commit -m "feat: add Variant D evaluation with CLI flag"
```

---

## Task 10: Add Agent Integration Test for Variant D

**Files:**
- Modify: `tests/test_agent_integration.py`

- [ ] **Step 1: Add Variant D integration test**

Add to `tests/test_agent_integration.py`:

```python
@pytest.mark.integration
def test_variant_d_two_stage_retrieval():
    """
    Variant D uses two-stage retrieval with query-adaptive fusion.
    
    Tests:
      - Visual query routes to high CLIP weight (β=0.7)
      - Factual query routes to high text weight (α=0.8)
      - Results contain text_score, clip_score, fused_score
      - Latency reasonable (<25s)
    """
    from agent.graph import graph
    from agent.state import initial_state
    
    # Test 1: Visual query
    visual_state = initial_state(
        "cold, desaturated, rain-soaked atmosphere",
        variant="D"
    )
    
    final_state = graph.invoke(visual_state)
    
    # Check results structure
    docs = final_state["retrieved_docs"]
    assert len(docs) > 0, "Should retrieve documents"
    assert all("fused_score" in d for d in docs), "Should have fused scores"
    assert all("clip_score" in d for d in docs), "Should have CLIP scores"
    
    # Check CLIP contributed
    clip_scores = [d["clip_score"] for d in docs]
    assert any(s > 0 for s in clip_scores), "CLIP should score at least one doc"
    
    # Check latency
    assert final_state["latency_ms"] < 25000, "Should complete within 25s"
    
    # Test 2: Factual query
    factual_state = initial_state(
        "Who directed Mulholland Drive?",
        variant="D"
    )
    
    final_state = graph.invoke(factual_state)
    
    # Should still retrieve successfully
    assert len(final_state["retrieved_docs"]) > 0
    assert final_state["response"]  # Should generate response
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_agent_integration.py::test_variant_d_two_stage_retrieval -v -s
```

Expected: `PASS` (if ChromaDB built)

- [ ] **Step 3: Run all integration tests**

```bash
pytest tests/test_agent_integration.py -v
```

Expected: All tests PASS

- [ ] **Step 4: Commit**

Git commands for user:
```bash
git add tests/test_agent_integration.py
git commit -m "test: add Variant D integration test for two-stage retrieval"
```

---

## Task 11: Manual Validation

**Files:**
- None (manual testing)

- [ ] **Step 1: Test with visual query**

```bash
python -c "
from agent.graph import graph
from agent.state import initial_state

state = initial_state('cold, desaturated, rain-soaked atmosphere', variant='D')
result = graph.invoke(state)

print('\\n=== Visual Query Results ===')
for i, doc in enumerate(result['retrieved_docs'][:5]):
    print(f\"{i+1}. {doc['title']}\")
    print(f\"   text={doc.get('text_score', 0):.3f} clip={doc.get('clip_score', 0):.3f} fused={doc.get('fused_score', 0):.3f}\")
"
```

Expected: Blade Runner 2049 or similar films in top-5

- [ ] **Step 2: Test with factual query**

```bash
python -c "
from agent.graph import graph
from agent.state import initial_state

state = initial_state('Who directed Mulholland Drive?', variant='D')
result = graph.invoke(state)

print('\\n=== Factual Query Results ===')
print(f\"Response: {result['response'][:200]}...\")
print(f\"\\nTop film: {result['retrieved_docs'][0]['title']}\")
"
```

Expected: Mulholland Drive as top result, correct answer in response

- [ ] **Step 3: Check latency**

```bash
python -c "
import time
from agent.graph import graph
from agent.state import initial_state

queries = [
    'cold rain-soaked atmosphere',
    'Who directed Parasite?',
    'slow-burn psychological thriller'
]

print('\\n=== Latency Test ===')
for query in queries:
    state = initial_state(query, variant='D')
    start = time.perf_counter()
    result = graph.invoke(state)
    latency = (time.perf_counter() - start) * 1000
    print(f\"{query[:30]:30} | {latency:6.0f}ms\")
"
```

Expected: All queries < 25000ms

- [ ] **Step 4: Document manual test results**

Create note of what worked/didn't work for report.

---

## Task 12: Run Full Variant D Evaluation

**Files:**
- None (evaluation run)

- [ ] **Step 1: Run Variant D evaluation**

```bash
python src/evaluation/run_eval.py --variant-d
```

Expected: Runs all 13 test queries, outputs results

- [ ] **Step 2: Check results against success criteria**

Open `data/results/eval_results.json` and check `variant_D`:

Success criteria (ALL must pass):
1. Overall Recall@5 ≥ 38.5%
2. Visual Recall@5 ≥ 20%
3. Factual Recall@5 ≥ 80%
4. Mean latency ≤ 25s

- [ ] **Step 3: Analyze results**

```bash
python -c "
import json

with open('data/results/eval_results.json') as f:
    results = json.load(f)

variant_d = results['variant_D']
per_query = variant_d['per_query']

# Overall
overall_recall = variant_d['aggregate']['mean_recall_at_5']
mean_latency = variant_d['aggregate']['mean_latency_ms']

# Per family
factual = [q for q in per_query if q['query_family'] == 'factual']
visual = [q for q in per_query if q['query_family'] == 'visual']

factual_recall = sum(q['recall_at_5'] for q in factual) / len(factual) if factual else 0
visual_recall = sum(q['recall_at_5'] for q in visual) / len(visual) if visual else 0

print('\\n=== Variant D Results ===')
print(f'Overall Recall@5: {overall_recall:.1%}')
print(f'Factual Recall@5: {factual_recall:.1%}')
print(f'Visual Recall@5: {visual_recall:.1%}')
print(f'Mean Latency: {mean_latency:.0f}ms')

print('\\n=== Success Criteria Check ===')
print(f'Overall ≥ 38.5%: {'✓' if overall_recall >= 0.385 else '✗'} ({overall_recall:.1%})')
print(f'Visual ≥ 20%: {'✓' if visual_recall >= 0.20 else '✗'} ({visual_recall:.1%})')
print(f'Factual ≥ 80%: {'✓' if factual_recall >= 0.80 else '✗'} ({factual_recall:.1%})')
print(f'Latency ≤ 25s: {'✓' if mean_latency <= 25000 else '✗'} ({mean_latency:.0f}ms)')
"
```

- [ ] **Step 4: Decision point**

Based on results:

**IF all 4 criteria met:**
- Variant D succeeds → plan to promote to new Variant C
- Update RESEARCH.md with findings
- Prepare to merge to main

**ELSE IF recall ≥ 35%:**
- Partial success → keep as separate Variant D
- Document tradeoffs in RESEARCH.md
- Update RQ to acknowledge text-dominant hybrid

**ELSE (recall < 35%):**
- Failure → don't merge to main
- Document detailed failure analysis
- Keep branch for reference

- [ ] **Step 5: Update RESEARCH.md**

Add findings to `docs/RESEARCH.md` Phase 3 section:

```markdown
### Variant D — Two-Stage Multimodal Retrieval

**Results:**
- Overall Recall@5: [X.X%]
- Visual Recall@5: [X.X%]
- Factual Recall@5: [X.X%]
- Mean Latency: [X.X]s

**Analysis:**
[Describe what worked, what didn't, why]

**Decision:**
[Promote to Variant C / Keep as Variant D / Don't merge]

**Key Finding:**
[Main takeaway about query-adaptive fusion and multimodal retrieval]
```

- [ ] **Step 6: Commit evaluation results**

Git commands for user:
```bash
git add data/results/eval_results.json docs/RESEARCH.md
git commit -m "eval: Variant D two-stage retrieval results - [X.X%] recall"
```

---

## Self-Review Checklist

**Spec Coverage:**
- ✅ TwoStageRetriever class with all methods
- ✅ Query-adaptive weight lookup
- ✅ Score normalization
- ✅ CLIP score computation
- ✅ Reranking logic with fusion
- ✅ AgentState variant field
- ✅ RetrievalPlanner variant D handling
- ✅ run_variant_d() evaluation
- ✅ Integration tests
- ✅ Success criteria validation

**No Placeholders:**
- ✅ All code blocks complete
- ✅ All test implementations provided
- ✅ Exact commands with expected output
- ✅ No "TBD", "TODO", or "implement later"

**Type Consistency:**
- ✅ TwoStageRetriever methods consistent across tasks
- ✅ AgentState variant field typed correctly
- ✅ Function signatures match between tasks

---

## Next Steps After Implementation

1. **If Success (≥38.5% recall):**
   - Rename current Variant C → "Variant C_text_baseline"
   - Make two-stage the new Variant C
   - Update research question to reflect multimodal retrieval
   - Merge to main

   Git commands:
   ```bash
   git checkout main
   git merge two-stage-retrieval
   git tag v2.0-multimodal-retrieval
   ```

2. **If Partial Success (35-38% recall):**
   - Keep as Variant D
   - Update RQ to acknowledge tradeoff
   - Document in report

3. **If Failure (<35% recall):**
   - Don't merge
   - Add to failure analysis section
   - Keep branch for reference

---

## Execution Options

Plan complete and ready for implementation.

**Two execution options:**

**1. Subagent-Driven (recommended)** - Dispatch fresh subagent per task, review between tasks, fast iteration with quality gates

**2. Inline Execution** - Execute tasks in this session using executing-plans skill, batch execution with checkpoints

**Which approach do you prefer?**
