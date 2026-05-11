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
            "factual": (0.8, 0.2),      # Trust text, minimal CLIP (80% recall - working well)
            "visual": (0.0, 1.0),        # EXPERIMENT 1: Pure CLIP - eliminate text interference
            "multi_hop": (0.85, 0.15),   # Heavy text bias for constraint-heavy queries with metadata filtering
            "hybrid": (0.5, 0.5),        # Equal contribution
        }
        alpha, beta = weights.get(query_type, (0.6, 0.4))  # Default: text-favored
        logger.debug(f"Query type '{query_type}': α={alpha}, β={beta}")
        return alpha, beta

    def retrieve(
        self,
        query: str,
        query_type: str = "factual",
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        Two-stage retrieval with query-adaptive fusion.

        Stage 1: Get text candidates
        Stage 2: Score with CLIP, fuse scores adaptively, return top-k

        Args:
            query: User query text
            query_type: One of [factual, visual, multi_hop, hybrid]
            metadata_filter: Optional metadata filter for text retrieval

        Returns:
            Top-k documents after reranking, each with:
              - doc_id, film_id, title, content, modality
              - text_score, clip_score, fused_score
              - metadata
        """
        # Stage 1: Get candidates via text retrieval
        candidates = self.text_retriever.retrieve(query, metadata_filter=metadata_filter)

        if not candidates:
            logger.warning(f"No candidates from text retrieval for: {query}")
            return []

        logger.info(f"Stage 1: Retrieved {len(candidates)} text candidates")

        # Stage 2: Get CLIP scores for all candidates
        clip_results = self.clip_retriever.retrieve_by_text(query)

        # Build a lookup: film_id → CLIP score
        clip_scores = {}
        for clip_result in clip_results:
            film_id = clip_result.get("film_id", "")
            if film_id:
                clip_scores[film_id] = clip_result.get("score", 0.0)

        logger.info(f"Stage 2: Retrieved {len(clip_results)} CLIP results")

        # Get fusion weights
        alpha, beta = self._get_weights(query_type)

        # Fuse scores and add to candidates
        for candidate in candidates:
            film_id = candidate.get("film_id", "")
            text_score = candidate.get("score", 0.0)
            clip_score = clip_scores.get(film_id, 0.0)  # Default to 0 if no CLIP match

            # Fused score: weighted combination
            fused_score = alpha * text_score + beta * clip_score

            candidate["text_score"] = round(text_score, 4)
            candidate["clip_score"] = round(clip_score, 4)
            candidate["fused_score"] = round(fused_score, 4)

        # Re-rank by fused score
        candidates_sorted = sorted(candidates, key=lambda x: x["fused_score"], reverse=True)

        # Return top-k
        results = candidates_sorted[:self.top_k]

        logger.info(f"Two-stage retrieval: {len(candidates)} candidates → {len(results)} results")
        return results
