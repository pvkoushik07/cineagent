"""
Two-Stage Retriever — Text Recall + CLIP Reranking

Stage 1: TextRetriever gets top-K candidates (K=20)
Stage 2: CLIP reranks candidates using query-adaptive fusion weights

Query-Adaptive Weights:
  - factual: 80% text, 20% CLIP (text dominates)
  - visual: 5% text, 95% CLIP (EXPERIMENT 3: extreme CLIP with tiny text signal)
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

    def _get_clip_scores_for_candidates(self, query: str, candidates: list[dict]) -> dict:
        """
        Get CLIP scores for each CANDIDATE FILM ONLY.

        FIX for Bug #1: Instead of searching the entire image collection,
        get CLIP scores only for films that appear in the text candidates.
        This requires getting images per film and computing CLIP scores.

        Args:
            query: User query text
            candidates: List of text candidates from Stage 1

        Returns:
            Dict mapping film_id → CLIP score (best score for that film's images)
        """
        # Extract unique film IDs from candidates
        candidate_film_ids = list(set([c.get("film_id") for c in candidates if c.get("film_id")]))
        logger.debug(f"Computing CLIP scores for {len(candidate_film_ids)} candidate films")

        if not candidate_film_ids:
            return {}

        # Query CLIP collection with metadata filter for these specific films
        # This restricts CLIP search to only images from candidate films
        clip_scores = {}

        try:
            # Get CLIP embedding for the query
            query_embedding = self.clip_retriever.model.encode(query).tolist()

            # For each candidate film, get its best CLIP score
            for film_id in candidate_film_ids:
                # Query only for images with this film_id
                results = self.clip_retriever.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=1,  # Just get the best match for this film
                    where={"film_id": film_id},
                    include=["distances"],
                )

                if results["distances"] and len(results["distances"][0]) > 0:
                    distance = results["distances"][0][0]
                    score = 1 - distance  # Convert distance to similarity
                    clip_scores[film_id] = round(score, 4)
                else:
                    # No image for this film in CLIP collection
                    clip_scores[film_id] = 0.0

        except Exception as e:
            logger.warning(f"Error computing CLIP scores for candidates: {e}")
            # Return 0 scores for all films if error occurs
            clip_scores = {film_id: 0.0 for film_id in candidate_film_ids}

        return clip_scores

    def _normalize_and_fuse_scores(
        self,
        candidates: list[dict],
        clip_scores: dict,
        alpha: float,
        beta: float
    ) -> list[dict]:
        """
        Normalize text and CLIP scores, then compute fused scores.

        FIX for Bug #3: Normalize both score types to [0, 1] before fusion
        to ensure weights are meaningful and not arbitrary.

        Args:
            candidates: List of text candidates
            clip_scores: Dict mapping film_id → CLIP score
            alpha: Weight for text score
            beta: Weight for CLIP score

        Returns:
            Updated candidates with normalized and fused scores
        """
        # Add CLIP scores to candidates
        for candidate in candidates:
            film_id = candidate.get("film_id", "")
            clip_score = clip_scores.get(film_id, 0.0)
            candidate["clip_score"] = clip_score

        # Extract all scores for normalization
        text_scores = [c.get("score", 0.0) for c in candidates]
        clip_scores_list = [c.get("clip_score", 0.0) for c in candidates]

        # Normalize text scores to [0, 1]
        text_min, text_max = min(text_scores) if text_scores else 0, max(text_scores) if text_scores else 1
        text_range = text_max - text_min if text_max > text_min else 1.0

        # Normalize CLIP scores to [0, 1]
        clip_min, clip_max = min(clip_scores_list) if clip_scores_list else 0, max(clip_scores_list) if clip_scores_list else 1
        clip_range = clip_max - clip_min if clip_max > clip_min else 1.0

        # Compute normalized and fused scores
        for candidate in candidates:
            text_score = candidate.get("score", 0.0)
            clip_score = candidate.get("clip_score", 0.0)

            # Min-max normalization to [0, 1]
            text_norm = (text_score - text_min) / text_range if text_max > text_min else 0.5
            clip_norm = (clip_score - clip_min) / clip_range if clip_max > clip_min else 0.5

            # Weighted fusion
            fused_score = alpha * text_norm + beta * clip_norm

            candidate["text_score"] = round(text_score, 4)
            candidate["clip_score"] = round(clip_score, 4)
            candidate["text_score_normalized"] = round(text_norm, 4)
            candidate["clip_score_normalized"] = round(clip_norm, 4)
            candidate["fused_score"] = round(fused_score, 4)

        return candidates

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
            "visual": (0.05, 0.95),      # EXPERIMENT 3: Extreme CLIP with tiny text signal
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
        Stage 3: Deduplicate results (keep first occurrence of each film)

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
        # EXPERIMENT 4: CLIP-only reranking - text for diverse candidates, pure CLIP for ranking
        if query_type == "visual":
            # Large candidate pool for diversity, then pure CLIP ranking
            effective_k = 100
        elif query_type == "multi_hop":
            effective_k = self.candidate_k
        else:
            effective_k = self.candidate_k

        # Stage 1: Get candidates via text retrieval with adjusted k
        original_k = self.text_retriever.top_k
        self.text_retriever.top_k = effective_k
        candidates = self.text_retriever.retrieve(query, metadata_filter=metadata_filter)
        self.text_retriever.top_k = original_k  # Restore

        if not candidates:
            logger.warning(f"No candidates from text retrieval for: {query}")
            return []

        logger.info(f"Stage 1: Retrieved {len(candidates)} text candidates (k={effective_k})")

        # Stage 2: Get CLIP scores FOR CANDIDATE FILMS ONLY
        # FIX: Instead of searching entire image collection, compute CLIP scores
        # only for films that appear in text candidates
        clip_scores = self._get_clip_scores_for_candidates(query, candidates)

        logger.info(f"Stage 2: Computed CLIP scores for {len(clip_scores)} candidate films")

        # EXPERIMENT 4: For visual queries, use pure CLIP ranking (no fusion)
        if query_type == "visual":
            # Add CLIP scores to candidates
            for candidate in candidates:
                film_id = candidate.get("film_id")
                candidate["clip_score"] = clip_scores.get(film_id, 0.0)
                candidate["text_score"] = candidate.get("score", 0.0)
                # Use CLIP score directly as fused score (no fusion with text)
                candidate["fused_score"] = candidate["clip_score"]

            # Re-rank by pure CLIP score
            candidates_sorted = sorted(candidates, key=lambda x: x["fused_score"], reverse=True)
            logger.info(f"Visual query: ranked by pure CLIP scores (no text fusion)")
        else:
            # Get fusion weights for non-visual queries
            alpha, beta = self._get_weights(query_type)

            # Normalize scores before fusion
            candidates = self._normalize_and_fuse_scores(candidates, clip_scores, alpha, beta)

            # Re-rank by fused score
            candidates_sorted = sorted(candidates, key=lambda x: x["fused_score"], reverse=True)

        # Stage 3: Deduplicate by film_id - keep only first occurrence of each film
        seen_films = set()
        deduped_results = []
        for result in candidates_sorted:
            film_id = result.get("film_id")
            if film_id and film_id not in seen_films:
                seen_films.add(film_id)
                deduped_results.append(result)
                if len(deduped_results) >= self.top_k:
                    break

        logger.info(f"Two-stage retrieval: {len(candidates)} candidates → {len(deduped_results)} results (deduplicated)")
        return deduped_results
