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

See ARCHITECTURE.md ADR-005 for design rationale.
"""

import logging
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOP_K, RRF_K
from retrieval.text_retriever import TextRetriever
from retrieval.clip_retriever import CLIPRetriever
from retrieval.caption_retriever import CaptionRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Fuses text, caption, and CLIP retrieval results using Reciprocal Rank Fusion.

    Expected to outperform single-modality retrieval because:
    - Text retrieval handles factual/plot queries well
    - CLIP handles visual/mood queries well
    - Caption retrieval adds a secondary visual signal via text
    - RRF rewards films that appear across multiple lists

    This is the core of the research hypothesis test.
    """

    def __init__(self, top_k: int = TOP_K, rrf_k: int = RRF_K) -> None:
        """
        Initialise all three sub-retrievers.

        Args:
            top_k: Final number of fused results to return
            rrf_k: RRF constant (default 60 per literature standard)
        """
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.text_retriever = TextRetriever(top_k=top_k * 2)     # fetch more for fusion
        self.clip_retriever = CLIPRetriever(top_k=top_k * 2)
        self.caption_retriever = CaptionRetriever(top_k=top_k * 2)
        logger.info("HybridRetriever initialised with RRF fusion")

    def retrieve(
        self,
        query: str,
        metadata_filter: dict | None = None,
        use_clip: bool = True,
        use_captions: bool = True,
    ) -> list[dict]:
        """
        Retrieve and fuse results from multiple retrieval strategies.

        Args:
            query: Natural language query
            metadata_filter: Optional metadata pre-filter applied to all retrievers
            use_clip: Whether to include CLIP image retrieval
            use_captions: Whether to include caption-based retrieval

        Returns:
            Top-k fused results sorted by RRF score, each with source_lists field
            showing which retrievers contributed to the score
        """
        all_result_lists: list[list[dict]] = []
        source_labels: list[str] = []

        # Always include text retrieval
        text_results = self.text_retriever.retrieve(
            query=query,
            metadata_filter=metadata_filter,
            doc_types=["plot"],  # text-only: plots, not captions
        )
        all_result_lists.append(text_results)
        source_labels.append("text")

        # CLIP image retrieval
        if use_clip:
            clip_results = self.clip_retriever.retrieve_by_text(
                query=query,
                metadata_filter=metadata_filter,
            )
            all_result_lists.append(clip_results)
            source_labels.append("clip")

        # Caption retrieval
        if use_captions:
            caption_results = self.caption_retriever.retrieve(
                query=query,
                metadata_filter=metadata_filter,
            )
            all_result_lists.append(caption_results)
            source_labels.append("caption")

        return self._rrf_fuse(all_result_lists, source_labels)

    def _rrf_fuse(
        self,
        result_lists: list[list[dict]],
        source_labels: list[str],
    ) -> list[dict]:
        """
        Apply Reciprocal Rank Fusion across multiple ranked result lists.

        Groups results by film_id (not doc_id) so that a film appearing
        as both a text doc and a CLIP image doc is properly merged.

        Args:
            result_lists: List of ranked result lists from individual retrievers
            source_labels: Names of each retriever (for debugging/reporting)

        Returns:
            Fused and re-ranked list of top_k results
        """
        # film_id → accumulated RRF score
        rrf_scores: dict[str, float] = defaultdict(float)
        # film_id → best representative result dict
        best_results: dict[str, dict] = {}
        # film_id → list of contributing sources
        contributing_sources: dict[str, list[str]] = defaultdict(list)

        for result_list, label in zip(result_lists, source_labels):
            for rank, result in enumerate(result_list, start=1):
                film_id = result["film_id"]
                rrf_score = 1.0 / (self.rrf_k + rank)
                rrf_scores[film_id] += rrf_score
                contributing_sources[film_id].append(f"{label}@{rank}")

                # Keep the highest-scoring individual result as representative
                if film_id not in best_results or result["score"] > best_results[film_id]["score"]:
                    best_results[film_id] = result

        # Sort by RRF score descending
        ranked_film_ids = sorted(rrf_scores.keys(), key=lambda fid: rrf_scores[fid], reverse=True)

        fused = []
        for film_id in ranked_film_ids[: self.top_k]:
            result = best_results[film_id].copy()
            result["rrf_score"] = round(rrf_scores[film_id], 6)
            result["source_lists"] = contributing_sources[film_id]
            fused.append(result)

        logger.debug(
            f"RRF fusion: {sum(len(r) for r in result_lists)} total results → {len(fused)} fused"
        )
        return fused
