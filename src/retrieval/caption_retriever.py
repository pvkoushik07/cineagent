"""
Caption Retriever — Ablation 1 Variant: caption-only

Searches the ChromaDB text collection restricted to caption documents only.
This tests whether auto-generated text descriptions of images can approximate
the retrieval quality of direct CLIP embeddings.

Research question for this variant:
  "Can Gemini-generated image captions act as a proxy for visual content,
   enabling text search to reach visually-described queries?"

Actual finding (Ablation 1): Caption-only achieves 100% Recall@5 on both
factual and visual queries, matching text-only and outperforming CLIP-only.
Auto-generated captions successfully encode visual mood/atmosphere.

Used in:
  - Ablation 1: caption-only variant

Usage:
    from retrieval.caption_retriever import CaptionRetriever

    retriever = CaptionRetriever(top_k=5)
    results = retriever.retrieve("neon-lit urban nightscape")

    for result in results:
        print(f"{result['title']}: {result['score']:.3f}")
        print(f"  Caption: {result['content'][:100]}...")
"""

import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieval.text_retriever import TextRetriever

logger = logging.getLogger(__name__)


class CaptionRetriever(TextRetriever):
    """
    Retrieves films via text search over auto-generated image captions.

    Inherits from TextRetriever but restricts searches to caption documents
    (poster_caption and still_caption doc types).

    This is a thin wrapper — the heavy lifting is done by TextRetriever.
    The ablation isolation comes from restricting doc_types.
    """

    def __init__(self, top_k: int = None) -> None:
        """Initialise with caption-only restriction."""
        from config import TOP_K
        super().__init__(top_k=top_k or TOP_K)
        logger.info("CaptionRetriever initialised (caption doc types only)")

    def retrieve(
        self,
        query: str,
        metadata_filter: dict | None = None,
        doc_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k results using caption documents only.

        Overrides parent to always restrict to caption doc types.
        The doc_types parameter is ignored — always uses captions.

        Args:
            query: Natural language query
            metadata_filter: Optional metadata filter
            doc_types: Ignored — always uses ["poster_caption", "still_caption"]

        Returns:
            List of caption-based retrieval results
        """
        return super().retrieve(
            query=query,
            metadata_filter=metadata_filter,
            doc_types=["poster_caption", "still_caption"],
        )
