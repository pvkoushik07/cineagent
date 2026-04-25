"""
CLIP Retriever — Ablation 1 Variant: CLIP-only

Searches the ChromaDB image collection using CLIP embeddings.
Encodes the text query into CLIP's shared text-image embedding space,
then retrieves visually similar posters and scene stills.

Originally hypothesized as the KEY modality for cross-modal visual queries.

Actual finding (Ablation 1): CLIP-only achieves only 20% Recall@5 on visual
queries, dramatically underperforming text-based methods (100%). CLIP's text
encoder appears weaker than MiniLM for mapping abstract mood descriptions to
visual content. Text embeddings + rich captions outperform true multimodal
embeddings in this KB.

Used in:
  - Ablation 1: CLIP-only variant
  - Variant B (fixed RAG): as the image component
  - Variant C (full agent): as the clip_search tool

Usage:
    from retrieval.clip_retriever import CLIPRetriever

    retriever = CLIPRetriever(top_k=5)
    results = retriever.retrieve("cold, desaturated, rain-soaked atmosphere")

    for result in results:
        print(f"{result['title']}: {result['score']:.3f}")
        print(f"  Image: {result['content']}")
"""

import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIR,
    IMAGE_COLLECTION_NAME,
    IMAGE_EMBEDDING_MODEL,
    TOP_K,
)

logger = logging.getLogger(__name__)


class CLIPRetriever:
    """
    Retrieves film images from the image index using CLIP cross-modal embeddings.

    CLIP encodes text and images into the same vector space, enabling
    text queries to retrieve visually matching images without any
    textual metadata on those images.

    Strength: visual/mood/aesthetic queries
    Weakness: precise factual queries (director names, dates)
    """

    def __init__(self, top_k: int = TOP_K) -> None:
        """
        Initialise the CLIP retriever.

        Args:
            top_k: Number of results to return per query
        """
        self.top_k = top_k
        # CLIP model handles both text and image encoding
        self.model = SentenceTransformer(IMAGE_EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_collection(IMAGE_COLLECTION_NAME)
        logger.info(f"CLIPRetriever initialised. Collection size: {self.collection.count()}")

    def retrieve(
        self,
        query: str,
        image_types: list[str] | None = None,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        Retrieve images matching a text query (convenience method).

        Alias for retrieve_by_text(). Use this for standard text queries.

        Args:
            query: Natural language description of desired visual content
            image_types: Optional filter for image type
            metadata_filter: Optional metadata filters

        Returns:
            List of result dicts with image paths and similarity scores
        """
        return self.retrieve_by_text(query, image_types, metadata_filter)

    def retrieve_by_text(
        self,
        query: str,
        image_types: list[str] | None = None,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        Retrieve images matching a text query via CLIP cross-modal search.

        Args:
            query: Natural language description of desired visual content
            image_types: Optional filter for image type: ["poster"] or ["still"]
                         or None for both
            metadata_filter: Optional ChromaDB where-filter for metadata fields

        Returns:
            List of result dicts with image paths and similarity scores
        """
        # CLIP encodes text and images in the same space — encode query as text
        query_embedding = self.model.encode(query).tolist()

        where = {}
        if image_types:
            where["image_type"] = {"$in": image_types}
        if metadata_filter:
            where.update(metadata_filter)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def retrieve_by_image(self, image_path: str) -> list[dict]:
        """
        Retrieve images visually similar to a given image (image-to-image search).

        Args:
            image_path: Path to query image file

        Returns:
            List of visually similar image results
        """
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        image_embedding = self.model.encode(img).tolist()

        results = self.collection.query(
            query_embeddings=[image_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def _format_results(self, raw: dict) -> list[dict]:
        """
        Convert ChromaDB query output to standardised result format.

        Args:
            raw: Raw ChromaDB query response

        Returns:
            List of standardised result dicts
        """
        formatted = []
        ids = raw["ids"][0]
        docs = raw["documents"][0]    # image paths stored as documents
        metas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for doc_id, image_path, metadata, distance in zip(ids, docs, metas, distances):
            score = 1 - distance

            formatted.append({
                "doc_id": doc_id,
                "film_id": metadata.get("film_id", ""),
                "title": metadata.get("title", ""),
                "modality": "image",
                "image_type": metadata.get("image_type", ""),
                "content": image_path,   # path to the image file
                "score": round(score, 4),
                "metadata": metadata,
            })

        return formatted
