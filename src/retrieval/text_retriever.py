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
"""

import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIR,
    TEXT_COLLECTION_NAME,
    TEXT_EMBEDDING_MODEL,
    TOP_K,
)

logger = logging.getLogger(__name__)


class TextRetriever:
    """
    Retrieves film documents from the text index using semantic similarity.

    Searches plot summaries and reviews. Strong for factual queries
    (director, cast, plot details). Weak for visual/mood queries.
    """

    def __init__(self, top_k: int = TOP_K) -> None:
        """
        Initialise the text retriever.

        Args:
            top_k: Number of results to return per query
        """
        self.top_k = top_k
        self.model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_collection(TEXT_COLLECTION_NAME)
        logger.info(f"TextRetriever initialised. Collection size: {self.collection.count()}")

    def retrieve(
        self,
        query: str,
        metadata_filter: dict | None = None,
        doc_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k text documents for a query.

        Args:
            query: Natural language query string
            metadata_filter: Optional ChromaDB where-filter,
                             e.g. {"year": {"$gte": 2000}}
            doc_types: Optional list of doc_type values to restrict to,
                       e.g. ["plot"] to exclude captions

        Returns:
            List of result dicts with keys:
              doc_id, film_id, title, modality, content, score, metadata
        """
        query_embedding = self.model.encode(query).tolist()

        where = {}
        if metadata_filter:
            where.update(metadata_filter)
        if doc_types:
            where["doc_type"] = {"$in": doc_types}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            where=where if where else None,
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
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for doc_id, content, metadata, distance in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 = identical, -1 = opposite
            score = 1 - distance

            formatted.append({
                "doc_id": doc_id,
                "film_id": metadata.get("film_id", ""),
                "title": metadata.get("title", ""),
                "modality": "text",
                "content": content,
                "score": round(score, 4),
                "metadata": metadata,
            })

        return formatted
