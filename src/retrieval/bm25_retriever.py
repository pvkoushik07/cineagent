"""
BM25 Sparse Retrieval — keyword-based search for thematic queries.

Uses BM25Okapi algorithm for sparse retrieval, complements dense embeddings
by providing strong keyword matching (e.g., "social commentary" → exact match).

This is particularly effective for thematic multi-hop queries where MiniLM
embeddings prioritize semantic context over keyword presence.

Usage:
    retriever = BM25Retriever()
    results = retriever.retrieve("dark social commentary", k=10)
"""

import logging
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_PERSIST_DIR, TEXT_COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Sparse keyword-based retrieval using BM25 algorithm.

    Complements dense retrieval (MiniLM) by providing strong keyword matching.
    Particularly useful for thematic queries where specific terms matter
    (e.g., "social commentary", "class struggle").
    """

    def __init__(self):
        """Initialize BM25 retriever by loading and indexing all text documents."""
        logger.info("Initializing BM25Retriever...")

        # Load all documents from ChromaDB text collection
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(TEXT_COLLECTION_NAME)

        # Get all documents with metadata
        all_docs = collection.get(include=["documents", "metadatas"])

        self.documents = all_docs["documents"]
        self.metadatas = all_docs["metadatas"]
        self.ids = all_docs["ids"]

        # Tokenize documents for BM25 (lowercase + split on whitespace)
        logger.info(f"Tokenizing {len(self.documents)} documents for BM25...")
        tokenized_docs = [doc.lower().split() for doc in self.documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)

        logger.info(f"BM25Retriever initialized with {len(self.documents)} documents")

    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        """
        Retrieve top-k documents using BM25 sparse scoring.

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            List of dicts with doc_id, film_id, content, metadata, score
        """
        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices by score (descending)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        # Build result list
        results = []
        for idx in top_indices:
            results.append({
                "doc_id": self.ids[idx],
                "film_id": self.metadatas[idx].get("film_id", ""),
                "content": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(scores[idx]),
            })

        logger.debug(f"BM25 retrieved {len(results)} results for query: {query[:50]}...")
        return results


if __name__ == "__main__":
    # Test BM25 retriever
    retriever = BM25Retriever()

    # Test query: should find Parasite with high score
    test_query = "dark social commentary class struggle"
    results = retriever.retrieve(test_query, k=5)

    print(f"\n🔍 BM25 Test Query: '{test_query}'")
    print(f"📊 Top 5 Results:\n")

    for i, result in enumerate(results, 1):
        title = result["metadata"].get("title", "Unknown")
        score = result["score"]
        content_preview = result["content"][:100]

        print(f"{i}. {title} (score: {score:.2f})")
        print(f"   {content_preview}...\n")
