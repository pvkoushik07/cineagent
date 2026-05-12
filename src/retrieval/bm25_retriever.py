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

    def retrieve(self, query: str, k: int = 10, metadata_filter: dict | None = None) -> list[dict]:
        """
        Retrieve top-k documents using BM25 sparse scoring.

        Args:
            query: Search query string
            k: Number of results to return
            metadata_filter: Optional ChromaDB-style metadata filter (applied post-retrieval)

        Returns:
            List of dicts with doc_id, film_id, content, metadata, score
        """
        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top indices by score (need more for post-filtering)
        retrieve_k = k * 10 if metadata_filter else k
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:retrieve_k]

        # Build result list
        results = []
        for idx in top_indices:
            result = {
                "doc_id": self.ids[idx],
                "film_id": self.metadatas[idx].get("film_id", ""),
                "content": self.documents[idx],
                "metadata": self.metadatas[idx],
                "score": float(scores[idx]),
            }
            results.append(result)

        # Apply metadata filtering if provided
        if metadata_filter:
            results = self._apply_metadata_filter(results, metadata_filter)
            logger.debug(f"BM25 filtered to {len(results)} results")

        # Trim to requested k
        results = results[:k]

        logger.debug(f"BM25 retrieved {len(results)} results for query: {query[:50]}...")
        return results

    def _apply_metadata_filter(self, results: list[dict], filter_dict: dict) -> list[dict]:
        """Apply ChromaDB-style metadata filter to results."""
        # Handle $and operator
        if "$and" in filter_dict:
            conditions = filter_dict["$and"]
            filtered = results
            for condition in conditions:
                filtered = self._apply_metadata_filter(filtered, condition)
            return filtered

        # Apply individual filters
        filtered_results = []
        for result in results:
            metadata = result["metadata"]
            matches = True

            for key, value in filter_dict.items():
                if key.startswith("$"):
                    continue  # Skip operators

                if isinstance(value, dict):
                    # Handle operators like $gte, $ne, etc.
                    metadata_value = metadata.get(key)
                    for op, op_value in value.items():
                        if op == "$gte" and not (metadata_value and metadata_value >= op_value):
                            matches = False
                        elif op == "$lte" and not (metadata_value and metadata_value <= op_value):
                            matches = False
                        elif op == "$ne" and metadata_value == op_value:
                            matches = False
                        elif op == "$in" and metadata_value not in op_value:
                            matches = False
                else:
                    # Direct equality
                    if metadata.get(key) != value:
                        matches = False

            if matches:
                filtered_results.append(result)

        return filtered_results


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
