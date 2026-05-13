"""
LangChain Tool Wrappers for Retrieval Functions

Provides tool wrappers that expose retrieval functions to the LangGraph agent.
These tools are used by the RetrievalPlanner node for dynamic tool selection.
"""

from typing import List
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.text_retriever import TextRetriever
from retrieval.clip_retriever import CLIPRetriever
from retrieval.hybrid_retriever import HybridRetriever


# ── Retriever Singletons ──────────────────────────────────────────────────────
# Initialize retrievers once to avoid expensive re-initialization

_text_retriever: TextRetriever | None = None
_clip_retriever: CLIPRetriever | None = None
_hybrid_retriever: HybridRetriever | None = None


def _get_text_retriever() -> TextRetriever:
    """Get or initialize text retriever singleton."""
    global _text_retriever
    if _text_retriever is None:
        _text_retriever = TextRetriever()
    return _text_retriever


def _get_clip_retriever() -> CLIPRetriever:
    """Get or initialize CLIP retriever singleton."""
    global _clip_retriever
    if _clip_retriever is None:
        _clip_retriever = CLIPRetriever()
    return _clip_retriever


def _get_hybrid_retriever() -> HybridRetriever:
    """Get or initialize hybrid retriever singleton."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


# ── Tool Functions ────────────────────────────────────────────────────────────
# These functions can be wrapped by LangChain @tool decorator or called directly


def text_search(query: str, k: int = 10, filters: dict | None = None) -> List[dict]:
    """
    Search for films using text embeddings (plot, reviews, captions).

    Best for: Factual queries, genre/theme searches, director/actor queries.

    Args:
        query: Natural language search query
        k: Number of results to return (default: 10)
        filters: Optional metadata filters (e.g., {"year": {"$gte": 2010}})

    Returns:
        List of dicts with film metadata and relevance scores

    Example:
        >>> text_search("psychological thriller with twist ending", k=5)
        [{"film_id": "550", "title": "Fight Club", "score": 0.89, ...}, ...]
    """
    retriever = _get_text_retriever()
    return retriever.retrieve(query, k=k, filters=filters)


def clip_search(query: str, k: int = 10, filters: dict | None = None) -> List[dict]:
    """
    Search for films using CLIP image embeddings (posters + scene stills).

    Best for: Visual aesthetic queries, mood/atmosphere, color palette, cinematography.

    Args:
        query: Visual description query (e.g., "cold desaturated urban atmosphere")
        k: Number of results to return (default: 10)
        filters: Optional metadata filters

    Returns:
        List of dicts with film metadata and CLIP similarity scores

    Example:
        >>> clip_search("neon-lit urban nightscape wet streets", k=5)
        [{"film_id": "77338", "title": "Drive", "score": 0.76, ...}, ...]
    """
    retriever = _get_clip_retriever()
    return retriever.retrieve(query, k=k, filters=filters)


def hybrid_search(
    query: str,
    k: int = 10,
    filters: dict | None = None,
    weights: dict | None = None
) -> List[dict]:
    """
    Search using Reciprocal Rank Fusion (RRF) across text, CLIP, and BM25.

    Best for: Queries mixing factual + visual + thematic elements.
    Combines text embeddings, CLIP image search, and BM25 keyword matching.

    Args:
        query: Search query (can mix factual and visual aspects)
        k: Number of results to return (default: 10)
        filters: Optional metadata filters
        weights: Optional weights for different retrievers
                 (e.g., {"text": 0.4, "clip": 0.3, "bm25": 0.3})

    Returns:
        List of dicts with film metadata and fused scores

    Example:
        >>> hybrid_search("dark social commentary, non-English, after 2010", k=5,
        ...               filters={"language": {"$ne": "English"}})
        [{"film_id": "496243", "title": "Parasite", "score": 0.92, ...}, ...]
    """
    retriever = _get_hybrid_retriever()
    return retriever.retrieve(query, k=k, filters=filters, weights=weights)


# ── Tool Metadata for LangChain Integration ───────────────────────────────────

TOOL_DESCRIPTIONS = {
    "text_search": {
        "name": "text_search",
        "description": "Search films using text embeddings (plot, reviews, themes). "
                      "Best for factual queries, genres, directors, themes.",
        "parameters": {
            "query": "Natural language search query",
            "k": "Number of results (default: 10)",
            "filters": "Optional metadata filters (dict)"
        }
    },
    "clip_search": {
        "name": "clip_search",
        "description": "Search films using CLIP image embeddings (visual aesthetics). "
                      "Best for mood, atmosphere, color palette, cinematography.",
        "parameters": {
            "query": "Visual description query",
            "k": "Number of results (default: 10)",
            "filters": "Optional metadata filters (dict)"
        }
    },
    "hybrid_search": {
        "name": "hybrid_search",
        "description": "Search using RRF fusion (text + CLIP + BM25). "
                      "Best for queries mixing factual, visual, and thematic elements.",
        "parameters": {
            "query": "Search query (factual + visual)",
            "k": "Number of results (default: 10)",
            "filters": "Optional metadata filters (dict)",
            "weights": "Optional retriever weights (dict)"
        }
    }
}


# ── LangChain Tool Wrappers (Optional) ────────────────────────────────────────
# Uncomment if using LangChain's @tool decorator

# from langchain.tools import tool
#
# @tool
# def text_search_tool(query: str, k: int = 10) -> List[dict]:
#     """Text-based film search using plot and review embeddings."""
#     return text_search(query, k)
#
# @tool
# def clip_search_tool(query: str, k: int = 10) -> List[dict]:
#     """Visual film search using CLIP image embeddings."""
#     return clip_search(query, k)
#
# @tool
# def hybrid_search_tool(query: str, k: int = 10) -> List[dict]:
#     """Hybrid film search using RRF fusion."""
#     return hybrid_search(query, k)


if __name__ == "__main__":
    # Demo usage
    print("=== CineAgent Tool Wrappers Demo ===\n")

    # Example 1: Text search
    print("1. Text search: 'psychological thriller'")
    results = text_search("psychological thriller with twist ending", k=3)
    for r in results:
        print(f"   - {r.get('title', 'Unknown')} (score: {r.get('score', 0):.2f})")

    print("\n2. CLIP search: 'neon urban atmosphere'")
    results = clip_search("neon-lit urban nightscape", k=3)
    for r in results:
        print(f"   - {r.get('title', 'Unknown')} (score: {r.get('score', 0):.2f})")

    print("\n3. Hybrid search: 'dark social commentary, non-English'")
    results = hybrid_search("dark social commentary, non-English, recent", k=3)
    for r in results:
        print(f"   - {r.get('title', 'Unknown')} (score: {r.get('score', 0):.2f})")
