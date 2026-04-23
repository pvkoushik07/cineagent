"""
AgentState — the single shared state object passed between all LangGraph nodes.

Every node reads from and writes to this state. TypedDict is the LangGraph
native pattern — do not convert to Pydantic without updating ARCHITECTURE.md.
"""

from typing import TypedDict, Optional


class TasteProfile(TypedDict):
    """
    Structured representation of user's film preferences.
    Updated by TasteProfileUpdater node on each turn.
    """
    preferred_genres: list[str]
    preferred_directors: list[str]
    preferred_languages: list[str]      # e.g. ["non-English", "French"]
    year_range: dict                    # {"min": int|None, "max": int|None}
    avoid_genres: list[str]
    watched: list[str]                  # film IDs or titles already seen
    mood_keywords: list[str]            # e.g. ["slow-burn", "bleak", "cerebral"]
    confidence: float                   # 0.0-1.0: how confident in the profile


class RetrievedDocument(TypedDict):
    """A single retrieved document from ChromaDB."""
    doc_id: str
    film_id: str
    film_title: str
    modality: str                       # "text" | "poster" | "still" | "caption"
    content: str                        # text content or image path
    score: float                        # similarity score
    metadata: dict


class AgentState(TypedDict):
    """
    Full state of the CineAgent across a conversation turn.
    All nodes read from and write to this dict.
    """
    # ── Input ────────────────────────────────────────────────────────────────
    query: str
    conversation_history: list[dict]    # [{"role": "user"|"assistant", "content": str}]

    # ── Router Output ─────────────────────────────────────────────────────────
    query_type: str                     # "factual" | "visual" | "hybrid" | "multi_hop"
    retrieval_strategy: str             # "text" | "clip" | "caption" | "hybrid"

    # ── Retrieval Output ──────────────────────────────────────────────────────
    retrieved_docs: list[RetrievedDocument]
    retrieved_images: list[str]         # paths to image files for multimodal synthesis

    # ── Taste Profile (persists across turns) ────────────────────────────────
    taste_profile: TasteProfile

    # ── Synthesis Output ──────────────────────────────────────────────────────
    response: str
    cited_films: list[str]              # film titles mentioned in response

    # ── Verification ──────────────────────────────────────────────────────────
    verified: bool
    verification_reason: Optional[str]  # why verification failed, if it did
    retry_count: int                    # prevent infinite re-routing loops

    # ── Evaluation Instrumentation ────────────────────────────────────────────
    tool_calls_count: int
    latency_ms: float


def empty_taste_profile() -> TasteProfile:
    """Return a blank taste profile for a new conversation."""
    return TasteProfile(
        preferred_genres=[],
        preferred_directors=[],
        preferred_languages=[],
        year_range={"min": None, "max": None},
        avoid_genres=[],
        watched=[],
        mood_keywords=[],
        confidence=0.0,
    )


def initial_state(query: str) -> AgentState:
    """Return an initial AgentState for the first turn of a conversation."""
    return AgentState(
        query=query,
        conversation_history=[],
        query_type="",
        retrieval_strategy="",
        retrieved_docs=[],
        retrieved_images=[],
        taste_profile=empty_taste_profile(),
        response="",
        cited_films=[],
        verified=False,
        verification_reason=None,
        retry_count=0,
        tool_calls_count=0,
        latency_ms=0.0,
    )
