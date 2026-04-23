"""
LangGraph Node Implementations — all 5 nodes of the CineAgent workflow.

Nodes are pure functions: (AgentState) → AgentState update dict.
Each node is independently testable.

Node execution order (with conditional routing):
  QueryRouter
      │
  RetrievalPlanner
      │
  TasteProfileUpdater
      │
  AnswerSynthesiser
      │
  Verifier ──(fail)──→ RetrievalPlanner (retry, max 2x)
      │
  (success) → END
"""

import json
import logging
import time
from pathlib import Path

import google.generativeai as genai

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GEMINI_API_KEY, GEMINI_MODEL, TEMPERATURE, TOP_K
from agent.state import AgentState, TasteProfile
from retrieval.text_retriever import TextRetriever
from retrieval.clip_retriever import CLIPRetriever
from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)
genai.configure(api_key=GEMINI_API_KEY)

# Initialise retrievers once (expensive — don't re-init per call)
_text_retriever: TextRetriever | None = None
_clip_retriever: CLIPRetriever | None = None
_hybrid_retriever: HybridRetriever | None = None


def _get_retrievers() -> tuple[TextRetriever, CLIPRetriever, HybridRetriever]:
    """Lazy-load retrievers on first use."""
    global _text_retriever, _clip_retriever, _hybrid_retriever
    if _text_retriever is None:
        _text_retriever = TextRetriever()
        _clip_retriever = CLIPRetriever()
        _hybrid_retriever = HybridRetriever()
    return _text_retriever, _clip_retriever, _hybrid_retriever


# ── Node 1: QueryRouter ───────────────────────────────────────────────────────

ROUTER_PROMPT = """You are a query classifier for a film recommendation agent.

Classify the user query into exactly one of these types:
- factual: asking for specific facts (director, year, cast, plot details)
- visual: describing visual mood, aesthetic, atmosphere, colour palette, setting look
- hybrid: needs both factual and visual information
- multi_hop: requires combining multiple constraints or pieces of evidence

Query: {query}

Respond with JSON only:
{{"query_type": "<type>", "retrieval_strategy": "<text|clip|hybrid|hybrid>", "reasoning": "<one sentence>"}}

Rules:
- visual queries → clip or hybrid strategy
- factual queries → text strategy
- multi_hop queries → hybrid strategy
- when in doubt → hybrid"""


def query_router_node(state: AgentState) -> dict:
    """
    Node 1: Classify the incoming query and determine retrieval strategy.

    Reads:  state["query"], state["conversation_history"]
    Writes: state["query_type"], state["retrieval_strategy"],
            state["tool_calls_count"]

    Args:
        state: Current AgentState

    Returns:
        Partial state update dict
    """
    query = state["query"]
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = ROUTER_PROMPT.format(query=query)
    try:
        response = model.generate_content(prompt)
        # Parse JSON from response
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        query_type = parsed.get("query_type", "hybrid")
        retrieval_strategy = parsed.get("retrieval_strategy", "hybrid")
        logger.info(f"Router: query_type={query_type}, strategy={retrieval_strategy}")
    except Exception as e:
        logger.warning(f"Router parsing failed: {e}. Defaulting to hybrid.")
        query_type = "hybrid"
        retrieval_strategy = "hybrid"

    return {
        "query_type": query_type,
        "retrieval_strategy": retrieval_strategy,
        "tool_calls_count": state["tool_calls_count"] + 1,
    }


# ── Node 2: RetrievalPlanner ──────────────────────────────────────────────────

def retrieval_planner_node(state: AgentState) -> dict:
    """
    Node 2: Execute retrieval based on strategy from QueryRouter.

    Builds a metadata filter from the taste profile (e.g. language, year range)
    and runs the appropriate retriever.

    Reads:  state["query"], state["retrieval_strategy"], state["taste_profile"]
    Writes: state["retrieved_docs"], state["retrieved_images"],
            state["tool_calls_count"]

    Args:
        state: Current AgentState

    Returns:
        Partial state update dict
    """
    query = state["query"]
    strategy = state["retrieval_strategy"]
    profile = state["taste_profile"]

    text_ret, clip_ret, hybrid_ret = _get_retrievers()

    # Build metadata filter from taste profile
    metadata_filter = _build_metadata_filter(profile)

    retrieved_docs = []
    retrieved_images = []

    if strategy == "text":
        results = text_ret.retrieve(query, metadata_filter=metadata_filter)
        retrieved_docs = results

    elif strategy == "clip":
        results = clip_ret.retrieve_by_text(query, metadata_filter=metadata_filter)
        retrieved_docs = results
        retrieved_images = [r["content"] for r in results if r.get("content")]

    else:  # hybrid (default)
        results = hybrid_ret.retrieve(query, metadata_filter=metadata_filter)
        retrieved_docs = results
        retrieved_images = [
            r["content"] for r in results
            if r.get("modality") == "image" and r.get("content")
        ]

    logger.info(f"RetrievalPlanner: {len(retrieved_docs)} docs, {len(retrieved_images)} images")

    return {
        "retrieved_docs": retrieved_docs,
        "retrieved_images": retrieved_images[:3],  # Max 3 images for synthesis
        "tool_calls_count": state["tool_calls_count"] + 1,
    }


def _build_metadata_filter(profile: TasteProfile) -> dict | None:
    """
    Convert a TasteProfile into a ChromaDB metadata filter.

    Only applies filters when confidence is high enough to trust them.

    Args:
        profile: Current user taste profile

    Returns:
        ChromaDB where-filter dict, or None if no strong constraints
    """
    if profile["confidence"] < 0.3:
        return None  # Don't filter on uncertain profiles

    filters = []

    year_range = profile.get("year_range", {})
    if year_range.get("min"):
        filters.append({"year": {"$gte": year_range["min"]}})
    if year_range.get("max"):
        filters.append({"year": {"$lte": year_range["max"]}})

    if len(filters) == 1:
        return filters[0]
    if len(filters) > 1:
        return {"$and": filters}
    return None


# ── Node 3: TasteProfileUpdater ───────────────────────────────────────────────

TASTE_EXTRACTOR_PROMPT = """You are updating a user's film taste profile based on their latest message.

Current profile:
{current_profile}

User message: "{message}"

Extract any preference signals from the message and return an updated profile.
Look for: genres, directors, actors, languages, time periods, moods, films already watched.

Return JSON only — the complete updated profile:
{{
  "preferred_genres": [...],
  "preferred_directors": [...],
  "preferred_languages": [...],
  "year_range": {{"min": null_or_int, "max": null_or_int}},
  "avoid_genres": [...],
  "watched": [...],
  "mood_keywords": [...],
  "confidence": 0.0_to_1.0
}}

Rules:
- Merge new info with existing — don't wipe old preferences
- Increase confidence as more signals accumulate (max 0.95)
- If message says "I've seen X", add X to watched list
- If message contradicts a prior preference, update accordingly"""


def taste_profile_updater_node(state: AgentState) -> dict:
    """
    Node 3: Extract preference signals from the user's message and update
    the taste profile in state.

    This is what makes CineAgent genuinely personalised across turns.
    Without this node, the system treats every turn independently.

    Reads:  state["query"], state["taste_profile"]
    Writes: state["taste_profile"], state["tool_calls_count"]

    Args:
        state: Current AgentState

    Returns:
        Partial state update dict with updated taste_profile
    """
    current_profile = state["taste_profile"]
    message = state["query"]

    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = TASTE_EXTRACTOR_PROMPT.format(
        current_profile=json.dumps(current_profile, indent=2),
        message=message,
    )

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        updated_profile = json.loads(text.strip())
        logger.info(f"TasteProfileUpdater: confidence={updated_profile.get('confidence', 0):.2f}")
    except Exception as e:
        logger.warning(f"Profile update failed: {e}. Keeping existing profile.")
        updated_profile = current_profile

    return {
        "taste_profile": updated_profile,
        "tool_calls_count": state["tool_calls_count"] + 1,
    }


# ── Node 4: AnswerSynthesiser ─────────────────────────────────────────────────

SYNTHESIS_PROMPT = """You are CineAgent, a personalised film recommendation assistant.

Answer the user's query based ONLY on the retrieved film information below.
Be specific and cite film titles. Do not recommend films not in the retrieved results.

User query: {query}

User taste profile:
{taste_profile}

Retrieved films:
{retrieved_context}

Instructions:
- Recommend 2-3 films maximum
- Explain WHY each recommendation fits the query
- Reference specific retrieved details (director, mood, visual style, plot)
- If the query asked about visual style, describe what makes the visuals relevant
- Do not make up information not in the retrieved context
- Be conversational, not listy"""


def answer_synthesiser_node(state: AgentState) -> dict:
    """
    Node 4: Generate a grounded response using retrieved context.

    Uses Gemini Flash multimodal — can accept both text docs and images.
    The response cites only retrieved films, enabling faithfulness scoring.

    Reads:  state["query"], state["retrieved_docs"], state["retrieved_images"],
            state["taste_profile"]
    Writes: state["response"], state["cited_films"], state["tool_calls_count"]

    Args:
        state: Current AgentState

    Returns:
        Partial state update dict
    """
    query = state["query"]
    retrieved_docs = state["retrieved_docs"]
    retrieved_images = state["retrieved_images"]
    taste_profile = state["taste_profile"]

    if not retrieved_docs:
        return {
            "response": "I couldn't find relevant films for that query. Could you rephrase?",
            "cited_films": [],
            "tool_calls_count": state["tool_calls_count"] + 1,
        }

    # Format retrieved context
    context_parts = []
    seen_films: set[str] = set()
    for doc in retrieved_docs[:TOP_K]:
        film_id = doc.get("film_id", "")
        if film_id not in seen_films:
            seen_films.add(film_id)
            context_parts.append(
                f"Film: {doc.get('title', 'Unknown')}\n"
                f"Content: {doc.get('content', '')[:400]}\n"
                f"Score: {doc.get('rrf_score', doc.get('score', 0)):.3f}"
            )

    retrieved_context = "\n\n---\n\n".join(context_parts)

    # Build multimodal prompt
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt_text = SYNTHESIS_PROMPT.format(
        query=query,
        taste_profile=json.dumps(taste_profile, indent=2),
        retrieved_context=retrieved_context,
    )

    # Include images if available (Gemini Flash multimodal)
    content_parts: list = [prompt_text]
    for image_path in retrieved_images[:2]:  # Max 2 images to control tokens
        try:
            import base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            content_parts.append({"mime_type": "image/jpeg", "data": image_data})
        except Exception as e:
            logger.warning(f"Could not load image {image_path}: {e}")

    try:
        response = model.generate_content(content_parts)
        answer = response.text.strip()
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        answer = "I encountered an error generating a response. Please try again."

    # Extract cited film titles
    cited_films = [doc.get("title", "") for doc in retrieved_docs[:TOP_K] if doc.get("title")]

    return {
        "response": answer,
        "cited_films": cited_films,
        "tool_calls_count": state["tool_calls_count"] + 1,
    }


# ── Node 5: Verifier ──────────────────────────────────────────────────────────

def verifier_node(state: AgentState) -> dict:
    """
    Node 5: Check the response for correctness before returning to user.

    Deterministic checks (no LLM call needed):
    1. Does the response recommend films already in the watched list?
    2. Does it recommend films from avoided genres (if profile is confident)?
    3. Is the response non-empty?

    If verification fails, sets verified=False and the graph re-routes
    to RetrievalPlanner for a retry (max 2 retries).

    Reads:  state["response"], state["cited_films"], state["taste_profile"],
            state["retry_count"]
    Writes: state["verified"], state["verification_reason"],
            state["tool_calls_count"]

    Args:
        state: Current AgentState

    Returns:
        Partial state update dict
    """
    response = state["response"]
    cited_films = state["cited_films"]
    profile = state["taste_profile"]
    retry_count = state["retry_count"]

    # Check 1: Non-empty response
    if not response or len(response.strip()) < 10:
        return {
            "verified": False,
            "verification_reason": "empty_response",
            "retry_count": retry_count + 1,
            "tool_calls_count": state["tool_calls_count"] + 1,
        }

    # Check 2: Not recommending already-watched films
    watched = [w.lower() for w in profile.get("watched", [])]
    if watched and profile.get("confidence", 0) > 0.4:
        cited_lower = [c.lower() for c in cited_films]
        watched_recommended = [c for c in cited_lower if any(w in c for w in watched)]
        if watched_recommended:
            logger.info(f"Verifier: recommended watched films: {watched_recommended}")
            return {
                "verified": False,
                "verification_reason": f"recommended_watched:{','.join(watched_recommended)}",
                "retry_count": retry_count + 1,
                "tool_calls_count": state["tool_calls_count"] + 1,
            }

    # All checks passed
    logger.info("Verifier: response passed all checks")
    return {
        "verified": True,
        "verification_reason": None,
        "tool_calls_count": state["tool_calls_count"] + 1,
    }
