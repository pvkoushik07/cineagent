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

def parse_json_safe(text: str) -> dict:
    """
    Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        text: Raw LLM response text that may contain JSON
        
    Returns:
        Parsed dict, or empty dict if parsing fails
    """
    try:
        # Strip markdown code blocks if present
        if "```" in text:
            parts = text.split("```")
            # Take the content between first pair of ```
            if len(parts) >= 2:
                text = parts[1]
                # Remove 'json' language identifier if present
                if text.strip().startswith("json"):
                    text = text.strip()[4:]
                text = text.strip()
        
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {text[:200]}... Error: {e}")
        return {}
    
def format_retrieved_docs(docs: list[dict], max_docs: int = 5) -> str:
    """
    Format retrieved documents for LLM prompts.
    
    Args:
        docs: List of retrieved document dicts
        max_docs: Maximum number of docs to include
        
    Returns:
        Formatted string with doc content
    """
    if not docs:
        return "No documents retrieved."
    
    formatted_lines = []
    for i, doc in enumerate(docs[:max_docs], 1):
        formatted_lines.append(f"[{i}] {doc.get('title', 'Unknown')} (Film ID: {doc.get('film_id', 'N/A')})")
        formatted_lines.append(f"    Content: {doc.get('content', '')[:300]}...")
        formatted_lines.append("")
    
    return "\n".join(formatted_lines)

def format_taste_profile(profile: dict) -> str:
    """
    Format taste profile dict for LLM prompts.
    
    Args:
        profile: TasteProfile dict
        
    Returns:
        Human-readable formatted string
    """
    lines = []
    if profile.get("preferred_genres"):
        lines.append(f"Preferred genres: {', '.join(profile['preferred_genres'])}")
    if profile.get("preferred_directors"):
        lines.append(f"Preferred directors: {', '.join(profile['preferred_directors'])}")
    if profile.get("preferred_languages"):
        lines.append(f"Preferred languages: {', '.join(profile['preferred_languages'])}")
    if profile.get("mood_keywords"):
        lines.append(f"Mood preferences: {', '.join(profile['mood_keywords'])}")
    if profile.get("avoid_genres"):
        lines.append(f"Avoid: {', '.join(profile['avoid_genres'])}")
    if profile.get("watched"):
        lines.append(f"Already watched: {', '.join(profile['watched'][:5])}")
        if len(profile["watched"]) > 5:
            lines.append(f"  (and {len(profile['watched']) - 5} more)")
    
    year_range = profile.get("year_range", {})
    if year_range.get("min") or year_range.get("max"):
        min_year = year_range.get("min", "any")
        max_year = year_range.get("max", "any")
        lines.append(f"Year range: {min_year} - {max_year}")
    
    lines.append(f"Confidence: {profile.get('confidence', 0.0):.2f}")
    
    return "\n".join(lines) if lines else "No preferences recorded yet."


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

    Reads:  state["query"]
    Writes: state["query_type"], state["retrieval_strategy"], state["tool_calls_count"]

    Args:
        state: Current AgentState

    Returns:
        Partial state update dict
    """
    query = state["query"]

    prompt = f"""You are a query classifier for a film recommendation agent.

Classify the user query into exactly one of these types:
- factual: asking for specific facts (director, year, cast, plot details)
- visual: describing visual mood, aesthetic, atmosphere, color palette
- hybrid: needs both factual and visual information
- multi_hop: requires combining multiple constraints

Query: {query}

Respond with JSON only:
{{"query_type": "<type>", "retrieval_strategy": "text", "reasoning": "<one sentence>"}}

Note: retrieval_strategy is always "text" (empirically best from ablation study).
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        result = parse_json_safe(response.text)

        query_type = result.get("query_type", "hybrid")
        # Ensure valid query type
        if query_type not in ("factual", "visual", "hybrid", "multi_hop"):
            logger.warning(f"Invalid query_type '{query_type}', defaulting to 'hybrid'")
            query_type = "hybrid"

        logger.info(f"QueryRouter: classified as '{query_type}' - {result.get('reasoning', '')}")

        return {
            "query_type": query_type,
            "retrieval_strategy": "text",  # Always text (Phase 2 finding)
            "tool_calls_count": state["tool_calls_count"] + 1
        }

    except Exception as e:
        logger.error(f"QueryRouter LLM call failed: {e}")
        # Fallback to safe default
        return {
            "query_type": "hybrid",
            "retrieval_strategy": "text",
            "tool_calls_count": state["tool_calls_count"] + 1
        }


# ── Node 2: RetrievalPlanner ──────────────────────────────────────────────────

def retrieval_planner_node(state: AgentState) -> dict:
    """
    Node 2: Execute retrieval using text-only strategy.

    No LLM call - deterministic function.
    Always uses TextRetriever (Phase 2 proved it's best).

    Reads:  state["query"]
    Writes: state["retrieved_docs"], state["retrieved_images"], state["tool_calls_count"]

    Args:
        state: Current AgentState

    Returns:
        Partial state update dict
    """
    query = state["query"]

    try:
        # Get text retriever (lazy-loaded singleton)
        text_retriever, _, _ = _get_retrievers()

        # Retrieve documents
        results = text_retriever.retrieve(query)

        if not results:
            logger.warning(f"No results found for query: {query}")
            return {
                "retrieved_docs": [],
                "retrieved_images": [],
                "tool_calls_count": state["tool_calls_count"] + 1
            }

        # Extract image paths from metadata
        image_paths = []
        for doc in results:
            metadata = doc.get("metadata", {})

            # Add poster if available
            if metadata.get("poster_path"):
                image_paths.append(metadata["poster_path"])

            # Add stills if available
            if metadata.get("still_paths"):
                still_paths = metadata["still_paths"]
                if isinstance(still_paths, list):
                    image_paths.extend(still_paths)

        # Limit to 5 images total
        image_paths = image_paths[:5]

        logger.info(f"RetrievalPlanner: retrieved {len(results)} docs, {len(image_paths)} images")

        return {
            "retrieved_docs": results,
            "retrieved_images": image_paths,
            "tool_calls_count": state["tool_calls_count"] + 1
        }

    except Exception as e:
        logger.error(f"RetrievalPlanner failed: {e}")
        return {
            "retrieved_docs": [],
            "retrieved_images": [],
            "tool_calls_count": state["tool_calls_count"] + 1
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

def merge_taste_profiles(current: dict, update: dict) -> dict:
    """
    Merge new preferences into existing profile (additive).

    Args:
        current: Current TasteProfile dict
        update: New preferences to merge in

    Returns:
        Merged TasteProfile dict
    """
    merged = current.copy()

    # Append and deduplicate list fields
    list_fields = [
        "preferred_genres", "preferred_directors", "preferred_languages",
        "avoid_genres", "watched", "mood_keywords"
    ]
    for field in list_fields:
        current_list = merged.get(field, [])
        update_list = update.get(field, [])
        # Combine and deduplicate (case-insensitive for strings)
        combined = current_list + update_list
        if combined and isinstance(combined[0], str):
            # Deduplicate case-insensitively
            seen = set()
            deduped = []
            for item in combined:
                lower = item.lower()
                if lower not in seen:
                    seen.add(lower)
                    deduped.append(item)
            merged[field] = deduped
        else:
            merged[field] = list(set(combined))

    # Update year range (intersection if both specified)
    year_range = merged.get("year_range", {"min": None, "max": None})
    update_range = update.get("year_range", {"min": None, "max": None})

    if update_range.get("min") is not None:
        if year_range.get("min") is None:
            year_range["min"] = update_range["min"]
        else:
            year_range["min"] = max(year_range["min"], update_range["min"])

    if update_range.get("max") is not None:
        if year_range.get("max") is None:
            year_range["max"] = update_range["max"]
        else:
            year_range["max"] = min(year_range["max"], update_range["max"])

    merged["year_range"] = year_range

    # Update confidence (max of current and new)
    merged["confidence"] = max(
        current.get("confidence", 0.0),
        update.get("confidence", 0.0)
    )

    return merged


def taste_profile_updater_node(state: AgentState) -> dict:
    """
    Node 3: Extract user preferences from query and update taste profile.

    Reads:  state["query"], state["taste_profile"]
    Writes: state["taste_profile"]

    Args:
        state: Current AgentState

    Returns:
        Partial state update dict
    """
    query = state["query"]
    current_profile = state["taste_profile"]

    prompt = f"""You are extracting film preferences from a user query.

Current query: {query}

Current taste profile:
{json.dumps(current_profile, indent=2)}

Extract any NEW film preferences mentioned in this query.
Update the taste profile by identifying preferences to ADD (don't remove existing ones).

Return JSON with these fields (use empty lists if nothing to add):
{{
  "preferred_genres": ["thriller", "drama"],
  "preferred_directors": ["David Fincher"],
  "preferred_languages": ["non-English"],
  "year_range": {{"min": null, "max": 2010}},
  "avoid_genres": ["romantic comedy"],
  "watched": ["Film Title"],
  "mood_keywords": ["slow-burn", "psychological", "bleak"],
  "confidence": 0.8
}}

IMPORTANT for "watched" field:
- ONLY add films the user EXPLICITLY said they have watched
- Examples: "I've seen Oldboy", "I already watched Parasite", "I don't want films I've seen like Cache"
- DO NOT add films they're just asking about or mentioning
- Example of what NOT to add: "Who directed Parasite?" (just asking about it, not watched)

Confidence rules:
- 0.0-0.3: vague or no preferences mentioned
- 0.4-0.6: some preferences mentioned
- 0.7-1.0: explicit, detailed preferences

Return only the JSON.
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        update = parse_json_safe(response.text)

        if not update:
            logger.warning("TasteProfileUpdater: LLM returned empty/invalid JSON")
            return {"taste_profile": current_profile}

        # Merge with current profile
        merged_profile = merge_taste_profiles(current_profile, update)

        logger.info(f"TasteProfileUpdater: updated profile (confidence={merged_profile['confidence']:.2f})")

        return {"taste_profile": merged_profile}

    except Exception as e:
        logger.error(f"TasteProfileUpdater failed: {e}")
        # Return current profile unchanged
        return {"taste_profile": current_profile}


# ── Node 4: AnswerSynthesiser ─────────────────────────────────────────────────

def extract_cited_films(response: str, retrieved_docs: list[dict]) -> list[str]:
    """
    Extract film IDs mentioned in the response.

    Args:
        response: Generated response text
        retrieved_docs: List of retrieved documents

    Returns:
        List of film IDs that appear in the response
    """
    cited_film_ids = []
    response_lower = response.lower()

    for doc in retrieved_docs:
        title = doc.get("title", "")
        film_id = doc.get("film_id", "")

        if title and title.lower() in response_lower:
            if film_id and film_id not in cited_film_ids:
                cited_film_ids.append(film_id)

    return cited_film_ids


def answer_synthesiser_node(state: AgentState) -> dict:
    """
    Node 4: Generate natural language response with film recommendations.

    Reads:  state["query"], state["retrieved_docs"], state["taste_profile"],
            state["conversation_history"]
    Writes: state["response"], state["cited_films"]

    Args:
        state: Current AgentState

    Returns:
        Partial state update dict
    """
    query = state["query"]
    retrieved_docs = state["retrieved_docs"]
    taste_profile = state["taste_profile"]
    conversation_history = state.get("conversation_history", [])

    # Format context for prompt
    docs_text = format_retrieved_docs(retrieved_docs, max_docs=5)
    profile_text = format_taste_profile(taste_profile)

    # Format conversation history (last 3 turns)
    history_text = ""
    if conversation_history:
        recent_history = conversation_history[-6:]  # Last 3 turns (user + assistant)
        history_lines = []
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:200]
            history_lines.append(f"{role.capitalize()}: {content}")
        history_text = "\n".join(history_lines)

    prompt = f"""You are a knowledgeable film recommendation assistant.

User query: {query}

Retrieved films:
{docs_text}

User's taste profile:
{profile_text}

{f"Recent conversation:\n{history_text}\n" if history_text else ""}

Generate a helpful response:
1. Recommend 1-3 films from the retrieved results
2. Explain WHY each film matches the user's query and preferences
3. Reference specific aspects (director, themes, visual style, etc.)
4. Be concise but informative (2-3 sentences per film)
5. If user has watched a film, don't recommend it

Response:
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=500,
            )
        )

        response_text = response.text.strip()

        # Extract which films were cited
        cited_film_ids = extract_cited_films(response_text, retrieved_docs)

        logger.info(f"AnswerSynthesiser: generated response, cited {len(cited_film_ids)} films")

        return {
            "response": response_text,
            "cited_films": cited_film_ids
        }

    except Exception as e:
        logger.error(f"AnswerSynthesiser failed: {e}")
        # Return error message as response
        return {
            "response": "I apologize, but I'm having trouble generating a recommendation right now. Please try rephrasing your query.",
            "cited_films": []
        }


# ── Node 5: Verifier ──────────────────────────────────────────────────────────

def verify_rules(state: AgentState) -> tuple[bool, str | None]:
    """
    Fast deterministic checks for verification.

    Args:
        state: Current AgentState

    Returns:
        Tuple of (verified, reason). If verified=False, reason explains why.
    """
    response = state.get("response", "")
    cited_film_ids = state.get("cited_films", [])
    taste_profile = state.get("taste_profile", {})
    retrieved_docs = state.get("retrieved_docs", [])

    # Check 0: Empty response?
    if not response or not response.strip():
        return False, "empty_response"

    # Build film metadata lookup
    film_metadata = {doc["film_id"]: doc for doc in retrieved_docs if "film_id" in doc}

    # Check 1: Already watched?
    watched_list = taste_profile.get("watched", [])
    for film_id in cited_film_ids:
        # Check both film_id and title (in case watched list has titles)
        if film_id in watched_list:
            film_title = film_metadata.get(film_id, {}).get("title", film_id)
            return False, f"Film '{film_title}' already watched"

        # Also check if title is in watched list
        film_title = film_metadata.get(film_id, {}).get("title", "")
        if film_title and film_title in watched_list:
            return False, f"Film '{film_title}' already watched"

    # Check 2: Matches avoid genres?
    avoid_genres = taste_profile.get("avoid_genres", [])
    if avoid_genres:
        avoid_genres_lower = [g.lower() for g in avoid_genres]
        for film_id in cited_film_ids:
            film = film_metadata.get(film_id, {})
            film_genres = film.get("metadata", {}).get("genres", [])

            for genre in film_genres:
                if genre.lower() in avoid_genres_lower:
                    film_title = film.get("title", film_id)
                    return False, f"Film '{film_title}' is {genre}, user avoids this genre"

    return True, None


def verify_contradictions(state: AgentState) -> tuple[bool, str | None]:
    """
    LLM-based contradiction detection (only if profile confidence > 0.5).

    Args:
        state: Current AgentState

    Returns:
        Tuple of (verified, reason)
    """
    taste_profile = state.get("taste_profile", {})
    confidence = taste_profile.get("confidence", 0.0)

    # Skip if profile not confident enough
    if confidence <= 0.5:
        return True, None

    cited_film_ids = state.get("cited_films", [])
    retrieved_docs = state.get("retrieved_docs", [])

    if not cited_film_ids:
        return True, None

    # Build cited films description
    cited_films_text = []
    film_metadata = {doc["film_id"]: doc for doc in retrieved_docs if "film_id" in doc}

    for film_id in cited_film_ids:
        film = film_metadata.get(film_id, {})
        title = film.get("title", film_id)
        genres = film.get("metadata", {}).get("genres", [])
        cited_films_text.append(f"- {title} (genres: {', '.join(genres)})")

    prompt = f"""You are checking if film recommendations contradict user preferences.

User's taste profile:
{json.dumps(taste_profile, indent=2)}

Recommended films:
{chr(10).join(cited_films_text)}

Do these recommendations CONTRADICT the user's stated preferences?

Examples of contradictions:
- User said "no action", recommended action film
- User said "pre-2010", recommended 2015 film
- User prefers "non-English", recommended Hollywood English film
- User avoids "romance", recommended romantic comedy

Respond with JSON only:
{{"contradicts": true/false, "reason": "explanation if true, null if false"}}
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        result = parse_json_safe(response.text)

        if result.get("contradicts"):
            reason = result.get("reason", "Contradicts taste profile")
            return False, reason

        return True, None

    except Exception as e:
        logger.error(f"Contradiction check failed: {e}")
        # On error, pass verification (don't block on LLM failure)
        return True, None


def verifier_node(state: AgentState) -> dict:
    """
    Node 5: Check if recommendations are valid, trigger retry if not.

    Combines rule-based checks (fast) with LLM contradiction detection (optional).

    Reads:  state["cited_films"], state["taste_profile"], state["retrieved_docs"],
            state["retry_count"]
    Writes: state["verified"], state["verification_reason"], state["retry_count"]

    Args:
        state: Current AgentState

    Returns:
        Partial state update dict
    """
    # Rule-based checks first (deterministic, fast)
    verified, reason = verify_rules(state)
    if not verified:
        logger.warning(f"Verifier: rule check failed - {reason}")
        return {
            "verified": False,
            "verification_reason": reason,
            "retry_count": state.get("retry_count", 0) + 1
        }

    # LLM contradiction check (only if profile is confident)
    verified, reason = verify_contradictions(state)
    if not verified:
        logger.warning(f"Verifier: contradiction check failed - {reason}")
        return {
            "verified": False,
            "verification_reason": reason,
            "retry_count": state.get("retry_count", 0) + 1
        }

    # All checks passed
    logger.info("Verifier: all checks passed")
    return {
        "verified": True,
        "verification_reason": None,
        "retry_count": state.get("retry_count", 0)
    }
