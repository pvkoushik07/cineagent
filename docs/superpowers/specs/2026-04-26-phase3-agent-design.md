# Phase 3 LangGraph Agent Design Specification
**Date:** 2026-04-26  
**Phase:** 3 — LangGraph Agent Implementation  
**Status:** Approved

---

## Overview

Build a 5-node conversational LangGraph agent for personalized film recommendations with dynamic taste profile updating across conversation turns.

## Context

Phase 2 is complete:
- ✅ 4 retrieval variants tested (text, caption, CLIP, hybrid)
- ✅ Ablation 1 results: Text-only achieves 100% Recall@5 on both factual and visual queries
- ✅ CLIP-only: 20% recall on visual queries (underperforms)
- ✅ Hybrid RRF: 60% recall on visual queries (degraded by CLIP)
- ✅ Knowledge base: 469 films with multimodal data (plots, reviews, captions, images)

**Key finding from Phase 2:** Use text-only retrieval for the agent. No benefit from CLIP or hybrid fusion.

Existing scaffolding:
- `src/agent/state.py` - AgentState and TasteProfile TypedDicts fully defined
- `src/agent/graph.py` - Graph structure and conditional routing implemented
- `src/agent/nodes.py` - Node stubs exist, need full implementation

---

## Design Approach

**Selected: Hybrid (Rules + LLM)**

Use LLM where semantic understanding is needed, deterministic rules where possible.

**Why this approach:**
- QueryRouter: needs semantic classification → LLM
- RetrievalPlanner: Phase 2 proved text-only is best → deterministic (no LLM)
- TasteProfileUpdater: needs preference extraction → LLM with structured output
- AnswerSynthesiser: needs natural language generation → LLM
- Verifier: simple checks are deterministic, contradiction detection needs LLM → rules + LLM

**LLM calls per turn:** 3-4 (QueryRouter + TasteUpdater + Synthesiser + optional Verifier)

**Alternative approaches considered:**
- LLM-Heavy: 5 LLM calls per turn (slow, expensive, unnecessary for simple nodes)
- Rules-Heavy: Brittle keyword matching, no nuance, misses edge cases

---

## Architecture

### Graph Flow

```
User Query
    ↓
[1] QueryRouter (LLM)
    ↓
[2] RetrievalPlanner (Deterministic - TextRetriever only)
    ↓
[3] TasteProfileUpdater (LLM - structured JSON extraction)
    ↓
[4] AnswerSynthesiser (LLM - multimodal response with images)
    ↓
[5] Verifier (Rules + LLM)
    ↓
  verified? ──No (retry_count < 2)──→ RetrievalPlanner
    ↓ Yes
   END
```

### State Flow

**State updates by node:**
- QueryRouter writes: `query_type`, `retrieval_strategy="text"`, `tool_calls_count`
- RetrievalPlanner writes: `retrieved_docs`, `retrieved_images`, `tool_calls_count`
- TasteProfileUpdater writes: `taste_profile`
- AnswerSynthesiser writes: `response`, `cited_films`
- Verifier writes: `verified`, `verification_reason`, `retry_count`

**Conversation memory (multi-turn):**
- `taste_profile` persists across turns (carried over in `run_turn()`)
- `conversation_history` appends each turn
- Profile is additive: new preferences append, old ones persist

**Retry logic:**
- Verifier can fail verification → routes back to RetrievalPlanner
- Max 2 retries to prevent infinite loops
- After 2 failures: accept response, log warning

---

## Component Specifications

### Component 1: QueryRouter Node

**Purpose:** Classify query type and set retrieval strategy

**Implementation:** LLM call with structured prompt

**Input:** `state["query"]`, `state["conversation_history"]`

**Output:** `{"query_type": str, "retrieval_strategy": "text", "tool_calls_count": int}`

**Prompt template:**
```
You are a query classifier for a film recommendation agent.

Classify the user query into exactly one of these types:
- factual: asking for specific facts (director, year, cast, plot details)
- visual: describing visual mood, aesthetic, atmosphere, color palette
- hybrid: needs both factual and visual information
- multi_hop: requires combining multiple constraints

Query: {query}

Respond with JSON only:
{"query_type": "<type>", "retrieval_strategy": "text", "reasoning": "<one sentence>"}

Note: retrieval_strategy is always "text" (empirically best from ablation study).
```

**Error handling:**
- LLM call fails → default to `query_type="hybrid"`, `retrieval_strategy="text"`
- Invalid JSON → log warning, use default
- Always increment `tool_calls_count`

**Test cases:**
- "Who directed Mulholland Drive?" → factual
- "cold, desaturated, rain-soaked atmosphere" → visual
- "non-English thriller from 2000s" → multi_hop

---

### Component 2: RetrievalPlanner Node

**Purpose:** Execute text-based retrieval

**Implementation:** Deterministic - no LLM call

**Input:** `state["query"]`

**Output:** `{"retrieved_docs": list, "retrieved_images": list, "tool_calls_count": int}`

**Logic:**
```python
def retrieval_planner_node(state: AgentState) -> dict:
    retriever = get_text_retriever()  # Lazy-loaded singleton
    
    results = retriever.retrieve(state["query"], top_k=5)
    
    # Extract image paths from metadata
    image_paths = []
    for doc in results:
        if doc.get("metadata", {}).get("poster_path"):
            image_paths.append(doc["metadata"]["poster_path"])
        if doc.get("metadata", {}).get("still_paths"):
            image_paths.extend(doc["metadata"]["still_paths"])
    
    return {
        "retrieved_docs": results,
        "retrieved_images": image_paths[:5],  # Limit to 5 images
        "tool_calls_count": state["tool_calls_count"] + 1
    }
```

**Why no LLM:** Phase 2 proved text-only retrieval achieves 100% recall. No need for dynamic strategy selection.

**Error handling:**
- Retrieval fails → return empty lists, log error
- No results found → return empty lists, log warning

---

### Component 3: TasteProfileUpdater Node

**Purpose:** Extract user preferences from query and update taste profile

**Implementation:** LLM call with structured JSON output

**Input:** `state["query"]`, `state["taste_profile"]`, `state["conversation_history"]`

**Output:** `{"taste_profile": TasteProfile}`

**Prompt template:**
```
You are extracting film preferences from a user query.

Current query: {query}
Current taste profile: {json.dumps(taste_profile, indent=2)}

Extract any NEW film preferences mentioned in this query.
Update the taste profile by ADDING new preferences (don't remove existing ones).

Return JSON with these fields (empty lists if nothing to add):
{
  "preferred_genres": ["thriller", "drama"],
  "preferred_directors": ["David Fincher"],
  "preferred_languages": ["non-English"],
  "year_range": {"min": null, "max": 2010},
  "avoid_genres": ["romantic comedy"],
  "watched": ["Oldboy", "Parasite"],
  "mood_keywords": ["slow-burn", "psychological", "bleak"],
  "confidence": 0.8
}

Confidence rules:
- 0.0-0.3: vague or no preferences mentioned
- 0.4-0.6: some preferences mentioned
- 0.7-1.0: explicit, detailed preferences

Return only the JSON.
```

**Merge logic:**
```python
def merge_taste_profile(current: TasteProfile, update: TasteProfile) -> TasteProfile:
    """Merge new preferences into existing profile (additive)."""
    merged = current.copy()
    
    # Append and deduplicate lists
    for key in ["preferred_genres", "preferred_directors", "preferred_languages", 
                "avoid_genres", "watched", "mood_keywords"]:
        merged[key] = list(set(current[key] + update[key]))
    
    # Update year range (take intersection if both specified)
    if update["year_range"]["min"] is not None:
        if merged["year_range"]["min"] is None:
            merged["year_range"]["min"] = update["year_range"]["min"]
        else:
            merged["year_range"]["min"] = max(merged["year_range"]["min"], 
                                              update["year_range"]["min"])
    
    if update["year_range"]["max"] is not None:
        if merged["year_range"]["max"] is None:
            merged["year_range"]["max"] = update["year_range"]["max"]
        else:
            merged["year_range"]["max"] = min(merged["year_range"]["max"], 
                                              update["year_range"]["max"])
    
    # Update confidence (max of current and new)
    merged["confidence"] = max(current["confidence"], update["confidence"])
    
    return merged
```

**Error handling:**
- LLM fails → return current profile unchanged, log error
- Invalid JSON → return current profile, log warning
- Missing fields in JSON → use empty lists/defaults

---

### Component 4: AnswerSynthesiser Node

**Purpose:** Generate natural language response with film recommendations

**Implementation:** LLM call with multimodal prompt

**Input:** `state["query"]`, `state["retrieved_docs"]`, `state["retrieved_images"]`, `state["taste_profile"]`, `state["conversation_history"]`

**Output:** `{"response": str, "cited_films": list[str]}`

**Prompt template:**
```
You are a knowledgeable film recommendation assistant.

User query: {query}

Retrieved films:
{format_retrieved_docs(retrieved_docs)}

User's taste profile:
{format_taste_profile(taste_profile)}

Conversation history:
{format_conversation_history(conversation_history)}

Generate a helpful response:
1. Recommend 1-3 films from the retrieved results
2. Explain WHY each film matches the user's query and preferences
3. Reference specific visual aspects if images are available
4. Be concise but informative (2-3 sentences per film)
5. If user has watched a film, don't recommend it

Response:
```

**Response parsing:**
```python
def extract_cited_films(response: str, retrieved_docs: list) -> list[str]:
    """Extract film IDs mentioned in the response."""
    cited_films = []
    for doc in retrieved_docs:
        film_title = doc.get("title", "")
        if film_title.lower() in response.lower():
            cited_films.append(doc["film_id"])
    return cited_films
```

**Multimodal handling:**
- Retrieved docs contain image paths in metadata
- Agent can reference images in response ("As seen in the poster...")
- For CLI: print image paths at end of response
- For notebook: display images inline using IPython.display

**Error handling:**
- LLM fails → return error message as response
- Empty retrieved docs → respond with "No films found matching your query"
- Cited films extraction fails → return empty list, log warning

---

### Component 5: Verifier Node

**Purpose:** Check if recommendations are valid, trigger retry if not

**Implementation:** Rules + LLM hybrid

**Input:** `state["cited_films"]`, `state["taste_profile"]`, `state["retrieved_docs"]`, `state["retry_count"]`

**Output:** `{"verified": bool, "verification_reason": str | None, "retry_count": int}`

**Verification logic:**

**Step 1: Rule-based checks (deterministic)**
```python
def verify_rules(state: AgentState) -> tuple[bool, str | None]:
    """Fast deterministic checks."""
    cited_film_ids = state["cited_films"]
    taste_profile = state["taste_profile"]
    retrieved_docs = state["retrieved_docs"]
    
    # Check 1: Already watched?
    for film_id in cited_film_ids:
        if film_id in taste_profile["watched"]:
            return False, f"Film {film_id} already watched"
    
    # Check 2: Matches avoid list?
    for doc in retrieved_docs:
        if doc["film_id"] in cited_film_ids:
            film_genres = doc.get("metadata", {}).get("genres", [])
            for avoid_genre in taste_profile["avoid_genres"]:
                if avoid_genre.lower() in [g.lower() for g in film_genres]:
                    return False, f"Film has {avoid_genre}, user avoids this genre"
    
    return True, None
```

**Step 2: LLM contradiction check (only if confidence > 0.5)**
```python
def verify_contradictions(state: AgentState) -> tuple[bool, str | None]:
    """LLM-based contradiction detection."""
    if state["taste_profile"]["confidence"] <= 0.5:
        return True, None  # Skip if profile not confident
    
    prompt = f"""
    User's taste profile:
    {json.dumps(state["taste_profile"], indent=2)}
    
    Recommended films:
    {format_cited_films(state["cited_films"], state["retrieved_docs"])}
    
    Do these recommendations CONTRADICT the user's preferences?
    Examples of contradictions:
    - User said "no action", recommended action film
    - User said "pre-2010", recommended 2015 film
    - User prefers "non-English", recommended Hollywood film
    
    Respond with JSON:
    {{"contradicts": true/false, "reason": "explanation if true, null if false"}}
    """
    
    response = llm_call(prompt)
    result = parse_json(response)
    
    if result.get("contradicts"):
        return False, result.get("reason", "Contradicts taste profile")
    
    return True, None
```

**Combined verification:**
```python
def verifier_node(state: AgentState) -> dict:
    # Rule checks first (fast)
    verified, reason = verify_rules(state)
    if not verified:
        return {
            "verified": False,
            "verification_reason": reason,
            "retry_count": state["retry_count"] + 1
        }
    
    # LLM contradiction check (if profile is confident)
    verified, reason = verify_contradictions(state)
    if not verified:
        return {
            "verified": False,
            "verification_reason": reason,
            "retry_count": state["retry_count"] + 1
        }
    
    # All checks passed
    return {
        "verified": True,
        "verification_reason": None,
        "retry_count": state["retry_count"]
    }
```

**Retry behavior:**
- If verification fails and `retry_count < 2`: route back to RetrievalPlanner
- If `retry_count >= 2`: accept response anyway, log warning
- Retry gives RetrievalPlanner a second chance (though it's deterministic, so same result expected)

---

## Data Flow & State Management

### State Updates (Immutable Pattern)

Each node returns a partial state dict:
```python
def node(state: AgentState) -> dict:
    # Read from state
    query = state["query"]
    
    # Do work...
    
    # Return ONLY the fields being updated
    return {
        "query_type": "factual",
        "tool_calls_count": state["tool_calls_count"] + 1
    }
```

LangGraph merges the partial update into the full state.

### Multi-Turn Conversation

**Initial turn:**
```python
state = initial_state(query)
```

**Subsequent turns:**
```python
state = run_turn(query, previous_state=previous_state)
# Inherits: taste_profile, conversation_history
```

**Conversation history format:**
```python
conversation_history = [
    {"role": "user", "content": "I love thrillers"},
    {"role": "assistant", "content": "Based on your interest..."},
    {"role": "user", "content": "Suggest something"},
]
```

### Image Handling

**Retrieval:**
- TextRetriever returns docs with metadata: `{"poster_path": "...", "still_paths": [...]}`
- RetrievalPlanner extracts these → `state["retrieved_images"]`

**Synthesis:**
- AnswerSynthesiser can reference images in response text
- For CLI: print image paths after response
- For notebook: use `IPython.display.Image()` to show inline

---

## Error Handling

### LLM Call Failures

```python
def safe_llm_call(prompt: str, default_result: dict) -> dict:
    """Call LLM with error handling."""
    try:
        response = model.generate_content(prompt)
        return parse_json_safe(response.text)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return default_result
```

### JSON Parsing Failures

```python
def parse_json_safe(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    try:
        # Strip markdown if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {text[:100]}... Error: {e}")
        return {}
```

### Retrieval Failures

```python
try:
    results = retriever.retrieve(query)
    if not results:
        logger.warning("No results found for query")
        return {"retrieved_docs": [], "retrieved_images": []}
except Exception as e:
    logger.error(f"Retrieval failed: {e}")
    return {"retrieved_docs": [], "retrieved_images": []}
```

### Logging Levels

- **INFO**: Node execution, successful operations, tool calls
- **WARNING**: Fallbacks triggered, retries, empty results, missing data
- **ERROR**: LLM failures, parsing errors, exceptions

### Graceful Degradation

- Never crash the agent - always return a state update
- If synthesis fails: return error message as response
- If retrieval fails: synthesize from memory/history
- If all fails: acknowledge the failure to the user

---

## Testing Strategy

### 1. Unit Tests (Per Node)

File: `tests/test_agent_nodes.py`

```python
def test_query_router_factual():
    state = mock_state(query="Who directed Mulholland Drive?")
    result = query_router_node(state)
    assert result["query_type"] == "factual"
    assert result["retrieval_strategy"] == "text"

def test_retrieval_planner_returns_docs():
    state = mock_state(query="psychological thriller")
    result = retrieval_planner_node(state)
    assert len(result["retrieved_docs"]) > 0
    assert "retrieved_images" in result

def test_taste_updater_extracts_genres():
    state = mock_state(
        query="I love slow-burn thrillers",
        taste_profile=empty_taste_profile()
    )
    result = taste_profile_updater_node(state)
    assert "thriller" in result["taste_profile"]["preferred_genres"]

def test_verifier_fails_on_watched():
    state = mock_state(
        cited_films=["1018"],
        taste_profile={"watched": ["1018"], ...}
    )
    result = verifier_node(state)
    assert result["verified"] == False
    assert "already watched" in result["verification_reason"].lower()
```

**Coverage target:** 80%+ for node functions

### 2. Integration Tests (Full Graph)

File: `tests/test_agent_integration.py`

```python
def test_full_turn_factual_query():
    state = run_turn("Who directed Parasite?")
    assert state["query_type"] == "factual"
    assert len(state["retrieved_docs"]) > 0
    assert "Bong Joon-ho" in state["response"] or "Parasite" in state["response"]
    assert state["verified"] == True

def test_multi_turn_memory():
    state1 = run_turn("I love thrillers")
    assert "thriller" in state1["taste_profile"]["preferred_genres"]
    
    state2 = run_turn("Suggest something", previous_state=state1)
    assert "thriller" in state2["taste_profile"]["preferred_genres"]
    assert len(state2["conversation_history"]) == 4  # 2 turns × 2 messages

def test_verifier_retry_on_watched():
    # Requires mocking to force watched film in recommendation
    # Verify retry_count increments and graph re-routes
    pass
```

### 3. Prompt Testing

File: `tests/test_prompts.py`

```python
def test_router_classifies_common_queries():
    """Smoke test: router handles common query patterns."""
    test_cases = [
        ("Who directed Inception?", "factual"),
        ("neon-lit nightscape", "visual"),
        ("non-English thriller from 2000s", "multi_hop"),
    ]
    for query, expected_type in test_cases:
        state = mock_state(query=query)
        result = query_router_node(state)
        assert result["query_type"] == expected_type
```

### 4. Manual Testing (Interactive CLI)

```bash
python src/agent/graph.py

# Test conversation flow:
Turn 1: "I love slow-burn psychological thrillers"
Turn 2: "Preferably non-English, pre-2010"
Turn 3: "I've already seen Oldboy and Caché"
# Expected: Turn 3 doesn't recommend Oldboy/Caché

# Test visual query:
Turn 1: "cold rainy atmosphere"
# Expected: Retrieves films, cites visual aspects

# Test factual query:
Turn 1: "Who directed Mulholland Drive?"
# Expected: Mentions David Lynch
```

### 5. Evaluation Instrumentation (Phase 4)

Agent must track metrics for Phase 4 evaluation:
- `state["tool_calls_count"]` - number of retrieval calls
- `state["latency_ms"]` - end-to-end turn time
- `state["retrieved_docs"]` - for Recall@5 calculation
- `state["response"]` - for ragas faithfulness scoring

---

## Success Criteria

Phase 3 is complete when:

✅ All 5 nodes fully implemented with error handling  
✅ Unit tests pass (>80% coverage on nodes)  
✅ Integration tests pass (full graph flows work)  
✅ Interactive CLI works for multi-turn conversations  
✅ Taste profile persists and updates across turns  
✅ Verifier retry logic works (can route back to retrieval)  
✅ Agent returns multimodal responses (text + image paths)  
✅ No crashes - graceful degradation on all errors  
✅ Git commit with "Phase 3 complete" message  

**Validation command:**
```bash
pytest tests/test_agent_nodes.py tests/test_agent_integration.py -v
python src/agent/graph.py  # Manual smoke test
```

---

## Known Limitations & Future Work

1. **Retry logic is limited** - RetrievalPlanner is deterministic, so retrying gives same results. Future: vary top_k or use different query reformulation.

2. **Image display CLI-limited** - CLI can only print paths. For better UX, use notebook or build simple web interface.

3. **Taste profile confidence is LLM-assigned** - Not calibrated. Future: use confidence based on number of explicit preferences.

4. **No query reformulation** - If retrieval fails, agent doesn't try rephrasing. Future: add QueryReformulator node.

5. **No streaming responses** - Agent returns full response at end. Future: stream tokens for better UX.

6. **Single retrieval strategy** - Always text-only (by design from Phase 2). Future: if CLIP improves, add dynamic strategy selection.

---

## Next Steps

After Phase 3 completion:
1. Commit Phase 3 code and tests
2. Manual smoke testing via interactive CLI
3. Proceed to Phase 4: Evaluation harness
4. Compare 3 variants: Plain LLM vs Static RAG vs Full Agent
5. Measure: Recall@5, ragas faithfulness, latency, tool calls
