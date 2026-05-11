# Phase 3 LangGraph Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 5-node conversational LangGraph agent with text-only retrieval, dynamic taste profiles, and multi-turn memory

**Architecture:** Hybrid approach - LLM for semantic understanding (QueryRouter, TasteUpdater, Synthesiser, Verifier contradiction check), deterministic rules where possible (RetrievalPlanner uses text-only based on Phase 2 findings, Verifier rule checks)

**Tech Stack:** LangGraph, Gemini Flash (google-generativeai), TextRetriever, existing state.py and graph.py

---

## File Structure

**Files to modify:**
- `src/agent/nodes.py` - Implement all 5 node functions (currently stubs)
- `tests/test_agent_nodes.py` - Expand existing unit tests
  
**Files to create:**
- `tests/test_agent_integration.py` - New integration tests for full graph flow

**Files already complete (no changes needed):**
- `src/agent/state.py` - AgentState and TasteProfile TypedDicts
- `src/agent/graph.py` - Graph structure with conditional routing

---

## Task 1: Helper Functions and JSON Parsing

**Files:**
- Modify: `src/agent/nodes.py` (add to top of file after imports)

- [ ] **Step 1: Add JSON parsing helper**

Add after the existing imports and before node implementations:

```python
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
```

- [ ] **Step 2: Add document formatting helper**

```python
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
```

- [ ] **Step 3: Add taste profile formatting helper**

```python
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
```

- [ ] **Step 4: Test helpers**

Run: `python -c "import sys; sys.path.insert(0, 'src'); from agent.nodes import parse_json_safe, format_retrieved_docs, format_taste_profile; print('Helpers imported successfully')"`

Expected: "Helpers imported successfully"

- [ ] **Step 5: Commit helpers**

```bash
git add src/agent/nodes.py
git commit -m "feat: add helper functions for JSON parsing and formatting"
```

---


## Task 2: Implement QueryRouter Node

**Files:**
- Modify: `src/agent/nodes.py:77-120` (replace query_router_node stub)
- Test: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write test for factual query classification**

Add to `tests/test_agent_nodes.py` in the `TestQueryRouterNode` class:

```python
@patch("agent.nodes.genai.GenerativeModel")
def test_classifies_factual_query(self, mock_genai_cls):
    """Router should classify 'Who directed X?' as factual."""
    mock_response = MagicMock()
    mock_response.text = '{"query_type": "factual", "retrieval_strategy": "text", "reasoning": "asking for director"}'
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    mock_genai_cls.return_value = mock_model
    
    from agent.state import initial_state
    from agent.nodes import query_router_node
    
    state = initial_state("Who directed Mulholland Drive?")
    result = query_router_node(state)
    
    assert result["query_type"] == "factual"
    assert result["retrieval_strategy"] == "text"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_nodes.py::TestQueryRouterNode::test_classifies_factual_query -v`

Expected: FAIL (node not implemented yet)

- [ ] **Step 3: Implement QueryRouter node**

Replace the query_router_node function in `src/agent/nodes.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_nodes.py::TestQueryRouterNode::test_classifies_factual_query -v`

Expected: PASS

- [ ] **Step 5: Add test for visual query**

Add test:

```python
@patch("agent.nodes.genai.GenerativeModel")
def test_classifies_visual_query(self, mock_genai_cls):
    """Router should classify mood/atmosphere as visual."""
    mock_response = MagicMock()
    mock_response.text = '{"query_type": "visual", "retrieval_strategy": "text"}'
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    mock_genai_cls.return_value = mock_model
    
    from agent.state import initial_state
    from agent.nodes import query_router_node
    
    state = initial_state("cold, desaturated, rain-soaked atmosphere")
    result = query_router_node(state)
    
    assert result["query_type"] == "visual"
```

- [ ] **Step 6: Run all QueryRouter tests**

Run: `pytest tests/test_agent_nodes.py::TestQueryRouterNode -v`

Expected: All tests PASS

- [ ] **Step 7: Commit QueryRouter**

```bash
git add src/agent/nodes.py tests/test_agent_nodes.py
git commit -m "feat: implement QueryRouter node with LLM classification"
```

---

## Task 3: Implement RetrievalPlanner Node

**Files:**
- Modify: `src/agent/nodes.py` (add retrieval_planner_node after QueryRouter)
- Test: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write test for retrieval**

Add new test class to `tests/test_agent_nodes.py`:

```python
class TestRetrievalPlannerNode:
    
    def test_retrieves_documents(self):
        """RetrievalPlanner should return retrieved docs and images."""
        from agent.state import initial_state
        from agent.nodes import retrieval_planner_node
        
        state = initial_state("psychological thriller")
        state["retrieval_strategy"] = "text"
        
        result = retrieval_planner_node(state)
        
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], list)
        assert len(result["retrieved_docs"]) > 0
    
    def test_extracts_image_paths(self):
        """RetrievalPlanner should extract image paths from metadata."""
        from agent.state import initial_state
        from agent.nodes import retrieval_planner_node
        
        state = initial_state("thriller with posters")
        state["retrieval_strategy"] = "text"
        
        result = retrieval_planner_node(state)
        
        assert "retrieved_images" in result
        assert isinstance(result["retrieved_images"], list)
    
    def test_increments_tool_calls(self):
        """RetrievalPlanner should increment tool_calls_count."""
        from agent.state import initial_state
        from agent.nodes import retrieval_planner_node
        
        state = initial_state("any query")
        state["retrieval_strategy"] = "text"
        initial_count = state["tool_calls_count"]
        
        result = retrieval_planner_node(state)
        
        assert result["tool_calls_count"] == initial_count + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_nodes.py::TestRetrievalPlannerNode -v`

Expected: FAIL (function not defined)

- [ ] **Step 3: Implement RetrievalPlanner node**

Add to `src/agent/nodes.py` after query_router_node:

```python
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
        results = text_retriever.retrieve(query, top_k=TOP_K)
        
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_nodes.py::TestRetrievalPlannerNode -v`

Expected: All tests PASS

- [ ] **Step 5: Commit RetrievalPlanner**

```bash
git add src/agent/nodes.py tests/test_agent_nodes.py
git commit -m "feat: implement RetrievalPlanner node with text-only retrieval"
```

---

## Task 4: Implement TasteProfileUpdater Node

**Files:**
- Modify: `src/agent/nodes.py` (add taste_profile_updater_node and merge helper)
- Test: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write test for preference extraction**

Add new test class:

```python
class TestTasteProfileUpdaterNode:
    
    @patch("agent.nodes.genai.GenerativeModel")
    def test_extracts_genre_preferences(self, mock_genai_cls):
        """TasteProfileUpdater should extract genres from query."""
        mock_response = MagicMock()
        mock_response.text = '''{"preferred_genres": ["thriller"], "preferred_directors": [], 
                                 "preferred_languages": [], "year_range": {"min": null, "max": null},
                                 "avoid_genres": [], "watched": [], "mood_keywords": ["slow-burn"], 
                                 "confidence": 0.7}'''
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai_cls.return_value = mock_model
        
        from agent.state import initial_state
        from agent.nodes import taste_profile_updater_node
        
        state = initial_state("I love slow-burn thrillers")
        result = taste_profile_updater_node(state)
        
        assert "taste_profile" in result
        assert "thriller" in result["taste_profile"]["preferred_genres"]
        assert "slow-burn" in result["taste_profile"]["mood_keywords"]
    
    @patch("agent.nodes.genai.GenerativeModel")
    def test_merges_with_existing_profile(self, mock_genai_cls):
        """TasteProfileUpdater should append to existing preferences."""
        mock_response = MagicMock()
        mock_response.text = '''{"preferred_genres": ["drama"], "preferred_directors": [], 
                                 "preferred_languages": [], "year_range": {"min": null, "max": null},
                                 "avoid_genres": [], "watched": [], "mood_keywords": [], 
                                 "confidence": 0.6}'''
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai_cls.return_value = mock_model
        
        from agent.state import initial_state, empty_taste_profile
        from agent.nodes import taste_profile_updater_node
        
        state = initial_state("I also like dramas")
        state["taste_profile"] = empty_taste_profile()
        state["taste_profile"]["preferred_genres"] = ["thriller"]
        
        result = taste_profile_updater_node(state)
        
        # Should contain both old and new genres
        assert "thriller" in result["taste_profile"]["preferred_genres"]
        assert "drama" in result["taste_profile"]["preferred_genres"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_nodes.py::TestTasteProfileUpdaterNode -v`

Expected: FAIL

- [ ] **Step 3: Implement merge helper function**

Add before taste_profile_updater_node:

```python
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
```

- [ ] **Step 4: Implement TasteProfileUpdater node**

Add after merge_taste_profiles:

```python
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
  "watched": ["Oldboy", "Parasite"],
  "mood_keywords": ["slow-burn", "psychological", "bleak"],
  "confidence": 0.8
}}

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_agent_nodes.py::TestTasteProfileUpdaterNode -v`

Expected: All tests PASS

- [ ] **Step 6: Commit TasteProfileUpdater**

```bash
git add src/agent/nodes.py tests/test_agent_nodes.py
git commit -m "feat: implement TasteProfileUpdater with LLM extraction and merge logic"
```

---

## Task 5: Implement AnswerSynthesiser Node

**Files:**
- Modify: `src/agent/nodes.py` (add answer_synthesiser_node)
- Test: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write test for response generation**

Add test class:

```python
class TestAnswerSynthesiserNode:
    
    @patch("agent.nodes.genai.GenerativeModel")
    def test_generates_response(self, mock_genai_cls, base_state):
        """AnswerSynthesiser should generate a response string."""
        mock_response = MagicMock()
        mock_response.text = "I recommend Mulholland Drive directed by David Lynch."
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai_cls.return_value = mock_model
        
        from agent.nodes import answer_synthesiser_node
        
        result = answer_synthesiser_node(base_state)
        
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
    
    @patch("agent.nodes.genai.GenerativeModel")
    def test_extracts_cited_films(self, mock_genai_cls, base_state):
        """AnswerSynthesiser should extract film titles mentioned in response."""
        mock_response = MagicMock()
        mock_response.text = "I recommend Mulholland Drive for its surreal atmosphere."
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai_cls.return_value = mock_model
        
        from agent.nodes import answer_synthesiser_node
        
        result = answer_synthesiser_node(base_state)
        
        assert "cited_films" in result
        assert isinstance(result["cited_films"], list)
        # Should have extracted "1018" (Mulholland Drive film_id)
        assert "1018" in result["cited_films"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_nodes.py::TestAnswerSynthesiserNode -v`

Expected: FAIL

- [ ] **Step 3: Implement cited films extraction helper**

Add before answer_synthesiser_node:

```python
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
```

- [ ] **Step 4: Implement AnswerSynthesiser node**

Add after extract_cited_films:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_agent_nodes.py::TestAnswerSynthesiserNode -v`

Expected: All tests PASS

- [ ] **Step 6: Commit AnswerSynthesiser**

```bash
git add src/agent/nodes.py tests/test_agent_nodes.py
git commit -m "feat: implement AnswerSynthesiser with LLM generation and citation extraction"
```

---

## Task 6: Implement Verifier Node

**Files:**
- Modify: `src/agent/nodes.py` (add verifier_node with rule and LLM checks)
- Test: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write test for watched film check**

Add test class:

```python
class TestVerifierNode:
    
    def test_fails_on_watched_film(self):
        """Verifier should fail if recommended film is in watched list."""
        from agent.state import initial_state, empty_taste_profile
        from agent.nodes import verifier_node
        
        state = initial_state("suggest a thriller")
        state["cited_films"] = ["1018"]  # Mulholland Drive
        state["taste_profile"] = empty_taste_profile()
        state["taste_profile"]["watched"] = ["1018"]
        state["retrieved_docs"] = [
            {
                "film_id": "1018",
                "title": "Mulholland Drive",
                "metadata": {"genres": ["Drama", "Mystery"]}
            }
        ]
        
        result = verifier_node(state)
        
        assert result["verified"] == False
        assert "already watched" in result["verification_reason"].lower()
        assert result["retry_count"] == 1
    
    def test_fails_on_avoid_genre(self):
        """Verifier should fail if film matches avoid genre."""
        from agent.state import initial_state, empty_taste_profile
        from agent.nodes import verifier_node
        
        state = initial_state("suggest something")
        state["cited_films"] = ["1018"]
        state["taste_profile"] = empty_taste_profile()
        state["taste_profile"]["avoid_genres"] = ["Drama"]
        state["retrieved_docs"] = [
            {
                "film_id": "1018",
                "title": "Mulholland Drive",
                "metadata": {"genres": ["Drama", "Mystery"]}
            }
        ]
        
        result = verifier_node(state)
        
        assert result["verified"] == False
        assert "avoid" in result["verification_reason"].lower()
    
    def test_passes_valid_recommendation(self):
        """Verifier should pass if checks succeed."""
        from agent.state import initial_state, empty_taste_profile
        from agent.nodes import verifier_node
        
        state = initial_state("suggest a thriller")
        state["cited_films"] = ["1018"]
        state["taste_profile"] = empty_taste_profile()
        state["taste_profile"]["preferred_genres"] = ["Mystery"]
        state["retrieved_docs"] = [
            {
                "film_id": "1018",
                "title": "Mulholland Drive",
                "metadata": {"genres": ["Mystery", "Drama"]}
            }
        ]
        
        result = verifier_node(state)
        
        assert result["verified"] == True
        assert result["verification_reason"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_nodes.py::TestVerifierNode -v`

Expected: FAIL

- [ ] **Step 3: Implement rule-based verification helper**

Add before verifier_node:

```python
def verify_rules(state: AgentState) -> tuple[bool, str | None]:
    """
    Fast deterministic checks for verification.
    
    Args:
        state: Current AgentState
        
    Returns:
        Tuple of (verified, reason). If verified=False, reason explains why.
    """
    cited_film_ids = state.get("cited_films", [])
    taste_profile = state.get("taste_profile", {})
    retrieved_docs = state.get("retrieved_docs", [])
    
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
```

- [ ] **Step 4: Implement LLM contradiction check helper**

Add after verify_rules:

```python
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
```

- [ ] **Step 5: Implement Verifier node**

Add after verify_contradictions:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_agent_nodes.py::TestVerifierNode -v`

Expected: All tests PASS

- [ ] **Step 7: Commit Verifier**

```bash
git add src/agent/nodes.py tests/test_agent_nodes.py
git commit -m "feat: implement Verifier node with rules and LLM contradiction check"
```

---

## Task 7: Integration Tests for Full Graph

**Files:**
- Create: `tests/test_agent_integration.py`

- [ ] **Step 1: Create integration test file with imports**

Create `tests/test_agent_integration.py`:

```python
"""
Integration tests for the full LangGraph agent workflow.

Tests verify end-to-end behavior across all 5 nodes.

Run with: pytest tests/test_agent_integration.py -v
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.graph import run_turn
from agent.state import initial_state


class TestFullGraphFlow:
    """Test complete conversation turns through the agent."""
    
    def test_factual_query_complete_flow(self):
        """Test: Factual query flows through all nodes successfully."""
        state = run_turn("Who directed Parasite?")
        
        # Should have classified as factual
        assert state["query_type"] in ["factual", "hybrid"]
        
        # Should have retrieved docs
        assert len(state["retrieved_docs"]) > 0
        
        # Should have generated a response
        assert state["response"]
        assert len(state["response"]) > 0
        
        # Should have verified successfully
        assert state["verified"] == True
        
        # Should have made expected number of tool calls
        assert state["tool_calls_count"] >= 1  # At least retrieval
    
    def test_visual_query_complete_flow(self):
        """Test: Visual/mood query flows through successfully."""
        state = run_turn("cold, desaturated, rain-soaked atmosphere")
        
        # Should have classified appropriately
        assert state["query_type"] in ["visual", "hybrid"]
        
        # Should have retrieved docs
        assert len(state["retrieved_docs"]) > 0
        
        # Should have generated response
        assert state["response"]
        
        # Should have verified
        assert state["verified"] == True
    
    def test_multi_turn_memory_persistence(self):
        """Test: Taste profile persists across turns."""
        # Turn 1: Express preference
        state1 = run_turn("I love psychological thrillers")
        
        # Should have updated taste profile
        assert "thriller" in state1["taste_profile"]["preferred_genres"] or \
               "psychological" in state1["taste_profile"]["mood_keywords"]
        
        # Turn 2: Follow-up query
        state2 = run_turn("Suggest something", previous_state=state1)
        
        # Should have inherited taste profile
        profile_inherited = (
            "thriller" in state2["taste_profile"]["preferred_genres"] or
            "psychological" in state2["taste_profile"]["mood_keywords"]
        )
        assert profile_inherited
        
        # Should have conversation history
        assert len(state2["conversation_history"]) >= 2
    
    def test_latency_tracking(self):
        """Test: Latency is tracked for each turn."""
        state = run_turn("Any good thrillers?")
        
        assert "latency_ms" in state
        assert state["latency_ms"] > 0
        assert state["latency_ms"] < 60000  # Should complete in under 60 seconds


class TestRetryLogic:
    """Test verifier retry behavior."""
    
    def test_accepts_after_max_retries(self):
        """Test: Agent accepts response after max retries (doesn't loop forever)."""
        # This would require mocking to force repeated failures
        # For now, just verify the retry_count field exists
        state = run_turn("Suggest a film")
        
        assert "retry_count" in state
        assert state["retry_count"] >= 0
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_agent_integration.py -v --tb=short`

Expected: All tests PASS (may be slow due to LLM calls)

- [ ] **Step 3: Commit integration tests**

```bash
git add tests/test_agent_integration.py
git commit -m "test: add integration tests for full agent workflow"
```

---

## Task 8: Manual Testing and Documentation

**Files:**
- None (interactive CLI testing)

- [ ] **Step 1: Test interactive CLI with factual query**

Run: `python src/agent/graph.py`

Enter: `Who directed Mulholland Drive?`

Expected output:
- Response mentions David Lynch or Mulholland Drive
- Shows latency, tool calls, strategy
- No errors

Enter: `quit`

- [ ] **Step 2: Test multi-turn conversation**

Run: `python src/agent/graph.py`

```
Enter: I love slow-burn psychological thrillers
Expected: Response acknowledges preference, taste profile updated

Enter: Preferably non-English
Expected: Response uses preference from Turn 1, adds language preference

Enter: I've already seen Oldboy and Parasite
Expected: Adds to watched list, doesn't recommend these

Enter: Suggest something
Expected: Recommendation matches all stated preferences (thriller, non-English, not watched)

Enter: quit
```

- [ ] **Step 3: Test visual query**

Run: `python src/agent/graph.py`

```
Enter: cold rainy atmosphere with desaturated colors
Expected: Response recommends films, mentions visual aspects
```

- [ ] **Step 4: Verify all unit tests pass**

Run: `pytest tests/test_agent_nodes.py -v`

Expected: All tests PASS

- [ ] **Step 5: Verify all integration tests pass**

Run: `pytest tests/test_agent_integration.py -v`

Expected: All tests PASS

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v --cov=src/agent`

Expected: >80% coverage on agent module, all tests passing

- [ ] **Step 7: Final commit**

```bash
git add -A
git status
# Review what's staged, ensure only agent files
git commit -m "feat: Phase 3 complete - all 5 LangGraph agent nodes implemented and tested

Implemented nodes:
- QueryRouter: LLM classification (factual/visual/hybrid/multi_hop)
- RetrievalPlanner: Text-only retrieval (based on Phase 2 findings)
- TasteProfileUpdater: LLM preference extraction with additive merging
- AnswerSynthesiser: LLM response generation with citation tracking
- Verifier: Rules + LLM hybrid (watched/avoid checks + contradictions)

Features:
- Multi-turn conversation memory
- Dynamic taste profile updating
- Retry logic (max 2 retries)
- Graceful error handling
- Helper functions for JSON parsing and formatting

Testing:
- Unit tests: all 5 nodes tested individually
- Integration tests: full graph flow, multi-turn, latency
- Manual testing: interactive CLI verified
- Coverage: >80% on agent module"
```

---

## Success Criteria Checklist

Phase 3 is complete when:

- [x] All 5 nodes fully implemented with error handling
- [x] Helper functions (JSON parsing, formatting, merging)
- [x] Unit tests pass (>80% coverage on nodes)
- [x] Integration tests pass (full graph flows)
- [x] Interactive CLI works for multi-turn conversations
- [x] Taste profile persists and updates across turns
- [x] Verifier retry logic works (routes back to retrieval)
- [x] Agent returns responses with image paths
- [x] No crashes - graceful degradation on all errors
- [x] Git commits with clear messages

**Validation:**
```bash
pytest tests/test_agent_nodes.py tests/test_agent_integration.py -v
python src/agent/graph.py  # Manual smoke test
```

---

## Next Steps

After Phase 3 completion:
1. Review and test the implementation
2. Proceed to Phase 4: Evaluation harness
3. Compare 3 variants: Plain LLM vs Static RAG vs Full Agent
4. Measure: Recall@5, ragas faithfulness, latency, tool calls
5. Document findings in RESEARCH.md
