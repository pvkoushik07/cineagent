"""
Unit tests for all 5 LangGraph agent nodes.

Tests verify that each node:
  1. Returns a dict (partial state update)
  2. Contains the expected keys
  3. Handles error conditions gracefully (no crashes)

Run with: pytest tests/test_agent_nodes.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.state import AgentState, initial_state, empty_taste_profile


# ── Shared Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> AgentState:
    """A minimal but complete AgentState for testing."""
    state = initial_state("Who directed Mulholland Drive?")
    state["retrieved_docs"] = [
        {
            "doc_id": "1018_plot",
            "film_id": "1018",
            "title": "Mulholland Drive",
            "modality": "text",
            "content": "Mulholland Drive (2001). Directed by David Lynch.",
            "score": 0.92,
            "metadata": {"film_id": "1018", "title": "Mulholland Drive", "year": 2001},
        }
    ]
    state["retrieved_images"] = []
    return state


@pytest.fixture
def mock_gemini():
    """Mock Gemini GenerativeModel returning a fixed response."""
    mock_response = MagicMock()
    mock_response.text = '{"query_type": "factual", "retrieval_strategy": "text", "reasoning": "asking for director"}'
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    return mock_model


# ── Node 1: QueryRouter ───────────────────────────────────────────────────────

class TestQueryRouterNode:

    @patch("agent.nodes.genai.GenerativeModel")
    def test_returns_query_type(self, mock_genai_cls, base_state, mock_gemini):
        mock_genai_cls.return_value = mock_gemini

        from agent.nodes import query_router_node
        result = query_router_node(base_state)

        assert "query_type" in result
        assert result["query_type"] in ("factual", "visual", "hybrid", "multi_hop")

    @patch("agent.nodes.genai.GenerativeModel")
    def test_returns_retrieval_strategy(self, mock_genai_cls, base_state, mock_gemini):
        mock_genai_cls.return_value = mock_gemini

        from agent.nodes import query_router_node
        result = query_router_node(base_state)

        assert "retrieval_strategy" in result
        assert result["retrieval_strategy"] in ("text", "clip", "hybrid")

    @patch("agent.nodes.genai.GenerativeModel")
    def test_increments_tool_calls(self, mock_genai_cls, base_state, mock_gemini):
        mock_genai_cls.return_value = mock_gemini
        initial_calls = base_state["tool_calls_count"]

        from agent.nodes import query_router_node
        result = query_router_node(base_state)

        assert result["tool_calls_count"] == initial_calls + 1

    @patch("agent.nodes.genai.GenerativeModel")
    def test_handles_bad_json_gracefully(self, mock_genai_cls, base_state):
        """Router must not crash if LLM returns malformed JSON."""
        mock_response = MagicMock()
        mock_response.text = "This is not JSON at all."
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai_cls.return_value = mock_model

        from agent.nodes import query_router_node
        result = query_router_node(base_state)

        # Should default to hybrid query type, but always text retrieval (Phase 2 finding)
        assert result["query_type"] == "hybrid"
        assert result["retrieval_strategy"] == "text"


# ── Node 3: TasteProfileUpdater ───────────────────────────────────────────────

class TestTasteProfileUpdaterNode:

    @patch("agent.nodes.genai.GenerativeModel")
    def test_updates_taste_profile(self, mock_genai_cls, base_state):
        updated_profile = {
            "preferred_genres": ["thriller"],
            "preferred_directors": [],
            "preferred_languages": [],
            "year_range": {"min": None, "max": None},
            "avoid_genres": [],
            "watched": [],
            "mood_keywords": ["slow-burn"],
            "confidence": 0.4,
        }
        mock_response = MagicMock()
        mock_response.text = __import__("json").dumps(updated_profile)
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai_cls.return_value = mock_model

        base_state["query"] = "I love slow-burn thrillers"
        from agent.nodes import taste_profile_updater_node
        result = taste_profile_updater_node(base_state)

        assert "taste_profile" in result
        profile = result["taste_profile"]
        assert isinstance(profile, dict)
        assert "preferred_genres" in profile
        assert "confidence" in profile

    @patch("agent.nodes.genai.GenerativeModel")
    def test_keeps_profile_on_failure(self, mock_genai_cls, base_state):
        """If LLM returns bad JSON, keep the existing profile unchanged."""
        mock_response = MagicMock()
        mock_response.text = "INVALID JSON"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai_cls.return_value = mock_model

        original_profile = base_state["taste_profile"].copy()
        from agent.nodes import taste_profile_updater_node
        result = taste_profile_updater_node(base_state)

        assert result["taste_profile"] == original_profile


# ── Node 5: Verifier ──────────────────────────────────────────────────────────

class TestVerifierNode:

    def test_passes_valid_response(self, base_state):
        base_state["response"] = "I recommend Mulholland Drive by David Lynch."
        base_state["cited_films"] = ["Mulholland Drive"]

        from agent.nodes import verifier_node
        result = verifier_node(base_state)

        assert result["verified"] is True

    def test_fails_empty_response(self, base_state):
        base_state["response"] = ""
        base_state["cited_films"] = []

        from agent.nodes import verifier_node
        result = verifier_node(base_state)

        assert result["verified"] is False
        assert result["verification_reason"] == "empty_response"

    def test_fails_if_watched_film_recommended(self, base_state):
        """Verifier must reject recommendations for films already watched."""
        # Add a mock retrieved doc so the verifier can find the film title
        base_state["retrieved_docs"] = [
            {"film_id": "670", "title": "Oldboy", "content": "plot text", "modality": "text"}
        ]
        base_state["response"] = "I recommend Oldboy, a great psychological thriller."
        base_state["cited_films"] = ["670"]  # Film IDs, not titles
        base_state["taste_profile"]["watched"] = ["Oldboy"]  # Watched list has title
        base_state["taste_profile"]["confidence"] = 0.8

        from agent.nodes import verifier_node
        result = verifier_node(base_state)

        assert result["verified"] is False
        assert "already watched" in result["verification_reason"]

    def test_increments_retry_count_on_failure(self, base_state):
        base_state["response"] = ""
        base_state["retry_count"] = 0

        from agent.nodes import verifier_node
        result = verifier_node(base_state)

        assert result["retry_count"] == 1


# ── AgentState Tests ──────────────────────────────────────────────────────────

class TestAgentState:

    def test_initial_state_has_all_keys(self):
        state = initial_state("test query")
        required_keys = {
            "query", "conversation_history", "query_type", "retrieval_strategy",
            "retrieved_docs", "retrieved_images", "taste_profile", "response",
            "cited_films", "verified", "verification_reason", "retry_count",
            "tool_calls_count", "latency_ms",
        }
        assert required_keys.issubset(state.keys())

    def test_empty_taste_profile_has_all_keys(self):
        profile = empty_taste_profile()
        required_keys = {
            "preferred_genres", "preferred_directors", "preferred_languages",
            "year_range", "avoid_genres", "watched", "mood_keywords", "confidence",
        }
        assert required_keys.issubset(profile.keys())
        assert profile["confidence"] == 0.0
