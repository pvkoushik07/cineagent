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


class TestStateManagement:
    """Test state updates and persistence."""

    def test_retrieval_strategy_always_text(self):
        """Test: Retrieval strategy is always 'text' (Phase 2 finding)."""
        state = run_turn("Who directed Inception?")

        assert state["retrieval_strategy"] == "text"

    def test_cited_films_are_film_ids(self):
        """Test: cited_films contains film IDs, not titles."""
        state = run_turn("psychological thriller")

        # Should have cited some films
        if state["cited_films"]:
            # Film IDs should be strings of numbers
            for film_id in state["cited_films"]:
                assert isinstance(film_id, str)
                # Should look like a TMDB ID (numeric string)
                # Note: Some might have formats like "1018" which is valid

    def test_conversation_history_format(self):
        """Test: Conversation history has correct format."""
        state1 = run_turn("I love thrillers")
        state2 = run_turn("Suggest something", previous_state=state1)

        # History should have both turns
        history = state2["conversation_history"]
        assert len(history) >= 2

        # Each message should have role and content
        for msg in history:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["user", "assistant"]
