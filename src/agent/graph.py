"""
CineAgent LangGraph StateGraph — the main agent workflow.

Wires all 5 nodes into a directed graph with conditional edges.
The Verifier node can re-route back to RetrievalPlanner on failure
(max 2 retries to prevent infinite loops).

Usage:
    python src/agent/graph.py          # interactive CLI
    from agent.graph import run_turn   # programmatic use
"""

import logging
import time
from pathlib import Path

from langgraph.graph import StateGraph, END

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.state import AgentState, initial_state
from agent.nodes import (
    query_router_node,
    retrieval_planner_node,
    taste_profile_updater_node,
    answer_synthesiser_node,
    verifier_node,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def should_retry(state: AgentState) -> str:
    """
    Conditional edge function after the Verifier node.

    Returns "retry" if verification failed and retry limit not reached.
    Returns "end" otherwise.

    Args:
        state: Current AgentState after Verifier has run

    Returns:
        "retry" or "end"
    """
    if not state["verified"] and state["retry_count"] < MAX_RETRIES:
        logger.info(f"Verifier failed: {state['verification_reason']}. Retrying ({state['retry_count']}/{MAX_RETRIES})")
        return "retry"
    return "end"


def build_graph() -> StateGraph:
    """
    Build and compile the CineAgent LangGraph workflow.

    Graph structure:
        query_router
            ↓
        retrieval_planner
            ↓
        taste_profile_updater
            ↓
        answer_synthesiser
            ↓
        verifier ──(retry)──→ retrieval_planner
            ↓ (end)
           END

    Returns:
        Compiled LangGraph StateGraph
    """
    graph = StateGraph(AgentState)

    # Add all 5 nodes
    graph.add_node("query_router", query_router_node)
    graph.add_node("retrieval_planner", retrieval_planner_node)
    graph.add_node("taste_profile_updater", taste_profile_updater_node)
    graph.add_node("answer_synthesiser", answer_synthesiser_node)
    graph.add_node("verifier", verifier_node)

    # Entry point
    graph.set_entry_point("query_router")

    # Linear edges
    graph.add_edge("query_router", "retrieval_planner")
    graph.add_edge("retrieval_planner", "taste_profile_updater")
    graph.add_edge("taste_profile_updater", "answer_synthesiser")
    graph.add_edge("answer_synthesiser", "verifier")

    # Conditional edge from verifier: retry or end
    graph.add_conditional_edges(
        "verifier",
        should_retry,
        {
            "retry": "retrieval_planner",
            "end": END,
        },
    )

    return graph.compile()


def run_turn(
    query: str,
    previous_state: AgentState | None = None,
) -> AgentState:
    """
    Run a single conversation turn through the CineAgent.

    Carries over taste_profile and conversation_history from previous turns,
    enabling multi-turn personalisation.

    Args:
        query: User's query string
        previous_state: State from the previous turn (for conversation continuity)

    Returns:
        Final AgentState after all nodes have executed
    """
    # Build initial state, inheriting memory from previous turn
    if previous_state:
        state = AgentState(
            query=query,
            conversation_history=previous_state["conversation_history"] + [
                {"role": "assistant", "content": previous_state["response"]}
            ],
            query_type="",
            retrieval_strategy="",
            retrieved_docs=[],
            retrieved_images=[],
            taste_profile=previous_state["taste_profile"],  # ← carry over memory
            response="",
            cited_films=[],
            verified=False,
            verification_reason=None,
            retry_count=0,
            tool_calls_count=0,
            latency_ms=0.0,
        )
    else:
        state = initial_state(query)

    # Add current user message to history
    state["conversation_history"].append({"role": "user", "content": query})

    # Run the graph
    agent = build_graph()
    start_time = time.perf_counter()
    final_state = agent.invoke(state)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    final_state["latency_ms"] = round(elapsed_ms, 2)

    logger.info(
        f"Turn complete: {final_state['tool_calls_count']} tool calls, "
        f"{elapsed_ms:.0f}ms, verified={final_state['verified']}"
    )

    return final_state


def run_interactive() -> None:
    """
    Interactive CLI for conversing with CineAgent.
    Maintains conversation state across turns.
    """
    print("\n=== CineAgent — Personalised Film Discovery ===")
    print("Type your query. Type 'quit' to exit, 'reset' to start a new conversation.\n")

    state: AgentState | None = None

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() == "quit":
            print("Goodbye!")
            break
        if query.lower() == "reset":
            state = None
            print("Conversation reset.\n")
            continue

        state = run_turn(query, previous_state=state)
        print(f"\nCineAgent: {state['response']}")
        print(f"[{state['latency_ms']:.0f}ms | {state['tool_calls_count']} calls | "
              f"strategy={state['retrieval_strategy']} | "
              f"taste_confidence={state['taste_profile']['confidence']:.2f}]\n")


if __name__ == "__main__":
    run_interactive()
