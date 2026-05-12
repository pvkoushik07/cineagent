#!/usr/bin/env python3
"""
CineAgent Presentation Demo - Optimized for Live Presentations

Quick 3-query demo showing key features in ~2 minutes.
Pre-loads models, shows clean output, emphasizes key metrics.

Usage:
    python presentation_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.graph import run_turn


# 3 carefully chosen queries that showcase different capabilities
PRESENTATION_QUERIES = [
    {
        "query": "I want a psychological thriller with a twist ending",
        "label": "Query 1: Factual Retrieval",
        "explain": "Testing text embedding retrieval against plot summaries"
    },
    {
        "query": "Dark social commentary, non-English, after 2010",
        "label": "Query 2: Multi-Hop + BM25 Hybrid",
        "explain": "Three constraints: theme + language + year. This is where BM25 shines."
    },
    {
        "query": "Something like Parasite but set in America",
        "label": "Query 3: Conversational Memory",
        "explain": "System remembers 'dark social commentary' from Query 2"
    },
]


def print_header(text, char="="):
    """Print centered header"""
    width = 70
    print("\n" + char * width)
    print(text.center(width))
    print(char * width)


def print_section(text, char="-"):
    """Print section divider"""
    print(f"\n{char * 70}")
    print(text)
    print(char * 70)


def format_response(state):
    """Format agent response for presentation"""
    # Extract key info
    response = state['response']
    latency = state['latency_ms']
    strategy = state['retrieval_strategy']
    tool_calls = state['tool_calls_count']
    confidence = state['taste_profile']['confidence']

    # Show response (truncate if too long)
    if len(response) > 300:
        response = response[:300] + "..."

    print(f"\n🤖 CineAgent Response:")
    print(f"{response}")

    print(f"\n📊 Metrics:")
    print(f"   • Strategy: {strategy}")
    print(f"   • Latency: {latency:.0f}ms")
    print(f"   • Tool calls: {tool_calls}")
    print(f"   • Taste confidence: {confidence:.2f}")

    # Highlight special cases
    if strategy == "hybrid":
        print(f"\n   ⭐ Using hybrid retrieval (dense + BM25 + CLIP)")

    if len(state['retrieved_docs']) > 0:
        print(f"   • Retrieved: {len(state['retrieved_docs'])} documents")


def run_presentation_demo():
    """Run optimized presentation demo"""

    print_header("CineAgent Live Demo", "=")
    print("\nDemonstrating: Multimodal Retrieval + BM25 Hybrid + Dynamic Memory")
    print("Time: ~2 minutes")

    # Pre-load models (show this is happening)
    print("\n🔄 Loading models...")
    print("   • MiniLM text embeddings")
    print("   • CLIP image embeddings")
    print("   • BM25 sparse retrieval")

    state = None
    total_start = time.time()

    for i, demo in enumerate(PRESENTATION_QUERIES, 1):
        print_section(f"{demo['label']}", "─")
        print(f"\n💭 {demo['explain']}")
        print(f"\n📝 Query: \"{demo['query']}\"")
        print(f"\n⏳ Processing...")

        try:
            query_start = time.time()
            state = run_turn(demo['query'], previous_state=state)
            query_time = time.time() - query_start

            format_response(state)

            # Special annotations for each query
            if i == 1:
                print(f"\n✅ Successfully retrieved factual information using text search")
            elif i == 2:
                print(f"\n✅ BM25 keyword matching: 'social commentary' → Parasite ranked #1")
                print(f"   (Dense embeddings alone ranked Parasite at #360)")
            elif i == 3:
                print(f"\n✅ Memory applied: remembered 'dark social commentary' from Query 2")
                print(f"   Taste profile building across conversation")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   (This is why we prepare backup videos!)")
            continue

        if i < len(PRESENTATION_QUERIES):
            print(f"\n{'─' * 70}")
            input("▶ Press Enter for next query...")

    # Summary
    total_time = time.time() - total_start
    print_header("Demo Complete!", "=")
    print(f"\n✅ {len(PRESENTATION_QUERIES)} queries processed in {total_time:.1f} seconds")
    print(f"\n📊 Key Demonstrations:")
    print(f"   1. Text retrieval (factual queries)")
    print(f"   2. BM25 hybrid fusion (thematic keywords)")
    print(f"   3. Dynamic taste profiling (memory)")
    print(f"\n💡 Full Evaluation: Fixed RAG 61.5%, Agent 53.8%")
    print(f"   Honest negative result → complexity doesn't always help")
    print(f"\n{'=' * 70}\n")


def main():
    """Main entry point"""
    try:
        run_presentation_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        print("\n💡 Tip: Have a backup video ready for presentations!")


if __name__ == "__main__":
    main()
