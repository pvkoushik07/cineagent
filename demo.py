#!/usr/bin/env python3
"""
CineAgent Interactive Demo

For markers/reviewers to quickly test the system with pre-written queries
or their own custom questions.

Usage:
    python demo.py              # Interactive mode
    python demo.py --demo       # Run demo queries
    python demo.py --quick      # Quick test (3 queries)
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.graph import run_turn, initial_state
from config import GEMINI_API_KEY

# Pre-written demo queries showcasing different capabilities
DEMO_QUERIES = [
    {
        "query": "I'm looking for a psychological thriller with a twist ending",
        "description": "Factual query - tests plot/genre retrieval"
    },
    {
        "query": "Find me something visually stunning with minimal dialogue",
        "description": "Visual query - tests CLIP + caption retrieval"
    },
    {
        "query": "I want a dark social commentary film, non-English, after 2010",
        "description": "Multi-hop query - tests BM25 + metadata filtering"
    },
    {
        "query": "Something like Parasite but set in America",
        "description": "Conversational refinement - tests taste profile memory"
    },
    {
        "query": "Not thrillers though, maybe something more uplifting",
        "description": "Preference update - tests dynamic taste profiling"
    },
]

QUICK_TEST_QUERIES = [
    "Who directed Mulholland Drive?",
    "Find me a cold, desaturated, rain-soaked urban atmosphere film",
    "Dark social commentary, non-English, recent",
]


def run_demo_queries():
    """Run pre-written demo queries with explanations."""
    print("\n" + "="*70)
    print("CineAgent Demo - Pre-written Test Queries")
    print("="*70)
    print("\nThis demonstrates the system's capabilities across different query types.\n")

    state = None

    for i, demo in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}/5: {demo['description']}")
        print(f"{'─'*70}")
        print(f"\n📝 Query: \"{demo['query']}\"")

        try:
            state = run_turn(demo['query'], previous_state=state)

            print(f"\n🤖 CineAgent Response:")
            print(f"{state['response']}")

            print(f"\n📊 Metrics:")
            print(f"   • Strategy: {state['retrieval_strategy']}")
            print(f"   • Latency: {state['latency_ms']:.0f}ms")
            print(f"   • Tool calls: {state['tool_calls_count']}")
            print(f"   • Taste confidence: {state['taste_profile']['confidence']:.2f}")

            if state['retrieved_docs']:
                print(f"   • Retrieved: {len(state['retrieved_docs'])} documents")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

        if i < len(DEMO_QUERIES):
            input("\n▶ Press Enter to continue to next query...")

    print(f"\n{'='*70}")
    print("Demo complete! The agent maintained conversation state across all queries.")
    print("="*70)


def run_quick_test():
    """Run quick 3-query test for basic functionality check."""
    print("\n" + "="*70)
    print("CineAgent Quick Test - 3 Queries")
    print("="*70)

    for i, query in enumerate(QUICK_TEST_QUERIES, 1):
        print(f"\n[{i}/3] Query: \"{query}\"")

        try:
            state = run_turn(query)
            print(f"✅ Success - {state['latency_ms']:.0f}ms")
            print(f"   Response preview: {state['response'][:100]}...")
        except Exception as e:
            print(f"❌ Failed: {e}")

    print(f"\n{'='*70}")
    print("Quick test complete!")
    print("="*70)


def run_interactive():
    """Interactive mode - user enters their own queries."""
    print("\n" + "="*70)
    print("CineAgent Interactive Demo")
    print("="*70)
    print("\nType your film query below. Commands:")
    print("  • 'quit' or Ctrl+C to exit")
    print("  • 'reset' to start a new conversation")
    print("  • 'help' for example queries")
    print("="*70 + "\n")

    state = None

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not query:
            continue

        if query.lower() == "quit":
            print("\nGoodbye! 👋")
            break

        if query.lower() == "reset":
            state = None
            print("\n✨ Conversation reset. Starting fresh!\n")
            continue

        if query.lower() == "help":
            print("\n📝 Example queries to try:")
            for demo in DEMO_QUERIES[:3]:
                print(f"   • {demo['query']}")
            print()
            continue

        try:
            state = run_turn(query, previous_state=state)

            print(f"\n🤖 CineAgent: {state['response']}")
            print(f"\n   [{state['latency_ms']:.0f}ms | {state['tool_calls_count']} calls | "
                  f"{state['retrieval_strategy']} | confidence={state['taste_profile']['confidence']:.2f}]\n")

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def check_setup():
    """Verify system is properly configured."""
    issues = []

    # Check API key
    if not GEMINI_API_KEY:
        issues.append("❌ GEMINI_API_KEY not set in .env file")
    else:
        print("✅ Gemini API key configured")

    # Check knowledge base
    kb_path = Path("data/indices")
    if not kb_path.exists() or not list(kb_path.glob("*")):
        issues.append("❌ Knowledge base not built. Run: python src/pipeline/kb_builder.py")
    else:
        print("✅ Knowledge base found")

    # Check dependencies
    try:
        import chromadb
        import sentence_transformers
        import langgraph
        print("✅ Core dependencies installed")
    except ImportError as e:
        issues.append(f"❌ Missing dependency: {e.name}")

    if issues:
        print("\n⚠️  Setup issues found:")
        for issue in issues:
            print(f"   {issue}")
        print("\nSee README.md for setup instructions.")
        return False

    print("\n✅ System ready!\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="CineAgent Interactive Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py              # Interactive mode
  python demo.py --demo       # Run all 5 demo queries
  python demo.py --quick      # Quick 3-query test
        """
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run pre-written demo queries"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick functionality test (3 queries)"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip setup verification"
    )

    args = parser.parse_args()

    # Verify setup unless skipped
    if not args.no_check:
        if not check_setup():
            sys.exit(1)

    # Run appropriate mode
    if args.demo:
        run_demo_queries()
    elif args.quick:
        run_quick_test()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
