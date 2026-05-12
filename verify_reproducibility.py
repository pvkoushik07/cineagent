#!/usr/bin/env python3
"""
Reproducibility Verification Script

Checks that all components are properly installed and functional.
Run this after setup to verify everything works.

Usage:
    python verify_reproducibility.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_python_version():
    """Verify Python 3.11+"""
    import sys
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (need 3.11+)")
        return False


def check_dependencies():
    """Verify all required packages are installed"""
    required = {
        "chromadb": "ChromaDB",
        "sentence_transformers": "Sentence Transformers",
        "langgraph": "LangGraph",
        "google.generativeai": "Google Generative AI",
        "rank_bm25": "Rank BM25",
    }

    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not installed")
            all_ok = False

    return all_ok


def check_environment():
    """Verify .env file and API keys"""
    from config import GEMINI_API_KEY, TMDB_API_KEY

    all_ok = True

    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        print("✅ Gemini API key configured")
    else:
        print("❌ Gemini API key not set (required for agent)")
        all_ok = False

    if TMDB_API_KEY and TMDB_API_KEY != "your_tmdb_api_key_here":
        print("✅ TMDB API key configured")
    else:
        print("⚠️  TMDB API key not set (needed only for KB building)")

    return all_ok


def check_knowledge_base():
    """Verify KB is built and accessible"""
    from config import CHROMA_PERSIST_DIR, TEXT_COLLECTION_NAME, IMAGE_COLLECTION_NAME
    import chromadb

    kb_path = Path(CHROMA_PERSIST_DIR)

    if not kb_path.exists():
        print("❌ Knowledge base directory not found")
        print(f"   Run: python src/pipeline/kb_builder.py")
        return False

    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Check text collection
        try:
            text_coll = client.get_collection(TEXT_COLLECTION_NAME)
            text_count = text_coll.count()
            print(f"✅ Text collection: {text_count:,} documents")
        except Exception as e:
            print(f"❌ Text collection error: {e}")
            return False

        # Check image collection
        try:
            image_coll = client.get_collection(IMAGE_COLLECTION_NAME)
            image_count = image_coll.count()
            print(f"✅ Image collection: {image_count:,} embeddings")
        except Exception as e:
            print(f"❌ Image collection error: {e}")
            return False

        if text_count < 100:
            print("⚠️  Warning: Text collection seems small (expected ~2600)")

        return True

    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False


def test_retrieval():
    """Test that retrieval components work"""
    try:
        from retrieval.text_retriever import TextRetriever
        from retrieval.hybrid_retriever import HybridRetriever

        # Test text retriever
        text_ret = TextRetriever(top_k=3)
        results = text_ret.retrieve("psychological thriller")

        if results and len(results) > 0:
            print(f"✅ Text retrieval working ({len(results)} results)")
        else:
            print("❌ Text retrieval returned no results")
            return False

        # Test hybrid retriever
        hybrid_ret = HybridRetriever(top_k=3)
        results = hybrid_ret.retrieve("dark thriller", use_clip=False)

        if results and len(results) > 0:
            print(f"✅ Hybrid retrieval working ({len(results)} results)")
        else:
            print("❌ Hybrid retrieval returned no results")
            return False

        return True

    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return False


def test_agent():
    """Test that agent runs end-to-end"""
    try:
        from agent.graph import run_turn

        print("🔄 Running test query through agent...")
        state = run_turn("Who directed Mulholland Drive?")

        if state and state.get("response"):
            print(f"✅ Agent working (responded in {state.get('latency_ms', 0):.0f}ms)")
            return True
        else:
            print("❌ Agent returned no response")
            return False

    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test that evaluation harness works"""
    try:
        from evaluation.test_suite import get_all_single_turn_tests
        from evaluation.metrics import recall_at_k

        tests = get_all_single_turn_tests()

        if len(tests) >= 10:
            print(f"✅ Evaluation suite loaded ({len(tests)} test queries)")
        else:
            print(f"⚠️  Warning: Only {len(tests)} test queries found")

        # Test metrics
        score = recall_at_k(
            retrieved_film_ids=["123", "456"],
            ground_truth_film_ids=["456", "789"],
            k=5
        )

        print(f"✅ Metrics working (recall_at_k={score})")
        return True

    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False


def main():
    print("="*70)
    print("CineAgent Reproducibility Verification")
    print("="*70)
    print()

    results = {}

    print("1. Python Version")
    print("-" * 40)
    results['python'] = check_python_version()
    print()

    print("2. Dependencies")
    print("-" * 40)
    results['dependencies'] = check_dependencies()
    print()

    print("3. Environment Configuration")
    print("-" * 40)
    results['environment'] = check_environment()
    print()

    print("4. Knowledge Base")
    print("-" * 40)
    results['kb'] = check_knowledge_base()
    print()

    print("5. Retrieval Components")
    print("-" * 40)
    results['retrieval'] = test_retrieval()
    print()

    print("6. Agent Workflow")
    print("-" * 40)
    results['agent'] = test_agent()
    print()

    print("7. Evaluation Harness")
    print("-" * 40)
    results['evaluation'] = test_evaluation()
    print()

    print("="*70)
    print("Summary")
    print("="*70)

    passed = sum(results.values())
    total = len(results)

    for component, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {component.replace('_', ' ').title()}")

    print()
    print(f"Result: {passed}/{total} checks passed")

    if passed == total:
        print("\n✅ System is fully reproducible and ready to use!")
        print("\nNext steps:")
        print("  • Run demo: python demo.py")
        print("  • Interactive: python src/agent/graph.py")
        print("  • Evaluate: python src/evaluation/run_eval.py --all")
        return 0
    else:
        print("\n⚠️  Some checks failed. See errors above.")
        print("\nFor setup help, see README.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
