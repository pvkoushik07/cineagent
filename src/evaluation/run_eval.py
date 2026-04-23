"""
Evaluation Harness — runs all system variants and saves results.

Compares:
  - Variant A: plain LLM (no retrieval)
  - Variant B: fixed RAG pipeline
  - Variant C: full CineAgent

Ablations:
  - Ablation 1: text-only vs caption-only vs CLIP-only vs hybrid
  - Ablation 2: no-memory vs static-memory vs dynamic-taste-updater

Results saved to data/results/eval_results.json.
Visualised in notebooks/04_evaluation.ipynb.

Usage:
    python src/evaluation/run_eval.py --all
    python src/evaluation/run_eval.py --variant A
    python src/evaluation/run_eval.py --ablation 1
"""

import argparse
import json
import logging
from pathlib import Path

import google.generativeai as genai

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GEMINI_API_KEY, GEMINI_MODEL, EVAL_RESULTS_FILE, TOP_K
from evaluation.test_suite import (
    get_all_single_turn_tests,
    get_conversational_tests,
    get_tests_by_family,
    TestCase,
)
from evaluation.metrics import (
    recall_at_k,
    compute_ragas_faithfulness,
    aggregate_metrics,
    LatencyTimer,
)
from retrieval.text_retriever import TextRetriever
from retrieval.clip_retriever import CLIPRetriever
from retrieval.caption_retriever import CaptionRetriever
from retrieval.hybrid_retriever import HybridRetriever
from agent.graph import run_turn
from agent.state import initial_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
genai.configure(api_key=GEMINI_API_KEY)


# ── Variant A: Plain LLM ──────────────────────────────────────────────────────

def run_plain_llm(test: TestCase) -> dict:
    """
    Run Variant A: plain Gemini Flash with no retrieval, no memory.

    Args:
        test: TestCase to evaluate

    Returns:
        Result dict with metrics
    """
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = f"Recommend a film that matches this description: {test.query}"

    with LatencyTimer() as t:
        try:
            response = model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            logger.error(f"Plain LLM failed: {e}")
            answer = ""

    # Plain LLM has no retrieval — Recall@k is always 0
    # (it can't retrieve the correct document if it doesn't search)
    faithfulness = compute_ragas_faithfulness(
        query=test.query,
        response=answer,
        retrieved_contexts=[],  # No retrieved context
    )

    return {
        "query_id": test.query_id,
        "query_family": test.query_family,
        "variant": "A_plain_llm",
        "response": answer,
        "retrieved_film_ids": [],
        "ground_truth_film_ids": test.ground_truth_film_ids,
        "recall_at_5": 0.0,  # No retrieval = no recall
        "faithfulness": faithfulness,
        "latency_ms": t.elapsed_ms,
        "tool_calls_count": 1,
    }


# ── Variant B: Fixed RAG Pipeline ────────────────────────────────────────────

def run_fixed_rag(test: TestCase, retriever: HybridRetriever) -> dict:
    """
    Run Variant B: fixed hybrid RAG pipeline, no routing, no memory.

    Args:
        test: TestCase to evaluate
        retriever: Pre-initialised HybridRetriever

    Returns:
        Result dict with metrics
    """
    with LatencyTimer() as t:
        results = retriever.retrieve(test.query)
        retrieved_film_ids = [r["film_id"] for r in results]
        retrieved_contexts = [r["content"] for r in results if r.get("content")]

        # Simple synthesis without agent
        model = genai.GenerativeModel(GEMINI_MODEL)
        context_str = "\n\n".join(retrieved_contexts[:3])
        prompt = (
            f"Based on these films:\n{context_str}\n\n"
            f"Answer: {test.query}\n"
            f"Recommend from the films above only."
        )
        try:
            response = model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = ""

    faithfulness = compute_ragas_faithfulness(
        query=test.query,
        response=answer,
        retrieved_contexts=retrieved_contexts,
    )

    return {
        "query_id": test.query_id,
        "query_family": test.query_family,
        "variant": "B_fixed_rag",
        "response": answer,
        "retrieved_film_ids": retrieved_film_ids,
        "ground_truth_film_ids": test.ground_truth_film_ids,
        "recall_at_5": recall_at_k(retrieved_film_ids, test.ground_truth_film_ids, k=5),
        "faithfulness": faithfulness,
        "latency_ms": t.elapsed_ms,
        "tool_calls_count": 2,  # retrieve + synthesise
    }


# ── Variant C: Full CineAgent ─────────────────────────────────────────────────

def run_full_agent(test: TestCase) -> dict:
    """
    Run Variant C: full CineAgent with routing, memory, and verification.

    Args:
        test: TestCase to evaluate

    Returns:
        Result dict with metrics
    """
    with LatencyTimer() as t:
        state = run_turn(test.query, previous_state=None)

    retrieved_film_ids = [r["film_id"] for r in state["retrieved_docs"]]
    retrieved_contexts = [r["content"] for r in state["retrieved_docs"] if r.get("content")]

    faithfulness = compute_ragas_faithfulness(
        query=test.query,
        response=state["response"],
        retrieved_contexts=retrieved_contexts,
    )

    return {
        "query_id": test.query_id,
        "query_family": test.query_family,
        "variant": "C_full_agent",
        "response": state["response"],
        "retrieved_film_ids": retrieved_film_ids,
        "ground_truth_film_ids": test.ground_truth_film_ids,
        "recall_at_5": recall_at_k(retrieved_film_ids, test.ground_truth_film_ids, k=5),
        "faithfulness": faithfulness,
        "latency_ms": state["latency_ms"],
        "tool_calls_count": state["tool_calls_count"],
        "query_type_classified": state["query_type"],
        "retrieval_strategy_used": state["retrieval_strategy"],
        "verified": state["verified"],
    }


# ── Ablation 1: Retrieval Design ──────────────────────────────────────────────

def run_retrieval_ablation(tests: list[TestCase]) -> list[dict]:
    """
    Ablation 1: Compare text-only, caption-only, CLIP-only, hybrid.

    Tests only on visual and factual query families where the difference
    between retrieval strategies is most pronounced.

    Args:
        tests: List of TestCase objects (filtered to visual + factual)

    Returns:
        List of result dicts with variant label
    """
    text_ret = TextRetriever()
    caption_ret = CaptionRetriever()
    clip_ret = CLIPRetriever()
    hybrid_ret = HybridRetriever()

    retrievers = {
        "text_only": text_ret,
        "caption_only": caption_ret,
        "clip_only": clip_ret,
        "hybrid_rrf": hybrid_ret,
    }

    results = []
    model = genai.GenerativeModel(GEMINI_MODEL)

    for test in tests:
        for variant_name, retriever in retrievers.items():
            with LatencyTimer() as t:
                if variant_name == "clip_only":
                    raw_results = clip_ret.retrieve_by_text(test.query)
                elif variant_name == "hybrid_rrf":
                    raw_results = hybrid_ret.retrieve(test.query)
                else:
                    raw_results = retriever.retrieve(test.query)

            retrieved_film_ids = [r["film_id"] for r in raw_results]

            results.append({
                "query_id": test.query_id,
                "query_family": test.query_family,
                "variant": f"ablation1_{variant_name}",
                "retrieved_film_ids": retrieved_film_ids,
                "ground_truth_film_ids": test.ground_truth_film_ids,
                "recall_at_5": recall_at_k(retrieved_film_ids, test.ground_truth_film_ids),
                "latency_ms": t.elapsed_ms,
                "tool_calls_count": 1,
                "faithfulness": -1,  # Not measured in retrieval ablation
            })

    return results


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_all_evaluations() -> dict:
    """
    Run all variants and ablations, save results to disk.

    Returns:
        Full results dict
    """
    all_tests = get_all_single_turn_tests()
    visual_factual_tests = [
        t for t in all_tests if t.query_family in ("visual", "factual")
    ]

    hybrid_retriever = HybridRetriever()
    full_results = {}

    logger.info("=== Running Variant A: Plain LLM ===")
    variant_a = [run_plain_llm(t) for t in all_tests]
    full_results["variant_A"] = {
        "per_query": variant_a,
        "summary": aggregate_metrics(variant_a),
    }

    logger.info("=== Running Variant B: Fixed RAG ===")
    variant_b = [run_fixed_rag(t, hybrid_retriever) for t in all_tests]
    full_results["variant_B"] = {
        "per_query": variant_b,
        "summary": aggregate_metrics(variant_b),
    }

    logger.info("=== Running Variant C: Full CineAgent ===")
    variant_c = [run_full_agent(t) for t in all_tests]
    full_results["variant_C"] = {
        "per_query": variant_c,
        "summary": aggregate_metrics(variant_c),
    }

    logger.info("=== Running Ablation 1: Retrieval Design ===")
    ablation_1 = run_retrieval_ablation(visual_factual_tests)
    full_results["ablation_1"] = {
        "per_query": ablation_1,
        "summary": aggregate_metrics(ablation_1),
    }

    # Save results
    with open(EVAL_RESULTS_FILE, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Results saved to {EVAL_RESULTS_FILE}")

    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CineAgent Evaluation Harness")
    parser.add_argument("--all", action="store_true", help="Run all variants and ablations")
    parser.add_argument("--variant", choices=["A", "B", "C"], help="Run a single variant")
    parser.add_argument("--ablation", choices=["1", "2"], help="Run a single ablation")
    args = parser.parse_args()

    if args.all or (not args.variant and not args.ablation):
        results = run_all_evaluations()
    elif args.variant == "A":
        tests = get_all_single_turn_tests()
        results = [run_plain_llm(t) for t in tests]
        print(json.dumps(aggregate_metrics(results), indent=2))
    elif args.variant == "B":
        tests = get_all_single_turn_tests()
        ret = HybridRetriever()
        results = [run_fixed_rag(t, ret) for t in tests]
        print(json.dumps(aggregate_metrics(results), indent=2))
    elif args.variant == "C":
        tests = get_all_single_turn_tests()
        results = [run_full_agent(t) for t in tests]
        print(json.dumps(aggregate_metrics(results), indent=2))
    elif args.ablation == "1":
        tests = get_all_single_turn_tests()
        visual_factual = [t for t in tests if t.query_family in ("visual", "factual")]
        results = run_retrieval_ablation(visual_factual)
        print(json.dumps(aggregate_metrics(results), indent=2))
