"""
Two-Stage Retriever Evaluation Script

Evaluates the two-stage retriever with query-type aware routing and fusion weights.

Usage:
    python src/evaluation/run_two_stage_eval.py --variant exp_visual-extreme-clip
    python src/evaluation/run_two_stage_eval.py --output data/results/exp_visual-extreme-clip.json
"""

import argparse
import json
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.test_suite import get_all_single_turn_tests
from evaluation.metrics import recall_at_k, aggregate_metrics, LatencyTimer
from retrieval.two_stage_retriever import TwoStageRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_two_stage_evaluation(variant_name: str = "exp_visual-extreme-clip") -> dict:
    """
    Evaluate the two-stage retriever on all test queries.

    Args:
        variant_name: Name for the variant (used in results)

    Returns:
        Results dict with per-query and summary metrics
    """
    tests = get_all_single_turn_tests()
    retriever = TwoStageRetriever(top_k=5, candidate_k=20)

    results = []

    for test in tests:
        with LatencyTimer() as t:
            raw_results = retriever.retrieve(
                query=test.query,
                query_type=test.query_family,
            )

        retrieved_film_ids = [r["film_id"] for r in raw_results]

        results.append({
            "query_id": test.query_id,
            "query_family": test.query_family,
            "variant": variant_name,
            "retrieved_film_ids": retrieved_film_ids,
            "ground_truth_film_ids": test.ground_truth_film_ids,
            "recall_at_5": recall_at_k(retrieved_film_ids, test.ground_truth_film_ids, k=5),
            "latency_ms": t.elapsed_ms,
            "tool_calls_count": 2,
            "faithfulness": -1,
        })

        logger.info(
            f"{test.query_id} ({test.query_family}): "
            f"Recall={results[-1]['recall_at_5']:.2f}, "
            f"Latency={t.elapsed_ms:.1f}ms"
        )

    return {
        variant_name: {
            "per_query": results,
            "summary": aggregate_metrics(results),
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Stage Retriever Evaluation")
    parser.add_argument(
        "--variant",
        default="exp_visual-extreme-clip",
        help="Variant name for results"
    )
    parser.add_argument(
        "--output",
        default="data/results/exp_visual-extreme-clip.json",
        help="Output file path"
    )
    args = parser.parse_args()

    logger.info(f"Running two-stage retriever evaluation: {args.variant}")
    full_results = run_two_stage_evaluation(variant_name=args.variant)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    summary = full_results[args.variant]["summary"]
    print("\n=== EVALUATION SUMMARY ===")
    for family, metrics in summary.items():
        print(f"\n{family.upper()}:")
        print(f"  Recall@5: {metrics['recall_at_5']:.1%}")
        print(f"  N Queries: {metrics['n_queries']}")
        print(f"  Latency: {metrics['mean_latency_ms']:.1f}ms")
