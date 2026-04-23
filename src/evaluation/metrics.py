"""
Evaluation Metrics — Recall@k, ragas faithfulness, latency, tool calls.

All metric functions take standardised inputs and return float scores.
The run_eval.py script calls these after each system variant run.

Metrics used (per RESEARCH.md):
  - Recall@5: retrieval quality
  - ragas faithfulness: answer groundedness (LLM-as-judge)
  - latency_ms: efficiency
  - tool_calls_count: efficiency
"""

import logging
import time
from pathlib import Path
from typing import Callable

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ── Recall@k ──────────────────────────────────────────────────────────────────

def recall_at_k(
    retrieved_film_ids: list[str],
    ground_truth_film_ids: list[str],
    k: int = 5,
) -> float:
    """
    Compute Recall@k for a single query.

    Recall@k = 1.0 if ANY ground truth film appears in the top-k results.
    This is a binary metric per query (0 or 1), averaged across queries.

    Args:
        retrieved_film_ids: Ordered list of retrieved film IDs (top-k first)
        ground_truth_film_ids: List of correct answer film IDs
        k: Cutoff rank

    Returns:
        1.0 if any ground truth film is in top-k, 0.0 otherwise
    """
    top_k = set(retrieved_film_ids[:k])
    ground_truth = set(ground_truth_film_ids)
    return 1.0 if top_k & ground_truth else 0.0


def mean_recall_at_k(
    results: list[dict],
    k: int = 5,
) -> float:
    """
    Compute mean Recall@k across multiple query results.

    Args:
        results: List of dicts with keys:
                 "retrieved_film_ids" and "ground_truth_film_ids"
        k: Cutoff rank

    Returns:
        Mean Recall@k as a float 0.0–1.0
    """
    if not results:
        return 0.0
    scores = [
        recall_at_k(r["retrieved_film_ids"], r["ground_truth_film_ids"], k)
        for r in results
    ]
    return sum(scores) / len(scores)


def mrr(
    results: list[dict],
) -> float:
    """
    Compute Mean Reciprocal Rank across multiple query results.

    MRR = mean of 1/rank where rank is position of first correct result.
    Returns 0 if no correct result found.

    Args:
        results: List of dicts with "retrieved_film_ids" and "ground_truth_film_ids"

    Returns:
        MRR score as float 0.0–1.0
    """
    reciprocal_ranks = []
    for r in results:
        ground_truth = set(r["ground_truth_film_ids"])
        rank = 0
        for i, film_id in enumerate(r["retrieved_film_ids"], start=1):
            if film_id in ground_truth:
                rank = i
                break
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


# ── ragas Faithfulness ────────────────────────────────────────────────────────

def compute_ragas_faithfulness(
    query: str,
    response: str,
    retrieved_contexts: list[str],
) -> float:
    """
    Compute ragas faithfulness score for a single query-response pair.

    Faithfulness measures whether the response is grounded in the
    retrieved context (not hallucinated). Score range: 0.0–1.0.

    Configured to use Gemini Flash as the judge LLM (not OpenAI default).
    See ARCHITECTURE.md ADR-009.

    Args:
        query: The user query
        response: The generated response
        retrieved_contexts: List of retrieved document text strings

    Returns:
        Faithfulness score 0.0–1.0, or -1.0 if scoring failed
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from ragas.llms import LangchainLLMWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import GEMINI_API_KEY, GEMINI_MODEL

        # Configure ragas to use Gemini Flash instead of OpenAI
        gemini_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
        )
        faithfulness.llm = LangchainLLMWrapper(gemini_llm)

        dataset = Dataset.from_dict({
            "question": [query],
            "answer": [response],
            "contexts": [retrieved_contexts],
        })

        result = evaluate(dataset, metrics=[faithfulness])
        score = result["faithfulness"]
        return float(score) if score is not None else 0.0

    except Exception as e:
        logger.warning(f"ragas faithfulness scoring failed: {e}")
        return -1.0  # Sentinel for failed scoring


# ── Latency Measurement ───────────────────────────────────────────────────────

class LatencyTimer:
    """
    Context manager for measuring execution latency.

    Usage:
        with LatencyTimer() as t:
            result = run_agent(query)
        print(t.elapsed_ms)
    """

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "LatencyTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_metrics(per_query_results: list[dict]) -> dict:
    """
    Aggregate per-query metrics into a summary report.

    Args:
        per_query_results: List of per-query result dicts, each containing:
            query_id, query_family, recall_at_5, faithfulness,
            latency_ms, tool_calls_count

    Returns:
        Summary dict with mean metrics per family and overall
    """
    if not per_query_results:
        return {}

    families = set(r["query_family"] for r in per_query_results)
    summary = {}

    for family in families:
        family_results = [r for r in per_query_results if r["query_family"] == family]
        summary[family] = {
            "n_queries": len(family_results),
            "recall_at_5": round(
                sum(r["recall_at_5"] for r in family_results) / len(family_results), 3
            ),
            "faithfulness": round(
                sum(r["faithfulness"] for r in family_results if r["faithfulness"] >= 0)
                / max(1, sum(1 for r in family_results if r["faithfulness"] >= 0)),
                3,
            ),
            "mean_latency_ms": round(
                sum(r["latency_ms"] for r in family_results) / len(family_results), 1
            ),
            "mean_tool_calls": round(
                sum(r["tool_calls_count"] for r in family_results) / len(family_results), 1
            ),
        }

    # Overall summary
    all_recall = [r["recall_at_5"] for r in per_query_results]
    all_faith = [r["faithfulness"] for r in per_query_results if r["faithfulness"] >= 0]
    all_latency = [r["latency_ms"] for r in per_query_results]
    all_tools = [r["tool_calls_count"] for r in per_query_results]

    summary["overall"] = {
        "n_queries": len(per_query_results),
        "recall_at_5": round(sum(all_recall) / len(all_recall), 3),
        "faithfulness": round(sum(all_faith) / len(all_faith), 3) if all_faith else -1,
        "mean_latency_ms": round(sum(all_latency) / len(all_latency), 1),
        "mean_tool_calls": round(sum(all_tools) / len(all_tools), 1),
    }

    return summary
