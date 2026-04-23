# Run Evaluation Suite

Runs all system variants against the fixed test suite and saves results.

## Pre-conditions (Check Before Running)

1. Knowledge base must be built — check data/indices/ exists
2. All 4 retrieval variants must be importable — run `pytest tests/test_retrievers.py`
3. Agent must be working — run `pytest tests/test_agent_nodes.py`
4. GEMINI_API_KEY must be set in .env (ragas needs it)

## Steps

1. Run `python src/evaluation/run_eval.py --variant A` (plain LLM baseline)
2. Run `python src/evaluation/run_eval.py --variant B` (fixed RAG pipeline)
3. Run `python src/evaluation/run_eval.py --variant C` (full CineAgent)
4. Run `python src/evaluation/run_eval.py --ablation 1` (retrieval variants)
5. Run `python src/evaluation/run_eval.py --ablation 2` (memory variants)
6. Results saved to data/results/eval_results.json
7. Open notebooks/04_evaluation.ipynb to visualise results

## Metrics Reported

Per variant, per query family:
- Recall@5 (%)
- ragas faithfulness score (0.0–1.0)
- Mean latency (ms)
- Mean tool calls per query

## Important

Do NOT change test queries in src/evaluation/test_suite.py after first run.
Results become incomparable if queries change.
