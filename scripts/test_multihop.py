"""Quick test for multi-hop query performance after Track 2 Step 1."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.test_suite import MULTIHOP_TESTS
from agent.graph import run_turn

print("Testing Track 2 Step 1: k=200 for multi-hop queries\n")
print("=" * 70)

results = []
for test in MULTIHOP_TESTS:
    print(f"\n{test.query_id}: {test.query}")
    print(f"Ground truth: {test.ground_truth_titles}")

    try:
        state = run_turn(test.query, previous_state=None)
        retrieved_film_ids = [r["film_id"] for r in state["retrieved_docs"]]

        # Check if any ground truth film in top-5
        top_5_ids = retrieved_film_ids[:5]
        hit = any(gt_id in top_5_ids for gt_id in test.ground_truth_film_ids)

        results.append({
            "query_id": test.query_id,
            "hit": hit,
            "retrieved_count": len(retrieved_film_ids),
            "top_5_ids": top_5_ids,
        })

        print(f"Retrieved: {len(retrieved_film_ids)} docs")
        print(f"Top-5 IDs: {top_5_ids}")
        print(f"Hit: {'✓' if hit else '✗'}")

    except Exception as e:
        print(f"Error: {e}")
        results.append({"query_id": test.query_id, "hit": False, "error": str(e)})

print("\n" + "=" * 70)
print("SUMMARY:")
hits = sum(1 for r in results if r.get("hit", False))
print(f"Multi-hop Recall@5: {hits}/{len(MULTIHOP_TESTS)} ({100*hits/len(MULTIHOP_TESTS):.1f}%)")
