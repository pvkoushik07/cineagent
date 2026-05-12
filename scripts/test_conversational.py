"""
Test conversational query sequences with memory.

Evaluates multi-turn conversations where taste profile should persist
across turns and influence retrieval.
"""

import sys
sys.path.insert(0, 'src')

import json
import logging
from agent.graph import run_turn
from agent.state import initial_state
from evaluation.test_suite import get_conversational_tests
from evaluation.metrics import recall_at_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_conversational_sequence(sequence):
    """Run a multi-turn conversational test sequence."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Sequence: {sequence.sequence_id}")
    logger.info(f"{'='*60}\n")

    state = None
    turn_results = []

    for i, turn_query in enumerate(sequence.turns, 1):
        logger.info(f"Turn {i}: {turn_query}")

        # Run turn with previous state
        state = run_turn(turn_query, previous_state=state)

        # Extract retrieved film IDs
        retrieved_film_ids = [r["film_id"] for r in state.get("retrieved_docs", [])]

        # Calculate recall for final turn
        if i == len(sequence.turns):
            recall = recall_at_k(
                retrieved_film_ids,
                sequence.ground_truth_film_ids,
                k=5
            )

            logger.info(f"Final Turn Results:")
            logger.info(f"  Retrieved: {retrieved_film_ids[:5]}")
            logger.info(f"  Ground Truth: {sequence.ground_truth_film_ids}")
            logger.info(f"  Ground Truth Titles: {sequence.ground_truth_titles}")
            logger.info(f"  Recall@5: {recall}")

        # Log taste profile
        taste_profile = state.get("taste_profile", {})
        logger.info(f"  Taste Profile: {taste_profile}\n")

        turn_results.append({
            "turn": i,
            "query": turn_query,
            "retrieved_ids": retrieved_film_ids[:5],
            "taste_profile": taste_profile,
        })

    # Final evaluation
    final_recall = recall_at_k(
        retrieved_film_ids,
        sequence.ground_truth_film_ids,
        k=5
    )

    result = {
        "sequence_id": sequence.sequence_id,
        "turns": turn_results,
        "final_recall_at_5": final_recall,
        "ground_truth_ids": sequence.ground_truth_film_ids,
        "ground_truth_titles": sequence.ground_truth_titles,
        "response": state.get("response", ""),
    }

    return result


def main():
    """Run all conversational tests."""
    sequences = get_conversational_tests()

    all_results = []

    for seq in sequences:
        try:
            result = test_conversational_sequence(seq)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed sequence {seq.sequence_id}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("CONVERSATIONAL TEST RESULTS")
    print("="*60)

    total_sequences = len(all_results)
    passing_sequences = sum(1 for r in all_results if r["final_recall_at_5"] > 0)
    avg_recall = sum(r["final_recall_at_5"] for r in all_results) / max(total_sequences, 1)

    print(f"\nSequences Tested: {total_sequences}")
    print(f"Sequences Passing: {passing_sequences}/{total_sequences}")
    print(f"Average Recall@5: {avg_recall:.1%}")

    for result in all_results:
        print(f"\n{result['sequence_id']}: Recall@5 = {result['final_recall_at_5']:.1%}")
        print(f"  Ground Truth: {result['ground_truth_titles']}")

    # Save results
    output_file = "data/results/conversational_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")

    return all_results


if __name__ == "__main__":
    main()
