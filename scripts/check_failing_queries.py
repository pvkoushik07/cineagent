"""Quick script to check which queries pass/fail."""
import sys
sys.path.insert(0, 'src')

from evaluation.test_suite import get_all_single_turn_tests
from agent.graph import run_turn

# Suppress logs
import logging
logging.basicConfig(level=logging.ERROR)

tests = get_all_single_turn_tests()

print('Query Results:')
print('=' * 80)

for test in tests:
    state = run_turn(test.query, previous_state=None)
    retrieved_ids = [str(r['film_id']) for r in state['retrieved_docs']]
    ground_truth = [str(x) for x in test.ground_truth_film_ids]

    # Check if any ground truth in retrieved
    hit = any(gt in retrieved_ids for gt in ground_truth)
    status = '✅' if hit else '❌'

    print(f'{status} {test.query_id} ({test.query_family}): {test.query[:50]}')
    print(f'   Expected: {test.ground_truth_titles}')
    if not hit:
        # Show what was retrieved instead
        from retrieval.text_retriever import TextRetriever
        retriever = TextRetriever()
        collection = retriever.client.get_collection('cineagent_text')

        retrieved_titles = []
        for rid in retrieved_ids[:3]:
            docs = collection.get(where={'film_id': int(rid)}, include=['metadatas'], limit=1)
            if docs['metadatas']:
                retrieved_titles.append(docs['metadatas'][0].get('title', 'Unknown'))

        print(f'   Got: {retrieved_titles[:3]}')
    print()

print('\nSummary by Family:')
print('=' * 80)
families = {}
for test in tests:
    if test.query_family not in families:
        families[test.query_family] = {'total': 0, 'passed': 0}
    families[test.query_family]['total'] += 1

    state = run_turn(test.query, previous_state=None)
    retrieved_ids = [str(r['film_id']) for r in state['retrieved_docs']]
    ground_truth = [str(x) for x in test.ground_truth_film_ids]
    if any(gt in retrieved_ids for gt in ground_truth):
        families[test.query_family]['passed'] += 1

for family, stats in families.items():
    pct = 100 * stats['passed'] / stats['total']
    print(f'{family}: {pct:.1f}% ({stats["passed"]}/{stats["total"]})')
