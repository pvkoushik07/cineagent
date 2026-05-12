"""Enrich priority films for testing."""
import sys
sys.path.insert(0, 'src')
import logging
logging.basicConfig(level=logging.INFO)

from pipeline.enrich_kb import enrich_film_data
from pathlib import Path
from config import RAW_DIR

# Priority films: all ground truth from test suite
priority_ids = [
    # Factual ground truth
    "1018", "496243", "670", "6977", "62",
    # Visual ground truth
    "335984", "14066", "353081", "510", "77338", "604", "9693",
    # Multi-hop ground truth (CORRECTED IDs)
    "496243", "517814", "1949", "314365", "8967", "16642",
]

print(f'Enriching {len(set(priority_ids))} priority films...')

for film_id in set(priority_ids):
    film_path = RAW_DIR / f"{film_id}.json"
    if film_path.exists():
        try:
            enrich_film_data(film_path)
        except Exception as e:
            print(f'Failed {film_id}: {e}')
    else:
        print(f'Missing: {film_id}')

print('Priority enrichment complete!')
