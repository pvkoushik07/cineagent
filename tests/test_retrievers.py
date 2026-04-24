"""
Unit tests for all retrieval variants.

These tests use a tiny mock ChromaDB collection so they run without
the full KB being built. Tests verify the interface contract of each
retriever — not the quality of results (that's the evaluation harness).

Run with: pytest tests/test_retrievers.py -v
"""

import json
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_CHROMA_RESPONSE = {
    "ids": [["film_123_plot", "film_456_plot", "film_789_plot"]],
    "documents": [["Plot of Film A.", "Plot of Film B.", "Plot of Film C."]],
    "metadatas": [
        [
            {"film_id": "123", "title": "Film A", "year": 2020, "doc_type": "plot"},
            {"film_id": "456", "title": "Film B", "year": 2019, "doc_type": "plot"},
            {"film_id": "789", "title": "Film C", "year": 2021, "doc_type": "plot"},
        ]
    ],
    "distances": [[0.1, 0.2, 0.3]],
}


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection returning standard results."""
    col = MagicMock()
    col.count.return_value = 500
    col.query.return_value = MOCK_CHROMA_RESPONSE
    col.get.return_value = {"ids": []}
    return col


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer returning a fixed embedding."""
    import numpy as np
    model = MagicMock()
    model.encode.return_value = np.zeros(384)  # MiniLM dimension
    return model


# ── TextRetriever Tests ───────────────────────────────────────────────────────

class TestTextRetriever:

    @patch("retrieval.text_retriever.chromadb.PersistentClient")
    @patch("retrieval.text_retriever.SentenceTransformer")
    def test_retrieve_returns_list(self, mock_st, mock_client, mock_chroma_collection, mock_sentence_transformer):
        mock_st.return_value = mock_sentence_transformer
        mock_client.return_value.get_collection.return_value = mock_chroma_collection

        from retrieval.text_retriever import TextRetriever
        retriever = TextRetriever(top_k=3)
        results = retriever.retrieve("who directed this film")

        assert isinstance(results, list)
        assert len(results) == 3

    @patch("retrieval.text_retriever.chromadb.PersistentClient")
    @patch("retrieval.text_retriever.SentenceTransformer")
    def test_retrieve_result_has_required_keys(self, mock_st, mock_client, mock_chroma_collection, mock_sentence_transformer):
        mock_st.return_value = mock_sentence_transformer
        mock_client.return_value.get_collection.return_value = mock_chroma_collection

        from retrieval.text_retriever import TextRetriever
        retriever = TextRetriever(top_k=3)
        results = retriever.retrieve("test query")

        required_keys = {"doc_id", "film_id", "title", "modality", "content", "score", "metadata"}
        for result in results:
            assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"

    @patch("retrieval.text_retriever.chromadb.PersistentClient")
    @patch("retrieval.text_retriever.SentenceTransformer")
    def test_retrieve_score_is_between_minus1_and_1(self, mock_st, mock_client, mock_chroma_collection, mock_sentence_transformer):
        mock_st.return_value = mock_sentence_transformer
        mock_client.return_value.get_collection.return_value = mock_chroma_collection

        from retrieval.text_retriever import TextRetriever
        retriever = TextRetriever(top_k=3)
        results = retriever.retrieve("test query")

        for result in results:
            assert -1.0 <= result["score"] <= 1.0, f"Score out of range: {result['score']}"


# ── HybridRetriever Tests ─────────────────────────────────────────────────────

class TestHybridRetriever:

    def test_rrf_fuse_combines_results(self):
        """RRF should merge two ranked lists and boost cross-appearing films."""
        from retrieval.hybrid_retriever import HybridRetriever

        # Create retriever with mocked sub-retrievers
        with patch("retrieval.hybrid_retriever.TextRetriever"), \
             patch("retrieval.hybrid_retriever.CLIPRetriever"), \
             patch("retrieval.hybrid_retriever.CaptionRetriever"):
            retriever = HybridRetriever(top_k=3)

        list_1 = [
            {"film_id": "A", "title": "Film A", "score": 0.9, "modality": "text", "doc_id": "A_plot", "content": "...", "metadata": {}},
            {"film_id": "B", "title": "Film B", "score": 0.8, "modality": "text", "doc_id": "B_plot", "content": "...", "metadata": {}},
        ]
        list_2 = [
            {"film_id": "B", "title": "Film B", "score": 0.85, "modality": "image", "doc_id": "B_poster", "content": "path/B.jpg", "metadata": {}},
            {"film_id": "C", "title": "Film C", "score": 0.7, "modality": "image", "doc_id": "C_poster", "content": "path/C.jpg", "metadata": {}},
        ]

        fused = retriever._rrf_fuse([list_1, list_2], ["text", "clip"])

        # Film B appears in both lists — should rank highest
        assert fused[0]["film_id"] == "B", "Film B should rank highest (appears in both lists)"

    def test_rrf_fuse_includes_rrf_score(self):
        """Fused results must have rrf_score field for debugging."""
        from retrieval.hybrid_retriever import HybridRetriever

        with patch("retrieval.hybrid_retriever.TextRetriever"), \
             patch("retrieval.hybrid_retriever.CLIPRetriever"), \
             patch("retrieval.hybrid_retriever.CaptionRetriever"):
            retriever = HybridRetriever(top_k=5)

        list_1 = [{"film_id": "X", "title": "X", "score": 0.9, "modality": "text", "doc_id": "X_plot", "content": "", "metadata": {}}]
        fused = retriever._rrf_fuse([list_1], ["text"])

        assert "rrf_score" in fused[0]
        assert "source_lists" in fused[0]


# ── Metrics Tests ─────────────────────────────────────────────────────────────

class TestMetrics:

    def test_recall_at_k_hit(self):
        from evaluation.metrics import recall_at_k
        retrieved = ["123", "456", "789"]
        ground_truth = ["456"]
        assert recall_at_k(retrieved, ground_truth, k=5) == 1.0

    def test_recall_at_k_miss(self):
        from evaluation.metrics import recall_at_k
        retrieved = ["123", "456", "789"]
        ground_truth = ["999"]
        assert recall_at_k(retrieved, ground_truth, k=5) == 0.0

    def test_recall_at_k_cutoff_respected(self):
        from evaluation.metrics import recall_at_k
        # Ground truth is at position 3 (index 2), but k=2 should miss it
        retrieved = ["A", "B", "CORRECT", "D"]
        ground_truth = ["CORRECT"]
        assert recall_at_k(retrieved, ground_truth, k=2) == 0.0
        assert recall_at_k(retrieved, ground_truth, k=3) == 1.0

    def test_mean_recall_averages_correctly(self):
        from evaluation.metrics import mean_recall_at_k
        results = [
            {"retrieved_film_ids": ["A"], "ground_truth_film_ids": ["A"]},  # hit
            {"retrieved_film_ids": ["B"], "ground_truth_film_ids": ["C"]},  # miss
        ]
        assert mean_recall_at_k(results, k=5) == 0.5

    def test_latency_timer_measures_elapsed(self):
        import time
        from evaluation.metrics import LatencyTimer
        with LatencyTimer() as t:
            time.sleep(0.05)
        assert t.elapsed_ms >= 50, f"Expected >= 50ms, got {t.elapsed_ms:.1f}ms"


# ── Personal Films Loader Tests ───────────────────────────────────────────────

class TestPersonalFilmsLoader:

    def test_load_personal_films_when_file_missing(self):
        """Test loading returns empty list when personal_films.json doesn't exist."""
        from pipeline.tmdb_fetcher import load_personal_films_ids

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "missing.json"
            film_ids = load_personal_films_ids(fake_path)
            assert film_ids == []

    def test_load_personal_films_extracts_ids(self):
        """Test loading extracts film IDs from personal_films.json."""
        from pipeline.tmdb_fetcher import load_personal_films_ids

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "personal_films.json"
            test_data = [
                {"id": 2048, "title": "Mulholland Drive", "year": 2001, "category": "watchlist"},
                {"id": 550, "title": "Fight Club", "year": 1999, "category": "director"},
            ]
            test_file.write_text(json.dumps(test_data))

            film_ids = load_personal_films_ids(test_file)

            assert len(film_ids) == 2
            assert 2048 in film_ids
            assert 550 in film_ids
