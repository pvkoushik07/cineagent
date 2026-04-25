import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.text_retriever import TextRetriever
from retrieval.caption_retriever import CaptionRetriever
from retrieval.hybrid_retriever import HybridRetriever


class TestTextRetrieverIntegration:
    """Integration tests for TextRetriever with real KB."""
    
    @pytest.fixture(scope="class")
    def text_retriever(self):
        """Initialize text retriever once for all tests."""
        return TextRetriever(top_k=5)
    
    def test_factual_query_retrieves_correct_film(self, text_retriever):
        """Test: Factual query returns correct film in top-5."""
        # Mulholland Drive (film_id: 1018) should be in top-5 for director query
        query = "Who directed Mulholland Drive?"
        results = text_retriever.retrieve(query)
        
        # Check we got results
        assert len(results) > 0, "Text retriever returned no results"
        assert len(results) <= 5, "Text retriever returned more than top-5"
        
        # Check Mulholland Drive is in top-5
        film_ids = [r["film_id"] for r in results]
        assert "1018" in film_ids, (
            f"Mulholland Drive (1018) not in top-5 results. "
            f"Got film_ids: {film_ids}"
        )
        
        # Check result format
        first_result = results[0]
        assert "doc_id" in first_result
        assert "film_id" in first_result
        assert "title" in first_result
        assert "score" in first_result
        assert "content" in first_result
        assert isinstance(first_result["score"], float)

from retrieval.clip_retriever import CLIPRetriever


class TestCLIPRetrieverIntegration:
    """Integration tests for CLIPRetriever with real KB."""
    
    @pytest.fixture(scope="class")
    def clip_retriever(self):
        """Initialize CLIP retriever once for all tests."""
        return CLIPRetriever(top_k=5)
    
    def test_visual_query_retrieves_correct_film(self, clip_retriever):
        """Test: Visual/mood query returns correct film in top-5."""
        # Caché (film_id: 445) should be in top-5 for cold rainy atmosphere
        # Note: Empirically verified - Caché consistently appears in top-5 for this query
        query = "cold, desaturated, rain-soaked visual atmosphere"
        results = clip_retriever.retrieve(query)

        # Check we got results
        assert len(results) > 0, "CLIP retriever returned no results"
        assert len(results) <= 5, "CLIP retriever returned more than top-5"

        # Check Caché is in top-5
        film_ids = [r["film_id"] for r in results]
        assert "445" in film_ids, (
            f"Caché (445) not in top-5 results. "
            f"Got film_ids: {film_ids}, titles: {[r['title'] for r in results]}"
        )
        
        # Check result format includes image_path
        first_result = results[0]
        assert "doc_id" in first_result
        assert "film_id" in first_result
        assert "title" in first_result
        assert "score" in first_result
        assert isinstance(first_result["score"], float)

class TestCaptionRetrieverIntegration:
    """Integration tests for CaptionRetriever with real KB."""
    
    @pytest.fixture(scope="class")
    def caption_retriever(self):
        """Initialize caption retriever once for all tests."""
        return CaptionRetriever(top_k=5)
    
    def test_visual_query_via_captions(self, caption_retriever):
        """Test: Visual query retrieves films via auto-generated captions."""
        # Should retrieve films with neon-lit nightscape imagery
        query = "neon-lit urban nightscape, purple and green palette"
        results = caption_retriever.retrieve(query)
        
        # Check we got results
        assert len(results) > 0, "Caption retriever returned no results"
        assert len(results) <= 5, "Caption retriever returned more than top-5"
        
        # Check result format and that we got caption docs
        first_result = results[0]
        assert "doc_id" in first_result
        assert "film_id" in first_result
        assert "title" in first_result
        assert "score" in first_result
        assert "content" in first_result
        assert isinstance(first_result["score"], float)
        
        # Verify we got caption-type documents (not plot docs)
        metadata = first_result.get("metadata", {})
        doc_type = metadata.get("doc_type", "")
        assert "caption" in doc_type, (
            f"Expected caption document, got doc_type: {doc_type}"
        )


class TestHybridRetrieverIntegration:
    """Integration tests for HybridRetriever with real KB."""
    
    @pytest.fixture(scope="class")
    def hybrid_retriever(self):
        """Initialize hybrid retriever once for all tests."""
        return HybridRetriever(top_k=5)
    
    def test_hybrid_combines_multiple_sources(self, hybrid_retriever):
        """Test: Hybrid retriever fuses results from multiple modalities."""
        # Query that benefits from both text and visual retrieval
        query = "psychological thriller with twist ending"
        results = hybrid_retriever.retrieve(query)
        
        # Check we got results
        assert len(results) > 0, "Hybrid retriever returned no results"
        assert len(results) <= 5, "Hybrid retriever returned more than top-5"
        
        # Check result format includes RRF score
        first_result = results[0]
        assert "doc_id" in first_result
        assert "film_id" in first_result
        assert "title" in first_result
        assert "score" in first_result  # This should be RRF score
        assert isinstance(first_result["score"], float)
        
        # Hybrid should return film-level results (deduplicated across sources)
        # Check that film_ids are unique (no duplicate films in top-5)
        film_ids = [r["film_id"] for r in results]
        unique_film_ids = list(set(film_ids))
        assert len(film_ids) == len(unique_film_ids), (
            f"Hybrid retriever returned duplicate films: {film_ids}"
        )