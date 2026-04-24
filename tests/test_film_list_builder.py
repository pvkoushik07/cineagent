"""Tests for film list builder."""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from src.pipeline.film_list_builder import search_films_by_title, search_films_by_director, load_personal_films, save_personal_films, add_film


def test_search_films_by_title_returns_results():
    """Test searching TMDB by title."""
    mock_response = {
        "results": [
            {"id": 2048, "title": "Mulholland Drive", "release_date": "2001-10-12"},
            {"id": 1895, "title": "Mulholland Falls", "release_date": "1996-04-26"},
        ]
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.raise_for_status.return_value = None

        results = search_films_by_title("mulholland")

        assert len(results) == 2
        assert results[0]["id"] == 2048
        assert results[0]["title"] == "Mulholland Drive"
        assert results[0]["year"] == 2001


def test_search_films_by_director_returns_filmography():
    """Test searching TMDB by director name."""
    # Mock search for person
    mock_person_search = {
        "results": [
            {"id": 5602, "name": "David Lynch"}
        ]
    }

    # Mock person credits
    mock_credits = {
        "crew": [
            {"id": 2048, "title": "Mulholland Drive", "release_date": "2001-10-12", "job": "Director"},
            {"id": 694, "title": "Blue Velvet", "release_date": "1986-09-19", "job": "Director"},
            {"id": 123, "title": "Dune", "release_date": "1984-12-14", "job": "Producer"},  # Not director
        ]
    }

    with patch("requests.get") as mock_get:
        def side_effect(*args, **kwargs):
            mock_resp = Mock()
            mock_resp.raise_for_status.return_value = None
            if "search/person" in args[0]:
                mock_resp.json.return_value = mock_person_search
            else:
                mock_resp.json.return_value = mock_credits
            return mock_resp

        mock_get.side_effect = side_effect

        results = search_films_by_director("David Lynch")

        assert len(results) == 2  # Only director credits
        assert results[0]["id"] == 2048
        assert results[1]["id"] == 694


def test_load_personal_films_when_file_missing():
    """Test loading returns empty list when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = Path(tmpdir) / "missing.json"
        films = load_personal_films(fake_path)
        assert films == []


def test_load_personal_films_when_file_exists():
    """Test loading returns existing films."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "films.json"
        test_data = [
            {"id": 2048, "title": "Mulholland Drive", "year": 2001, "category": "watchlist"}
        ]
        test_file.write_text(json.dumps(test_data))

        films = load_personal_films(test_file)
        assert len(films) == 1
        assert films[0]["id"] == 2048


def test_save_personal_films():
    """Test saving films to JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "films.json"
        test_data = [
            {"id": 2048, "title": "Mulholland Drive", "year": 2001, "category": "watchlist"}
        ]

        save_personal_films(test_data, test_file)

        # Verify file was written
        assert test_file.exists()
        loaded = json.loads(test_file.read_text())
        assert len(loaded) == 1
        assert loaded[0]["id"] == 2048


def test_add_film_to_list():
    """Test adding a film with duplicate prevention."""
    films = []

    # Add first film
    result = add_film(films, {"id": 2048, "title": "Mulholland Drive", "year": 2001}, "watchlist")
    assert result is True
    assert len(films) == 1
    assert films[0]["category"] == "watchlist"
    assert "added_at" in films[0]

    # Try to add duplicate
    result = add_film(films, {"id": 2048, "title": "Mulholland Drive", "year": 2001}, "director")
    assert result is False
    assert len(films) == 1  # Not added
