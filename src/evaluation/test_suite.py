"""
Evaluation Test Suite — fixed ground truth.

WARNING: Do NOT change test queries or ground truth after evaluation has started.
Results from different query sets are not comparable.
See docs/RESEARCH.md for the rationale behind each test case.

The test suite covers all 4 required query families from the assignment spec.
"""

from dataclasses import dataclass, field


@dataclass
class TestCase:
    """A single evaluation test case."""
    query_id: str
    query: str
    query_family: str               # factual | visual | multi_hop | conversational
    ground_truth_film_ids: list[str]  # TMDB IDs of correct answer films
    ground_truth_titles: list[str]    # Human-readable titles
    notes: str = ""                   # Why this test case was designed this way


@dataclass
class ConversationalTestCase:
    """A multi-turn conversational test case."""
    sequence_id: str
    turns: list[str]                  # Ordered list of user messages
    expected_state_after: list[dict]  # Expected taste_profile keys after each turn
    final_answer_must_not_include: list[str] = field(default_factory=list)
    final_answer_must_satisfy: list[str] = field(default_factory=list)


# ── Family 1: Factual Retrieval ───────────────────────────────────────────────
# Direct retrieval of stored knowledge.
# Text-only retrieval should handle these well.
# Evaluation: Recall@5 on text_collection.

FACTUAL_TESTS = [
    TestCase(
        query_id="F1_01",
        query="Who directed Mulholland Drive and when was it released?",
        query_family="factual",
        ground_truth_film_ids=["1018"],      # TMDB ID for Mulholland Drive
        ground_truth_titles=["Mulholland Drive"],
        notes="Classic factual retrieval — director + year in plot doc",
    ),
    TestCase(
        query_id="F1_02",
        query="What is the plot of Parasite by Bong Joon-ho?",
        query_family="factual",
        ground_truth_film_ids=["496243"],    # TMDB ID for Parasite
        ground_truth_titles=["Parasite"],
        notes="Plot retrieval — tests plot_text document retrieval",
    ),
    TestCase(
        query_id="F1_03",
        query="What genre is Oldboy and who stars in it?",
        query_family="factual",
        ground_truth_film_ids=["6972"],      # TMDB ID for Oldboy (2003)
        ground_truth_titles=["Oldboy"],
        notes="Genre + cast factual query",
    ),
    TestCase(
        query_id="F1_04",
        query="Who wrote and directed No Country for Old Men?",
        query_family="factual",
        ground_truth_film_ids=["6977"],      # TMDB ID
        ground_truth_titles=["No Country for Old Men"],
        notes="Writer + director — both in credits metadata",
    ),
    TestCase(
        query_id="F1_05",
        query="What is the runtime of 2001: A Space Odyssey?",
        query_family="factual",
        ground_truth_film_ids=["62"],        # TMDB ID
        ground_truth_titles=["2001: A Space Odyssey"],
        notes="Runtime — in metadata JSON, tests metadata retrieval",
    ),
]

# ── Family 2: Cross-Modal Visual Retrieval ────────────────────────────────────
# Queries where the answer is encoded in images, not text.
# Text-only retrieval MUST fail these for the hypothesis to be supported.
# CLIP retrieval should succeed.

VISUAL_TESTS = [
    TestCase(
        query_id="F2_01",
        query="cold desaturated rain-soaked urban visual atmosphere",
        query_family="visual",
        ground_truth_film_ids=["335984", "261"],  # Blade Runner 2049, Se7en
        ground_truth_titles=["Blade Runner 2049", "Se7en"],
        notes="KEY TEST: answer lives in still images, invisible to text search",
    ),
    TestCase(
        query_id="F2_02",
        query="warm golden dusty arid landscape wide open spaces",
        query_family="visual",
        ground_truth_film_ids=["14066", "353081"],  # There Will Be Blood, Sicario
        ground_truth_titles=["There Will Be Blood", "Sicario"],
        notes="Warm visual palette — CLIP on stills should retrieve these",
    ),
    TestCase(
        query_id="F2_03",
        query="clinical white sterile institutional interior lighting",
        query_family="visual",
        ground_truth_film_ids=["510"],  # One Flew Over the Cuckoo's Nest
        ground_truth_titles=["One Flew Over the Cuckoo's Nest"],
        notes="Clinical aesthetic — distinctive visual environment",
    ),
    TestCase(
        query_id="F2_04",
        query="neon-lit urban nightscape purple green wet streets",
        query_family="visual",
        ground_truth_film_ids=["77338", "604"],  # Drive, Collateral
        ground_truth_titles=["Drive", "Collateral"],
        notes="Neon noir aesthetic — poster and stills both encode this",
    ),
    TestCase(
        query_id="F2_05",
        query="foggy grey post-industrial wasteland bleak overcast",
        query_family="visual",
        ground_truth_film_ids=["9693"],  # Children of Men
        ground_truth_titles=["Children of Men"],
        notes="Dystopian grey palette — visually distinctive, not in text",
    ),
]

# ── Family 3: Multi-Hop Synthesis ─────────────────────────────────────────────
# Requires combining multiple constraints.
# Tests task decomposition in the RetrievalPlanner node.

MULTIHOP_TESTS = [
    TestCase(
        query_id="F3_01",
        query="dark social commentary film, non-English language, released after 2010",
        query_family="multi_hop",
        ground_truth_film_ids=["496243", "553604"],  # Parasite, Capernaum
        ground_truth_titles=["Parasite", "Capernaum"],
        notes="3 constraints: tone + language + year — requires metadata + semantic",
    ),
    TestCase(
        query_id="F3_02",
        query="true crime story, documentary-style realism, American setting",
        query_family="multi_hop",
        ground_truth_film_ids=["508439", "314365"],  # Minari... adjust to Zodiac/Spotlight
        ground_truth_titles=["Zodiac", "Spotlight"],
        notes="Genre + style + setting — multi-hop constraint combination",
    ),
    TestCase(
        query_id="F3_03",
        query="visually stunning film with minimal dialogue and focus on nature",
        query_family="multi_hop",
        ground_truth_film_ids=["45269", "3059"],  # Tree of Life, Days of Heaven
        ground_truth_titles=["The Tree of Life", "Days of Heaven"],
        notes="Visual quality + narrative style — needs CLIP + text",
    ),
]

# ── Family 4: Conversational / Memory-Sensitive ───────────────────────────────
# Multi-turn queries where memory is essential.
# Tests TasteProfileUpdater and Verifier nodes.

CONVERSATIONAL_TESTS = [
    ConversationalTestCase(
        sequence_id="F4_01",
        turns=[
            "I love slow-burn psychological thrillers",
            "preferably non-English language, made before 2010",
            "I've already seen Oldboy and Cache, suggest something else",
        ],
        expected_state_after=[
            {"mood_keywords": ["slow-burn", "psychological"], "preferred_genres": ["thriller"]},
            {"preferred_languages": ["non-English"], "year_range": {"max": 2010}},
            {"watched": ["Oldboy", "Cache"]},
        ],
        final_answer_must_not_include=["Oldboy", "Caché", "Cache"],
        final_answer_must_satisfy=[
            "non-English",
            "pre-2010",
            "thriller or psychological",
        ],
        # This is Ablation 2's key test: no-memory vs dynamic memory
    ),
    ConversationalTestCase(
        sequence_id="F4_02",
        turns=[
            "suggest something cerebral and mind-bending",
            "actually I don't like science fiction, keep it grounded",
            "I want something with an unexpected twist ending",
        ],
        expected_state_after=[
            {"mood_keywords": ["cerebral", "mind-bending"]},
            {"avoid_genres": ["science fiction", "sci-fi"]},
            {"mood_keywords": ["cerebral", "mind-bending", "twist"]},
        ],
        final_answer_must_not_include=[],
        final_answer_must_satisfy=["not sci-fi", "has twist", "cerebral"],
        # Tests preference contradiction handling
    ),
]


def get_all_single_turn_tests() -> list[TestCase]:
    """Return all single-turn test cases across families 1–3."""
    return FACTUAL_TESTS + VISUAL_TESTS + MULTIHOP_TESTS


def get_conversational_tests() -> list[ConversationalTestCase]:
    """Return all multi-turn conversational test sequences."""
    return CONVERSATIONAL_TESTS


def get_tests_by_family(family: str) -> list[TestCase]:
    """
    Return test cases for a specific query family.

    Args:
        family: "factual", "visual", or "multi_hop"

    Returns:
        List of TestCase objects
    """
    all_tests = get_all_single_turn_tests()
    return [t for t in all_tests if t.query_family == family]
