# Knowledge Base Investigation Results

**Date:** 2026-05-11  
**Investigation:** KB metadata quality issues preventing Track 2 implementation

---

## ROOT CAUSE: Wrong TMDB IDs in Test Suite

The KB data is **CORRECT**. The problem is that **test_suite.py has wrong TMDB IDs** for ground truth films.

### Issue #1: Field Name Mismatch

**Problem:** Track 2 code looked for `original_language` field, but KB stores it as `language`.

**KB Schema:**
```python
base_metadata = {
    "language": language,  # ✅ Field exists and has correct data
    ...
}
```

**Track 2 code looked for:**
```python
filter_dict["original_language"] = {"$ne": "en"}  # ❌ Wrong field name
```

**Fix:** Change Track 2 code to use `"language"` instead of `"original_language"`.

---

## Issue #2: Test Suite Ground Truth Has Wrong IDs

| Film | Test Suite ID | Actual Film | Correct ID | Correct Film |
|------|--------------|-------------|-----------|--------------|
| Parasite | 496243 | ✅ Parasite (2019, ko) | 496243 | Same ✅ |
| Capernaum | 553604 | ❌ Honest Thief (2020, en) | **517814** | Capernaum (2018, ar) |
| Zodiac | 314365 | ❌ Spotlight (2015, en) | **1949** | Zodiac (2007, en) |
| Spotlight | 508439 | ❌ Onward (2020, en, Animation) | **314365** | Spotlight (2015, en) |
| Tree of Life | 45269 | ❌ The King's Speech (2010, en) | **8967** | Tree of Life (2011, en) |
| Days of Heaven | 3059 | ❌ Intolerance (1916, en) | **16642** | Days of Heaven (1978, en) |

**Summary:**
- 5 out of 6 multi-hop/visual ground truth films have WRONG IDs
- Only Parasite has correct ID
- This explains why evaluation was failing - we were testing against wrong films!

---

## Actual KB Metadata Quality

**Checking Parasite (496243) - the one correct ID:**
```
Title: Parasite
Year: 2019 ✅
Language: ko ✅ (Korean)
Genres: Comedy, Thriller, Drama ✅
```

**Metadata is CORRECT when IDs are correct!**

---

## How This Happened

The test suite ground truth IDs were likely:
1. Copied from an external source without verification
2. Or TMDB IDs changed/shifted over time
3. Or manual entry errors when creating test cases

---

## Required Fixes

### Fix #1: Update Test Suite Ground Truth IDs

**File:** `src/evaluation/test_suite.py`

```python
# Multi-hop tests - CORRECTED IDs
MULTIHOP_TESTS = [
    TestCase(
        query_id="F3_01",
        query="dark social commentary film, non-English language, released after 2010",
        query_family="multi_hop",
        ground_truth_film_ids=["496243", "517814"],  # Parasite, Capernaum (FIXED)
        ground_truth_titles=["Parasite", "Capernaum"],
        notes="3 constraints: tone + language + year",
    ),
    TestCase(
        query_id="F3_02",
        query="true crime story, documentary-style realism, American setting",
        query_family="multi_hop",
        ground_truth_film_ids=["1949", "314365"],  # Zodiac, Spotlight (FIXED)
        ground_truth_titles=["Zodiac", "Spotlight"],
        notes="Genre + style + setting",
    ),
    TestCase(
        query_id="F3_03",
        query="visually stunning film with minimal dialogue and focus on nature",
        query_family="multi_hop",
        ground_truth_film_ids=["8967", "16642"],  # Tree of Life, Days of Heaven (FIXED)
        ground_truth_titles=["The Tree of Life", "Days of Heaven"],
        notes="Visual quality + narrative style",
    ),
]
```

### Fix #2: Update Track 2 Metadata Filter Field Name

**File:** `src/agent/nodes.py`

```python
# Language constraints
if "non-english" in query_lower or "non english" in query_lower:
    filter_dict["language"] = {"$ne": "en"}  # FIXED: was "original_language"
```

---

## Expected Results After Fixes

With correct ground truth IDs and correct field names:

**Track 2 should now work:**
- F3_01: Filter by `{year: {$gte: 2010}, language: {$ne: "en"}}`
  - Should retrieve: Parasite (2019, ko) ✅ and Capernaum (2018, ar) ✅
- F3_02: No metadata filters (no year/language constraints in query)
  - Relies on semantic search
- F3_03: No metadata filters
  - Relies on semantic search + visual embeddings

**Expected improvement:**
- Multi-hop: 0% → 33-66% (F3_01 should pass with metadata filtering)
- Overall: 38.5% → 45-53%

---

## Validation Steps

1. ✅ **DONE**: Verified KB has correct data when IDs are correct
2. ✅ **DONE**: Identified all wrong IDs in test suite
3. ✅ **DONE**: Verified correct IDs exist in KB
4. ⏳ **TODO**: Fix test_suite.py ground truth IDs
5. ⏳ **TODO**: Fix Track 2 field name (original_language → language)
6. ⏳ **TODO**: Re-run evaluation to measure improvement

---

## Key Learnings

1. **Always verify ground truth data** - Don't assume IDs are correct
2. **Check field names match between code and data** - Schema mismatches are silent failures
3. **KB quality was never the problem** - The TMDB data was fetched correctly
4. **Test suite maintenance is critical** - Wrong ground truth makes evaluation meaningless

---

## Next Steps

1. Fix test_suite.py with correct IDs
2. Fix Track 2 field name
3. Re-enable Track 2 implementation
4. Run clean evaluation
5. Document actual performance improvements
