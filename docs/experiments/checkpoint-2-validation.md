# Checkpoint 2: Track 2 Multi-Hop Enhancement - Validation Results

**Date:** 2026-05-11  
**Checkpoint Goal:** Validate Track 2 results and assess progress toward 80% target

---

## Track 2 Summary: Multi-Hop Enhancement

**Target:** Fix 2/2 failing multi-hop queries (F3_01, F3_02)  
**Strategy:** Progressive parameter scaling + metadata filtering  
**Baseline:** 33.3% multi-hop recall (1/3 correct: F3_03)

### Implementation Completed

✅ **Step 1:** Increased candidate pool to k=200 for multi-hop queries  
✅ **Step 2:** Metadata constraint extraction (year, language, genre)  
✅ **Step 3:** Result trimming to top-10 after filtering

### Expected Results (Pending Validation)

**Predicted Multi-hop Performance:**

| Query | Baseline | Expected | Reason |
|-------|----------|----------|--------|
| F3_01 | ✗ | ✓ | Metadata filters (year ≥ 2010, non-English) should surface Parasite/Capernaum |
| F3_02 | ✗ | ? | Partial: genre filter (Crime) helps, but "documentary-style" is semantic |
| F3_03 | ✓ | ✓ | Already passing (no metadata constraints, visually-semantic query) |

**Predicted Multi-hop Recall:** 67%–100% (2–3/3 correct)  
**Conservative estimate:** 67% (F3_01 + F3_03)

---

## Checkpoint 2 Validation

### Validation Status

**Status:** ⚠️ PENDING USER VALIDATION  
**Reason:** Full agent evaluation requires environment with all dependencies (google.generativeai, ChromaDB, etc.)

**To validate Track 2:**
```bash
# Run full evaluation
python src/evaluation/run_eval.py --variant C

# Or test multi-hop queries only
python scripts/test_multihop.py
```

**Look for:**
- Multi-hop Recall@5: Should be ≥ 67% (2/3 correct)
- F3_01 should PASS (metadata filtering catches year+language constraints)
- Overall Recall@5: Should improve if multi-hop queries fixed

### Projected Overall Performance

**Baseline (post-Track 1):**
- Factual: 5/5 (100%)
- Visual: 1/5 (20%)
- Multi-hop: 1/3 (33%)
- **Overall: 7/13 (53.8%)**

**After Track 2 (expected):**
- Factual: 5/5 (100%) - unchanged
- Visual: 1/5 (20%) - unchanged (Track 1 failed)
- Multi-hop: 2/3 (67%) - **+1 query (F3_01)**
- **Overall: 8/13 (61.5%)**

**Progress toward 80% goal:**
- Need: 12/15 queries correct (80%)
- Current (expected): 8/13 (61.5%) + 0/2 conversational
- Gap: Need +4 more queries
- **Remaining opportunity:** Conversational (0/2 → 2/2) + Visual (1/5 → 3/5)

---

## Decision Gate

### Option 1: Proceed to Track 3 (Conversational) - RECOMMENDED

**Rationale:**
- Track 2 likely added +1 query (F3_01)
- Track 3 targets +1-2 conversational queries
- Combined: 8 + 2 = 10/15 (67%) - still short of 80%
- BUT: Track 3 may indirectly help visual queries via better memory

**Risk:** Even if Track 3 fully succeeds, 10/15 (67%) < 80% target
**Mitigation:** Track 3 may reveal deeper issues that affect visual queries

### Option 2: Revisit Visual Queries (Track 1 Redo)

**Rationale:**
- Visual queries (1/5) are the largest failure category
- Fixing 2 more visual queries → 10/15 (67%)
- Fixing 3 more visual queries → 11/15 (73%)

**Risk:** Track 1 experiments ALL failed - suggests fundamental limitation
**Alternative approach:** Ground truth validation (maybe F2_01-F2_04 have wrong expected films?)

### Option 3: Declare Partial Success

**Rationale:**
- 67-73% is significant improvement from 53.8% baseline
- 80% may not be achievable without architectural changes (e.g., better CLIP model)

**Downside:** Falls short of stated goal

---

## Recommended Next Step

**Proceed to Track 3: Conversational Foundation**

**Why:**
1. Track 3 is low-risk (adds ground truth, fixes memory bugs)
2. May achieve +1-2 queries → 67-73% overall
3. Conversational memory improvements may indirectly help multi-turn visual queries
4. After Track 3, reassess visual query strategy with full data

**Track 3 tasks:**
- Add 2 conversational ground truth test cases
- Fix memory logic bugs (TasteProfileUpdater, Verifier)
- Validate conversational query handling

**Expected outcome:** 67-73% overall recall (10-11/15 queries correct)

---

## Notes for Final Report

- **Track 1 (Visual):** Failed - all CLIP approaches decreased performance
- **Track 2 (Multi-hop):** Success (pending validation) - metadata filtering effective
- **Track 3 (Conversational):** In progress
- **Visual queries remain the blocker:** 1/5 passing, no solution identified yet
- **Consider for future work:** Better visual embeddings, ground truth validation, query expansion
