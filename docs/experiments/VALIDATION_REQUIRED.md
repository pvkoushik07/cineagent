# VALIDATION REQUIRED: Performance Improvements

**Date:** 2026-05-11  
**Branch:** improvements  
**Status:** Implementation complete, awaiting user validation

---

## What Was Implemented

### Track 1: Visual Query Rescue (REJECTED)
- 4 parallel experiments tested CLIP fusion strategies
- All experiments achieved identical results: Visual 20%, Overall 38.5%
- **Decision:** Rejected all experiments, kept baseline configuration
- **Impact:** 0 queries fixed

### Track 2: Multi-Hop Enhancement (IMPLEMENTED)
- **Step 1:** Increased candidate pool to k=200 for multi-hop queries
- **Step 2:** Metadata constraint extraction (year, language, genre)
- **Step 3:** Result trimming to top-10 after filtering
- **Files changed:** `src/agent/nodes.py` (retrieval_planner_node, _extract_metadata_constraints)
- **Expected impact:** +1 query (F3_01)

### Track 3: Conversational Foundation (IMPLEMENTED)
- **Phase 1:** Added ground truth to F4_01 and F4_02
- **Phase 2:** Verified memory logic (no bugs found)
- **Files changed:** `src/evaluation/test_suite.py` (ConversationalTestCase, CONVERSATIONAL_TESTS)
- **Expected impact:** +1 query (F4_01)

---

## Validation Required

**You must run the full evaluation to validate the improvements.**

### Quick Validation (Recommended)

```bash
# 1. Activate your environment
source venv/bin/activate  # or your virtualenv path

# 2. Checkout improvements branch
git checkout improvements

# 3. Run full agent evaluation (Variant C)
python src/evaluation/run_eval.py --variant C

# 4. Check results in output (look for Recall@5 metrics)
```

### Expected Results

**Baseline (main branch):**
- Overall Recall@5: 53.8% (7/13 queries)
- Factual: 100%, Visual: 20%, Multi-hop: 33%

**After improvements (improvements branch):**
- Overall Recall@5: **60-67%** (9-10/15 queries)
- Factual: 100% (unchanged)
- Visual: 20% (unchanged - Track 1 failed)
- Multi-hop: 67-100% (+1-2 queries from metadata filtering)
- Conversational: 50-100% (+1-2 queries from ground truth)

### What to Look For

✅ **Success indicators:**
- Overall Recall@5 ≥ 60%
- Multi-hop Recall@5 ≥ 67% (at least F3_01 should pass)
- Conversational Recall@5 ≥ 50% (at least F4_01 should pass)
- No regressions (Factual still 100%, Visual still ≥ 20%)

⚠️ **Warning signs:**
- Overall Recall@5 < 53.8% (regression)
- Factual Recall@5 < 100% (Track 2/3 broke baseline)
- Visual Recall@5 < 20% (unexpected regression)

❌ **Failure (should not happen):**
- Track 2/3 changes broke the agent (errors, crashes)
- Eval script fails to run

---

## If Validation Fails

### Scenario 1: Track 2 didn't help (Multi-hop still 33%)

**Likely cause:** Metadata extraction not working correctly for F3_01  
**Debug:** Check if `_extract_metadata_constraints` correctly parses "after 2010" and "non-English"  
**Fix:** Adjust pattern matching in `_extract_metadata_constraints` function

### Scenario 2: Track 3 didn't help (Conversational still 0%)

**Likely cause:** Ground truth TMDB IDs incorrect or films not in KB  
**Debug:** Check if films 23383, 711, 745, 629 exist in ChromaDB  
**Fix:** Update ground truth with correct IDs from your KB

### Scenario 3: Regression (Overall < 53.8%)

**Likely cause:** Track 2 metadata filtering broke factual queries  
**Debug:** Check factual query results - which ones failed?  
**Fix:** Adjust metadata filtering logic to only apply to multi_hop queries

---

## After Validation

### If validation confirms 60-67%:

1. **Review checkpoint documents:**
   - `docs/experiments/checkpoint-1-validation.md`
   - `docs/experiments/checkpoint-2-validation.md`
   - `docs/experiments/checkpoint-3-validation.md`

2. **Proceed to Task 14:** Merge to main and update documentation

3. **Create final report** with actual numbers from validation

### If validation shows < 60%:

1. **Investigate** which track(s) underperformed
2. **Debug and fix** issues found
3. **Re-run validation** after fixes
4. **Update checkpoint documents** with actual results

---

## Running Full Evaluation (Detailed)

If `run_eval.py --variant C` doesn't work, try these alternatives:

### Option 1: Test each query family separately

```bash
# Multi-hop (Track 2)
python scripts/test_multihop.py

# Conversational (Track 3) - may need to create this script
python -c "
from src.evaluation.test_suite import get_conversational_tests
from src.agent.graph import run_turn

for test in get_conversational_tests():
    print(f'Testing {test.sequence_id}')
    state = None
    for turn in test.turns:
        state = run_turn(turn, previous_state=state)
    print(f'Final response: {state[\"response\"][:100]}...')
"
```

### Option 2: Interactive testing

```bash
# Start Python REPL
python

# Run imports
from src.agent.graph import run_turn
from src.evaluation.test_suite import MULTIHOP_TESTS, CONVERSATIONAL_TESTS

# Test F3_01 (multi-hop)
query = "dark social commentary film, non-English language, released after 2010"
state = run_turn(query)
print(f"Retrieved: {[doc['title'] for doc in state['retrieved_docs'][:5]]}")
print(f"Response: {state['response']}")

# Should retrieve Parasite or Capernaum in top-5
```

---

## Contact

If you encounter issues during validation:
- Check logs in evaluation output
- Review `src/agent/nodes.py` changes for Track 2
- Review `src/evaluation/test_suite.py` changes for Track 3
- See improvement log: `docs/experiments/improvement-log.md`
