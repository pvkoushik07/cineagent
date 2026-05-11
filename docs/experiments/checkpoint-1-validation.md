# Checkpoint 1: Track 1 Visual Query Rescue - Validation Results

**Date:** 2026-05-11  
**Checkpoint Goal:** Validate Track 1 results and decide whether to proceed to Track 2

---

## Track 1 Summary: Visual Query Rescue

**Target:** Fix 3/4 failing visual queries (F2_01, F2_02, F2_03, F2_04)  
**Experiments Conducted:** 4 parallel experiments testing different CLIP fusion strategies  
**Result:** ALL EXPERIMENTS FAILED

### Experiment Results

| Experiment | Strategy | Visual Recall | Overall Recall |
|------------|----------|---------------|----------------|
| Baseline | Query-adaptive (0.2, 0.8) text-CLIP | 20% (1/5) | 53.8% (7/13) |
| Exp 1 | Pure CLIP (0.0, 1.0), k=20 | 20% (1/5) | 38.5% (5/13) |
| Exp 2 | Pure CLIP (0.0, 1.0), k=100 | 20% (1/5) | 38.5% (5/13) |
| Exp 3 | Extreme CLIP (0.05, 0.95), k=50 | 20% (1/5) | 38.5% (5/13) |
| Exp 4 | CLIP-only rerank, k=100 | 20% (1/5) | 38.5% (5/13) |

### Key Findings

1. **No visual improvement:** All experiments achieved identical 20% visual recall (only F2_05 passing)
2. **Overall performance decreased:** All experiments reduced overall recall from 53.8% to 38.5%
3. **CLIP hurt other query types:**
   - Factual: 100% → 80% (lost 1 query)
   - Multi-hop: 33.3% → 0% (lost 1 query)
4. **Identical results across strategies:** Pure CLIP, extreme CLIP weights, and large candidate pools all produced the same results

### Root Cause Analysis

The identical results across all strategies suggest:
- **Fundamental CLIP limitation:** CLIP model may not be effective for abstract mood queries ("cold rainy atmosphere")
- **Ground truth issue:** Visual queries may not have corresponding visual content in knowledge base
- **Text superiority:** Text embeddings already capture visual descriptions effectively via plot summaries and captions

### Decision

**REJECT Track 1:** Keep baseline retriever configuration (hybrid or query-adaptive fusion)  
**Rationale:** All CLIP-focused approaches decreased overall performance without improving visual queries

---

## Checkpoint 1 Validation

### Baseline Performance (Pre-Track 1)

- **Overall Recall@5:** 53.8% (7/13 queries correct)
  - Factual: 5/5 (100%)
  - Visual: 1/5 (20%)
  - Multi-hop: 1/3 (33.3%)

### Current Performance (Post-Track 1)

- **Overall Recall@5:** 53.8% (7/13 queries correct) - NO REGRESSION
  - No experiment was merged to improvements branch
  - Baseline retriever unchanged

### Progress Toward Goal

- **Target:** 80%+ overall Recall@5 (12/15 queries, accounting for 2 new conversational queries)
- **Current:** 53.8% (7/13 queries)
- **Gap:** Need +5 queries to reach 12/15 (assuming all new conversational queries pass)
- **Track 1 contribution:** 0 queries fixed

### Decision Gate

**Proceed to Track 2: Multi-Hop Enhancement**

**Rationale:**
1. Track 1 experiments revealed that visual queries cannot be improved via CLIP fusion alone
2. No regression - baseline performance maintained
3. Multi-hop (Track 2) and conversational (Track 3) offer more promising improvement paths
4. Still have 4-7 days to complete remaining tracks
5. Can revisit visual queries if Track 2/3 solve them indirectly (e.g., better memory, multi-hop reasoning)

**Alternative considered:** Abort improvement strategy  
**Rejected because:** Track 2 (multi-hop) and Track 3 (conversational) have not been tested yet and may yield significant gains

---

## Next Steps

1. **Track 2:** Progressive multi-hop improvements via metadata filtering
   - Target: Fix F3_01 and F3_02 (2 queries)
   - Strategy: Filter by year/genre/director before retrieval
   - Expected gain: +2 queries = 69.2% overall

2. **Checkpoint 2:** Validate after Track 2, check for 80% or proceed to Track 3

3. **Track 3:** Add conversational ground truth and fix memory
   - Target: Get 1/2 conversational queries working
   - Expected gain: +1 query = 73.3% overall (still short of 80%)

4. **If 80% not reached:** Consider deeper architectural changes beyond scope of current plan
