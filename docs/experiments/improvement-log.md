# CineAgent Performance Improvement Experiment Log

**Goal:** Improve from 53.8% to 80%+ overall Recall@5  
**Date Started:** 2026-05-11  
**Baseline:** 53.8% (7/13 correct: 5 factual + 1 visual + 1 multi-hop)

---

## Track 1: Visual Query Rescue

**Target:** Fix 3/4 failing visual queries (F2_01, F2_02, F2_03, F2_04)  
**Hypothesis:** Text interference or insufficient candidates

### Experiment 1: Pure CLIP (exp/visual-pure-clip)
- **Date:** 
- **Change:** visual weights (0.0, 1.0) - eliminate text completely
- **Results:** 
- **Visual Recall:** 
- **Overall Recall:**
- **Decision:** 

### Experiment 2: More Candidates (exp/visual-more-candidates)
- **Date:** 
- **Change:** candidate_k=100 for visual queries
- **Results:** 
- **Visual Recall:** 
- **Overall Recall:**
- **Decision:** 

### Experiment 3: Extreme CLIP (exp/visual-extreme-clip)
- **Date:** 
- **Change:** visual weights (0.05, 0.95), candidate_k=50
- **Results:** 
- **Visual Recall:** 
- **Overall Recall:**
- **Decision:** 

### Experiment 4: CLIP-Only Reranking (exp/visual-clip-only-rerank)
- **Date:** 
- **Change:** Text for candidates, pure CLIP for ranking
- **Results:** 
- **Visual Recall:** 
- **Overall Recall:**
- **Decision:** 

### Winner:
- **Experiment:** NONE - Track 1 REJECTED
- **Reason:** All 4 experiments achieved identical results (Visual 20%, Overall 38.5%) and DECREASED overall performance from baseline 53.8% to 38.5%. Pure CLIP approaches hurt factual (100%→80%) and multi-hop (33%→0%) queries without improving visual queries. Baseline configuration (query-adaptive fusion) is superior.
- **Visual Recall:** 20% (1/5) - no improvement from baseline
- **Overall Recall:** 38.5% (5/13) - DECREASED from baseline 53.8%
- **Decision:** Keep baseline two-stage retriever configuration, proceed to Track 2 (multi-hop) with original weights 

---

## Track 2: Multi-Hop Enhancement

**Target:** Fix 2/2 failing multi-hop queries (F3_01, F3_02)  
**Strategy:** Progressive parameter scaling + metadata filtering  
**Baseline:** 33.3% (1/3 multi-hop correct: F3_03)

### Implementation (2026-05-11)

**Step 1: Increased Candidate Pool**
- Multi-hop queries now retrieve k=200 candidates (up from k=5)
- Rationale: Multiple constraints require larger pool to find satisfying films
- Change: `retrieval_planner_node` temporarily increases `text_retriever.top_k=200`

**Step 2: Metadata Constraint Extraction**
- Added `_extract_metadata_constraints()` function to parse query for filters
- Detects year constraints: "after 2010" → `{year: {$gte: 2010}}`
- Detects language constraints: "non-English" → `{original_language: {$ne: "en"}}`
- Detects genre constraints: "thriller", "documentary", etc. → `{genres: {$in: [genres]}}`
- Applies ChromaDB `where` filter before retrieval

**Step 3: Result Trimming**
- After metadata filtering, trim 200 results to top-10 for synthesis
- Balance between candidate diversity and context efficiency
- Filtered results more likely to satisfy all constraints

### Expected Impact

**F3_01**: "dark social commentary film, non-English language, released after 2010"
- Should extract: `{year: {$gte: 2010}, original_language: {$ne: "en"}}`
- Ground truth: Parasite (2019, Korean), Capernaum (2018, Arabic)
- Prediction: PASS (both films match filters)

**F3_02**: "true crime story, documentary-style realism, American setting"
- Should extract: `{genres: {$in: ["Crime"]}}`
- Ground truth: Zodiac, Spotlight
- Prediction: UNCERTAIN (needs "crime" in query, might miss "true crime")

**F3_03**: "visually stunning film with minimal dialogue and focus on nature"
- No obvious metadata constraints to extract
- May not benefit from Track 2 changes
- Prediction: BASELINE (already passing)

### Validation Status

**Status:** IMPLEMENTED, AWAITING VALIDATION  
**Validation Required:** Run full agent evaluation on MULTIHOP_TESTS  
**Expected Multi-hop Recall:** 66.7% (2/3) if F3_01 and F3_03 pass

---

## Track 3: Conversational Foundation

**Target:** Get 1/2 conversational queries working after adding ground truth  
**Baseline:** 0/2 (no ground truth existed previously)

### Phase 1: Ground Truth Addition (2026-05-11)

**Updated ConversationalTestCase class:**
- Added `ground_truth_film_ids: list[str]` field
- Added `ground_truth_titles: list[str]` field
- Enables Recall@5 evaluation for conversational sequences

**F4_01 Ground Truth:**
- Query sequence: Slow-burn thriller → non-English, pre-2010 → NOT Oldboy/Cache
- Ground truth: The Secret in Their Eyes (23383), Memories of Murder (711)
- Both match ALL constraints: psychological thriller, non-English, pre-2010, not in exclusion list

**F4_02 Ground Truth:**
- Query sequence: Cerebral/mind-bending → NOT sci-fi, grounded → twist ending
- Ground truth: The Sixth Sense (745), The Usual Suspects (629)
- Both match ALL constraints: cerebral, realistic (not sci-fi), famous twist endings

### Phase 2: Memory Logic Fixes (Conditional)

**Status:** ✅ NO FIXES NEEDED  
**Reason:** Memory logic bugs already fixed in commit 2d97597 (2026-04-26)

**Fixes already in baseline:**
1. **TasteProfileUpdater (lines 511-515):** Only adds explicitly watched films
   - ✓ Handles "I've seen X", "already watched Y", "don't want films I've seen like Z"
   - ✓ Excludes mentions ("Who directed X?" → NOT added to watched)

2. **Verifier verify_rules (lines 687-696):** Checks for watched films
   - ✓ Checks both film_id and title against watched list
   - ✓ Returns False with reason if watched film recommended

3. **Verifier verify_rules (lines 698-709):** Checks for avoided genres
   - ✓ Compares recommended film genres against avoid_genres list
   - ✓ Returns False with reason if avoided genre recommended

4. **Verifier verify_rules (line 679):** Checks for empty responses
   - ✓ Catches empty/whitespace-only responses

**Investigation Results:**
- Code review shows all memory logic correctly implemented
- TasteProfile merge function (lines 410-470) correctly deduplicates and combines preferences
- Verifier has both rule-based (fast) and LLM-based (thorough) contradiction checks

### Validation Status

**Status:** AWAITING VALIDATION  
**Expected:** 1-2/2 conversational queries correct (50-100%)  
**Impact on overall:** +1-2 queries → 62-69% overall recall (9-10/15 total)
