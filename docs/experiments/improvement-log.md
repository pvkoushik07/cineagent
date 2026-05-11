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

[To be filled during Track 3]
