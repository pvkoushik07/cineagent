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
- **Experiment:** 
- **Reason:** 
- **Visual Recall:** 
- **Overall Recall:** 

---

## Track 2: Multi-Hop Enhancement

**Target:** Fix 2/2 failing multi-hop queries (F3_01, F3_02)

[To be filled during Track 2]

---

## Track 3: Conversational Foundation

**Target:** Get 1/2 conversational queries working after adding ground truth

[To be filled during Track 3]
