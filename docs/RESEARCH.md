# Research Design Document
## CineAgent — INFS4205/7205 A3

This document is the source of truth for all research decisions.
**Do not implement anything that does not trace back to a section here.**

---

## Primary Research Question

> When users express film preferences through natural language that evolves across
> a conversation, does a multimodal agent with a dynamic taste profile —
> updated turn-by-turn using both visual poster/scene embeddings and textual
> plot signals — outperform a static RAG pipeline and a plain LLM on
> personalised recommendation accuracy?

## Secondary Research Question (Retrieval Design)

> Across the retrieval design space — text-only, caption-only, CLIP-only, and
> hybrid RRF fusion — which combination produces the highest Recall@5 for
> mood and aesthetic queries where the answer is encoded in visual content
> rather than text?

---

## Why These Questions Are Novel

Most movie recommender systems use either collaborative filtering (user-item
matrices) or text-based content filtering (genre tags, plot keywords).

What is novel here:
1. Using CLIP embeddings on scene stills to answer *mood and aesthetic* queries
   — a query like "cold, rain-soaked, isolating" has no keyword match in a plot
   synopsis but maps directly to visual content in scene photographs.
2. A *dynamic* taste profile that updates per-turn rather than a static user
   profile — testing whether incremental preference refinement improves
   retrieval quality across a conversation.
3. Comparing caption-based retrieval (text descriptions of images) against
   direct CLIP embedding retrieval — testing whether the bottleneck is visual
   encoding or linguistic description.

---

## Rubric Criterion Mapping

| Criterion | What satisfies it in this project |
|-----------|-----------------------------------|
| C1: Problem framing (4 marks) | Two-part RQ above, novel CLIP+memory angle |
| C2: KB & retrieval (4 marks) | 3 modalities, 4 retrieval variants, 2 ChromaDB collections |
| C3: Agent framework (4 marks) | 5-node LangGraph with routing, memory, verification |
| C4: Evaluation (4 marks) | 3 system variants, 2 ablations, 4 query families, 3 metrics |
| C5: Report & reproducibility (4 marks) | Notebooks, requirements.txt, failure analysis section |

---

## System Variants (What Gets Compared)

### Variant A — Plain LLM Baseline
- Input: raw user query
- Process: Gemini Flash with no retrieval, no memory
- Purpose: establishes floor; shows what ungrounded LLM produces
- Expected weakness: hallucinations, no personalisation, ignores visual queries

### Variant B — Fixed RAG Pipeline
- Input: raw user query
- Process: hybrid RRF retrieval (text + CLIP) → Gemini Flash synthesis
- No router, no memory, no taste profile — same retrieval path for all queries
- Purpose: shows value of retrieval over plain LLM; isolates agent contribution
- Expected weakness: no memory across turns, no adaptive routing

### Variant C — Full CineAgent (Final System)
- Input: user query + conversation history + taste profile state
- Process: QueryRouter → RetrievalPlanner → TasteProfileUpdater →
  AnswerSynthesiser → Verifier
- Purpose: full system; expected to outperform on conversational + visual queries

---

## Ablation 1 — Retrieval Design Space

Test all 4 retrieval variants on the **same fixed query set** (Family 2 visual
queries + Family 1 factual queries) to isolate the retrieval contribution.

| Variant | Index used | Expected strength | Expected weakness |
|---------|-----------|-------------------|-------------------|
| Text-only | MiniLM on plot+reviews | Factual queries | Visual/mood queries |
| Caption-only | MiniLM on auto-captions | Partial visual | Captions miss nuance |
| CLIP-only | CLIP on posters+stills | Visual/mood queries | Precise factual |
| Hybrid RRF | All above fused | Both query types | Slightly higher latency |

**Hypothesis:** Hybrid RRF will outperform all single-modality variants on
Family 2 queries. CLIP-only will outperform text-only on Family 2. Text-only
will outperform CLIP-only on Family 1.

**This is falsifiable.** If text-only beats hybrid on visual queries, that is
a valid and interesting finding — report it honestly.

---

## Ablation 2 — Memory Design Space

Test 3 memory variants on **Family 4 conversational queries** (multi-turn).

| Variant | Memory type | Behaviour |
|---------|-------------|-----------|
| No memory | None | Each turn treated independently |
| Static memory | Fixed system prompt with user prefs | Prefs set at Turn 1, never update |
| Dynamic taste updater | LangGraph state updated per turn | Prefs refined each turn |

**Hypothesis:** Dynamic taste updater will outperform static memory on Turn 3+
queries where user has refined or contradicted their initial preferences.

---

## Metrics

### Recall@5 (Primary retrieval metric)
- Definition: For a given query with a known correct answer document, is that
  document in the top-5 retrieved results?
- Implementation: build ground truth map {query_id: correct_doc_id}, run
  retrieval, check containment
- Reported as: percentage across all test queries per variant

### ragas Faithfulness (Answer quality metric)
- Definition: Are the claims in the generated answer supported by the retrieved
  context? (0.0–1.0 scale)
- Implementation: ragas library, configured to use Gemini Flash as judge
- Reported as: mean score per variant per query family

### End-to-end Latency (Efficiency metric)
- Definition: Wall-clock time from query received to response returned (ms)
- Implementation: Python time.perf_counter() wrapping the full agent call
- Reported as: mean ± std per variant

### Tool Call Count (Efficiency metric)
- Definition: Number of LangGraph node executions per query
- Implementation: counter incremented in each node, stored in AgentState
- Reported as: mean per query family per variant

---

## Ground Truth Test Set (Fixed — Do Not Change After Evaluation Starts)

### Family 1 — Factual (5 queries)
```
F1_01: "Who directed Mulholland Drive?" → correct: mulholland_drive_plot
F1_02: "What year was Parasite released?" → correct: parasite_plot
F1_03: "What genre is Oldboy?" → correct: oldboy_plot
F1_04: "Who stars in No Country for Old Men?" → correct: no_country_plot
F1_05: "How long is 2001: A Space Odyssey?" → correct: 2001_plot
```

### Family 2 — Cross-Modal Visual (5 queries)
```
F2_01: "cold, desaturated, rain-soaked visual atmosphere" → correct: blade_runner_2049_still
F2_02: "warm, golden, dusty western landscape" → correct: there_will_be_blood_still
F2_03: "clinical white sterile environments, institutional feel" → correct: one_flew_over_still
F2_04: "neon-lit urban nightscape, purple and green palette" → correct: collateral_still
F2_05: "foggy grey post-industrial wasteland" → correct: children_of_men_still
```

### Family 3 — Multi-Hop (3 queries)
```
F3_01: "dark social commentary, non-English, post-2010" → correct set: {parasite, capernaum, portrait}
F3_02: "based on true crime, documentary feel, American" → correct set: {zodiac, spotlight, minari}
F3_03: "visually stunning, minimal dialogue, nature-focused" → correct set: {tree_of_life, days_of_heaven}
```

### Family 4 — Conversational (2 multi-turn sequences)
```
F4_01:
  T1: "I love slow-burn psychological thrillers" → state update: {genre: thriller, pace: slow}
  T2: "preferably non-English, pre-2010" → state update: +{language: non-english, year_max: 2010}
  T3: "I've already seen Oldboy and Cache" → state update: +{watched: [oldboy, cache]}
  Expected T3 answer: does NOT include oldboy or cache; IS non-English pre-2010 thriller

F4_02:
  T1: "suggest something cerebral and mind-bending"
  T2: "actually I don't like sci-fi, keep it grounded"
  T3: "something with a twist ending"
  Expected T3: not sci-fi, has twist, cerebral — tests preference contradiction handling
```

---

## Failure Analysis Plan (Required for Top Band Criterion 5)

Document these known failure modes during evaluation:

1. **CLIP fails on minimalist/typographic posters** — a plain black poster with
   white text (e.g. Funny Games) encodes little visual mood. CLIP retrieval will
   underperform for films with non-representational poster design.

2. **Caption quality limits caption-only retrieval** — Gemini Flash captions are
   sometimes generic ("a dark room with two people talking"). This caps the
   ceiling of caption-only retrieval and should be visible in ablation results.

3. **Memory fails on contradictory preferences** — if user says "I like action"
   then "I hate violence", the taste updater must handle the contradiction. Log
   cases where it fails to reconcile.

4. **Multi-hop fails with ambiguous constraints** — "something like Parasite but
   funnier" combines mood similarity with sentiment shift. Multi-hop decomposition
   may miss the sentiment constraint.

---

## Build Phases

### Phase 1 — Data Pipeline (Week 1)
- [ ] TMDB API setup + fetch 500 films
- [ ] Download posters + 3 stills per film
- [ ] Auto-caption all images with Gemini Flash
- [ ] Build ChromaDB text_collection + image_collection
- [ ] Notebook 01 complete

### Phase 2 — Retrieval Layer (Week 2)
- [ ] text_retriever.py (MiniLM)
- [ ] clip_retriever.py (CLIP)
- [ ] caption_retriever.py
- [ ] hybrid_retriever.py (RRF)
- [ ] All 4 variants independently testable
- [ ] Notebook 02: ablation 1 initial results

### Phase 3 — LangGraph Agent (Week 3)
- [ ] state.py (AgentState)
- [ ] All 5 nodes in nodes.py
- [ ] graph.py (StateGraph with conditional edges)
- [ ] tools.py (LangChain tool wrappers)
- [ ] Notebook 03: agent demo

### Phase 4 — Evaluation (Week 4)
- [ ] test_suite.py (ground truth)
- [ ] metrics.py (Recall@k, ragas, latency)
- [ ] run_eval.py (all variants)
- [ ] Ablation 1 results table
- [ ] Ablation 2 results table
- [ ] Notebook 04: final charts

### Phase 5 — Report (Week 5)
- [ ] 4-page systems paper
- [ ] Failure analysis section
- [ ] All diagrams
- [ ] README final polish
