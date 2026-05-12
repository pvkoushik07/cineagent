# CineAgent Presentation Demo Guide

**Goal**: Show the system working in 2-3 minutes with high impact

---

## 🎯 Demo Strategy: Pre-Record + Live Hybrid

### Option A: Safe & Professional (Recommended)
**Pre-record the demo, play video, narrate live**

**Why this works:**
- No API failures or network issues during presentation
- Guaranteed timing (2-3 minutes exactly)
- Can edit out loading times
- Focus on explaining, not troubleshooting
- Backup plan if projector/laptop issues occur

**How to record:**
```bash
# Use screen recording tool (macOS: Cmd+Shift+5, Windows: Win+G)
# Record terminal window running demo.py --demo
# Or record QuickTime screencast

# Practice narration separately
# Final video: 2-3 minutes, shows 3 queries
```

### Option B: Live Demo (Higher Risk, Higher Impact)
**Run demo.py live with pre-written queries**

**Why risky:**
- API might be slow (Gemini Flash)
- Network issues
- Unexpected errors
- Hard to time exactly

**Mitigation:**
- Test immediately before presentation
- Have backup video ready
- Use `--quick` flag (faster)
- Prepare for 20-30 second response times

---

## 📊 Recommended Demo Script (3 minutes)

### Slide 1: "Let me show you how it works" (30 sec)

**Show terminal running:**
```bash
python demo.py --demo
```

**Narrate while loading:**
> "I'm running the interactive demo with 3 pre-written queries that showcase different capabilities. The system loads three modalities: text embeddings, CLIP image embeddings, and our BM25 sparse retrieval component."

---

### Query 1: Factual Retrieval (45 sec)

**Query shown:**
```
"I want a psychological thriller with a twist ending"
```

**System response (on screen):**
```
🤖 CineAgent: Based on your preferences, I recommend:

1. **Mulholland Drive** (2001)
   A surreal neo-noir psychological thriller...

2. **The Prestige** (2006)
   Two rival magicians in Victorian London...

📊 Metrics:
   • Strategy: text
   • Latency: 8,234ms
   • Taste confidence: 0.70
```

**Narrate:**
> "Query 1 is a straightforward factual retrieval. The system uses text embeddings to match 'psychological thriller' and 'twist ending' against plot summaries. Notice it retrieved the correct films in under 9 seconds."

---

### Query 2: Multi-Hop Thematic Query (45 sec)

**Query shown:**
```
"Dark social commentary, non-English, after 2010"
```

**System response:**
```
🤖 CineAgent: Based on your preferences:

**Parasite** (2019) - Korean
   All unemployed, Ki-taek's family takes peculiar interest...

📊 Metrics:
   • Strategy: hybrid (dense + BM25 + metadata filtering)
   • Retrieved: Parasite ranked #1 (was #360 with dense-only)
```

**Narrate:**
> "Query 2 demonstrates our key technical contribution: hybrid dense-sparse retrieval. This multi-hop query has three constraints. The system uses BM25 keyword matching for 'social commentary' - which ranked Parasite at position 1 versus position 360 with dense embeddings alone - then applies metadata filters for language and year. This is why we implemented BM25."

---

### Query 3: Conversational Memory (30 sec)

**Query shown:**
```
Turn 1: "Something like Parasite but set in America"
```

**System response:**
```
🤖 CineAgent: Given your interest in Parasite:

**Knives Out** (2019) - American ensemble thriller...

📊 Taste Profile Updated:
   • Genres: [thriller, drama]
   • Mood: [dark, social commentary]
   • Avoid: [Parasite]
```

**Narrate:**
> "Query 3 shows conversational memory. The system extracted 'dark social commentary' preference from Query 2, applied it here, and filtered by American setting. The taste profile confidence increased from 0.70 to 0.85 as preferences accumulate."

---

### Wrap-Up (30 sec)

**Show final stats on screen:**
```
Demo Complete!
✅ 3 queries processed
✅ Multiple retrieval strategies demonstrated
✅ Taste profile built across turns
```

**Narrate:**
> "In 3 queries, you've seen: text retrieval, hybrid BM25 fusion for thematic queries, and dynamic memory. The full evaluation runs 13 queries across 4 families. Fixed RAG achieved 61.5% recall, our agent 53.8% - an honest negative result that taught us when complexity helps and when it doesn't."

---

## 🎬 Pre-Recording Instructions

### Setup
1. **Clean terminal**: Clear history, resize to readable font size
2. **Screen size**: 1920x1080 or 1280x720 (standard projector)
3. **Font**: Increase terminal font size (18-20pt minimum)
4. **Colors**: High contrast theme (dark bg, bright text)

### Recording Checklist
```bash
# 1. Start screen recording
# 2. Show terminal prompt
# 3. Type slowly: python demo.py --demo
# 4. Wait for each response
# 5. Stop after 3 queries
# 6. Keep video under 3 minutes
```

### Editing Tips
- Cut out model loading time (speed up 2-4x)
- Add text overlays for key points:
  - "Strategy: hybrid (dense + BM25)"
  - "Parasite: #1 with BM25, #360 without"
  - "Taste confidence: 0.70 → 0.85"
- Keep actual response text visible

---

## 🚨 Backup Plan (If Live Demo Fails)

### Have Ready:
1. **Backup video** on USB drive
2. **Screenshots** of key responses (3 slides)
3. **Pre-written script** to narrate

### If demo crashes:
> "Let me show you the pre-recorded version I prepared. [Play video]. The system is working - we ran the full evaluation this morning with these results..."

[Switch to results slides]

---

## ⏱️ Time Management

| Segment | Duration | Notes |
|---------|----------|-------|
| Intro | 30s | "Let me show you..." |
| Query 1 | 45s | Factual retrieval |
| Query 2 | 45s | BM25 contribution |
| Query 3 | 30s | Memory |
| Wrap | 30s | Results summary |
| **Total** | **3min** | Buffer: 30s for transitions |

**Presentation context:**
- Probably 10-15 minute presentation total
- Demo should be 2-3 minutes (20% of time)
- Rest: motivation, architecture, results, conclusions

---

## 💡 Pro Tips

### Before Presentation
- [ ] Test demo.py 1 hour before
- [ ] Test demo.py 10 minutes before
- [ ] Have video backup ready
- [ ] Check API keys work
- [ ] Verify internet connection
- [ ] Test on presentation laptop/projector

### During Demo
- ✅ Explain WHAT you're showing before typing
- ✅ Let audience read responses (don't rush)
- ✅ Point out specific metrics (latency, strategy)
- ✅ Connect to research question
- ❌ Don't apologize if slow ("it's querying 2600 documents")
- ❌ Don't debug errors live (switch to backup)

### Narration Flow
1. **Before query**: What you're testing
2. **During loading**: What's happening (retrieval, fusion)
3. **After response**: Key metrics, connection to results

---

## 🎤 Sample Narration Script (Memorize This)

**Introduction (30s):**
> "Let me demonstrate the system live. I'll run three queries that showcase different capabilities: factual retrieval, our BM25 hybrid approach, and conversational memory."

**Query 1 - After typing:**
> "This is a standard factual query. The system uses MiniLM text embeddings to match against 2,600 plot summaries. [Response appears] You can see it retrieved Mulholland Drive and The Prestige - both psychological thrillers with twist endings. Response time: 8 seconds."

**Query 2 - After typing:**
> "Now a harder query: three constraints - dark social commentary, non-English, after 2010. This demonstrates our key technical contribution. [Response appears] The system used hybrid retrieval - dense embeddings plus BM25 keyword matching. Parasite ranked first because BM25 matched 'social commentary' exactly. With dense embeddings alone, it ranked 360th. This is why we integrated sparse retrieval."

**Query 3 - After typing:**
> "Final query shows memory. The system remembered 'dark social commentary' from Query 2, applied it here, and filtered for American setting. [Response appears] Notice the taste confidence increased from 0.70 to 0.85 - preferences accumulate across turns."

**Wrap-up:**
> "That's the system in action. The full evaluation runs 13 queries. Fixed RAG achieved 61.5%, our agent 53.8% - an honest negative result, but we learned that hybrid retrieval is essential and that complexity doesn't always help."

---

## 📹 Alternative: Video-Only Demo (Safest)

**Create a polished 2-minute video with:**
1. Screen recording of demo
2. Text overlays highlighting key points
3. Background music (subtle)
4. Clear captions for metrics
5. Your face in corner (optional - builds connection)

**Benefits:**
- No live demo stress
- Perfect timing
- Professional polish
- Can rehearse narration perfectly
- Always works

**Tools:**
- Record: QuickTime (Mac), OBS Studio (free, all platforms)
- Edit: iMovie (Mac), DaVinci Resolve (free, all platforms)
- Add overlays/captions in editor

---

## 🎯 Key Messages to Communicate

During your 3-minute demo, hit these points:

1. **"Multimodal"** - Text + images + captions, three modalities
2. **"BM25 hybrid"** - Our key technical contribution (show Parasite #1 vs #360)
3. **"Dynamic memory"** - Taste profile updates across turns (show 0.70 → 0.85)
4. **"Honest results"** - Fixed RAG beat agent (61.5% vs 53.8%), but we learned why

**What NOT to say:**
- ❌ "Sorry it's slow" 
- ❌ "This usually works better"
- ❌ "Let me try again"
- ❌ Technical jargon without explaining

**What TO say:**
- ✅ "The system is querying 2,600 documents"
- ✅ "Notice how it used hybrid retrieval here"
- ✅ "This demonstrates our BM25 contribution"
- ✅ "You can see the taste confidence increasing"

---

## ✅ Final Checklist

**Day Before:**
- [ ] Record backup demo video
- [ ] Test demo.py on presentation laptop
- [ ] Practice narration 3 times with timer
- [ ] Prepare backup screenshots
- [ ] Verify API keys work

**Morning Of:**
- [ ] Test internet connection
- [ ] Test demo.py once
- [ ] Load backup video on laptop + USB
- [ ] Increase terminal font size
- [ ] Clear terminal history
- [ ] Have script notes ready

**5 Minutes Before:**
- [ ] Test demo.py final time
- [ ] Open demo.py in terminal (ready to run)
- [ ] Open backup video (ready to play)
- [ ] Take deep breath

---

## 🎓 Remember

**Demo Purpose:**
- Show system EXISTS and WORKS
- Demonstrate KEY FEATURES (hybrid retrieval, memory)
- Support claims in report (BM25 improvement)
- Build credibility

**Demo is NOT:**
- Your entire presentation
- A debugging session
- A tutorial on how to use it
- The evaluation (save results for slides)

**If something goes wrong:**
- Stay calm
- Switch to backup video
- Say: "Let me show you the pre-recorded version"
- Continue confidently

---

Good luck! 🚀
