# Redrob Intelligent Candidate Ranking — Submission

**Challenge:** India Runs Data & AI Challenge — Intelligent Candidate Discovery & Ranking  
**Team:** Solo-Leveler
 

---

## What this does

Ranks 100,000 candidates from the Redrob platform against the Senior AI Engineer JD in under 60 seconds on a CPU-only machine with no network access — producing a top-100 shortlist with honest, per-candidate reasoning.

---

## Architecture

### Core insight

The JD says explicitly: *"The right answer involves reasoning about the gap between what the JD says and what the JD means."*  

I build a multi-signal scoring system across 8 dimensions with hard disqualifiers:

| Dimension | Weight | What it captures |
|-----------|--------|-----------------|
| Core skills | 28% | Production evidence of embeddings, vector DBs, hybrid search, NDCG/eval |
| Production evidence | 22% | Career descriptions mentioning deployed systems, real users, A/B tests |
| Career trajectory | 15% | Product-company ML career vs consulting-only (explicit JD disqualifier) |
| Title fit | 10% | Current role is actually ML/engineering (not HR Manager, Marketing, etc.) |
| Experience range | 7% | 5–9yr sweet spot, per JD |
| Behavioral signals | 7% | Recency, response rate, notice period, GitHub activity |
| Location | 6% | Pune/Noida/NCR/Hyderabad/Mumbai/Bangalore preferred |
| Domain fit | 5% | NLP/IR vs CV/Speech/Robotics (JD says CV/Speech without NLP = disqualifier) |

### Disqualifiers (score multiplied down hard)
- **Consulting-only career** (TCS, Infosys, Wipro, etc.) → 0.3× multiplier
- **Non-technical current title** (HR Manager, Content Writer, etc.) → 0.2× multiplier  
- **Wrong domain** (CV/Speech without NLP) → 0.4× multiplier

### Honeypot detection
Flags candidates with impossible timelines (duration > 50 years, implausible YoE vs career months, 15+ "expert" skills) and suppresses them to score 0.

### No LLM calls during ranking
The ranking step is pure Python with zero external API calls. Embeddings are not used — instead, we do careful keyword matching against a curated ontology derived from deep JD reading. This runs in ~30 seconds for 100K candidates.

---

## Running it

### Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+ required. No GPU. No network needed during ranking.

### Reproduce the submission

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_redrob_ai.csv
```

Expected runtime: ~50 seconds on a 16GB CPU-only machine.

### Validate

```bash
python validate_submission.py team_redrob_ai.csv
```

---

## Repository structure

```
redrob_ranker/
├── rank.py                    # Main entry point
├── src/
│   ├── features.py            # Scoring logic (8 dimensions + honeypot detection)
│   └── reasoning.py           # Per-candidate reasoning generation
├── requirements.txt
├── submission_metadata.yaml
├── team_redrob_ai.csv         # Output submission
└── README.md
```

---

## Design decisions

**Why no embeddings?** The compute constraint (5 min, CPU only, no GPU) and the dataset size (100K candidates) make dense retrieval impractical without pre-computation. More importantly, the JD warns that keyword-embedding approaches are the *trap* — a candidate listing "RAG" and "Pinecone" as skills but whose career history is all CV work will score high on embeddings. Our system instead checks for *production evidence* in career descriptions, which requires reading the profile narrative, not just matching skill tokens.

**Why hard disqualifiers?** The JD has explicit disqualifiers: consulting-only careers, non-technical titles, wrong domain. Implementing these as multipliers (not just soft weights) ensures they're actually enforced rather than washed out by strong skill signals.

**Why behavioral signals matter here?** A perfect-on-paper candidate last active 200 days ago with a 5% recruiter response rate is not actually hirable. We model this explicitly via the behavioral component.

---

