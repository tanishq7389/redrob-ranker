#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon: Intelligent Candidate Ranking

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Produces a top-100 candidate ranking for the Senior AI Engineer JD.
Runs entirely on CPU, no network, within 5 minutes for 100K candidates.
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

# Allow running from repo root or src/
sys.path.insert(0, str(Path(__file__).parent))

from src.features import score_candidate
from src.reasoning import build_reasoning


def load_candidates(path: str):
    """Load candidates from .jsonl or .jsonl.gz."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    opener = gzip.open if path.endswith(".gz") else open
    candidates = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def rank_candidates(candidates: list[dict], top_n: int = 100):
    """Score all candidates and return top_n sorted by score descending."""
    scored = []
    honeypots = 0

    for i, c in enumerate(candidates):
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i:,} / {len(candidates):,}...", flush=True)
        score, breakdown = score_candidate(c)
        if breakdown.get("honeypot"):
            honeypots += 1
        scored.append((score, breakdown, c))

    print(f"  Honeypots detected and suppressed: {honeypots}")
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]


def write_submission(ranked: list, out_path: str):
    """Write the submission CSV."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_idx, (score, breakdown, candidate) in enumerate(ranked, start=1):
            cid = candidate["candidate_id"]
            reasoning = build_reasoning(candidate, score, breakdown, rank_idx)
            # Escape any internal quotes in reasoning
            writer.writerow([cid, rank_idx, f"{score:.4f}", reasoning])

    print(f"  Written {len(ranked)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Redrob candidate ranker")
    parser.add_argument(
        "--candidates",
        default="./candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--out",
        default="./submission.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of candidates to include (default 100)",
    )
    args = parser.parse_args()

    t0 = time.time()
    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"  Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    print("Scoring candidates...")
    t1 = time.time()
    ranked = rank_candidates(candidates, top_n=args.top)
    print(f"  Scored in {time.time()-t1:.1f}s")

    print(f"Writing submission to {args.out}...")
    write_submission(ranked, args.out)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s total. Top score: {ranked[0][0]:.4f}")


if __name__ == "__main__":
    main()
