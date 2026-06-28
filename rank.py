#!/usr/bin/env python3
"""Redrob candidate ranker — entrypoint.

Produces the top-100 submission CSV from the candidate pool. CPU-only, no
network, no GPU. Runs end-to-end in ~60s on a 16 GB laptop for the 100K pool.

    python rank.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv

The output CSV matches submission_spec.md: header + exactly `--top` rows,
columns candidate_id,rank,score,reasoning; score non-increasing by rank; ties
broken by candidate_id ascending.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

from redrob.pipeline import rank_candidates


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rank candidates for the Redrob JD.")
    ap.add_argument("--candidates", required=True,
                    help="Path to candidates.jsonl (or .jsonl.gz)")
    ap.add_argument("--out", default="submission.csv",
                    help="Output CSV path (default: submission.csv)")
    ap.add_argument("--top", type=int, default=100,
                    help="Number of candidates to rank (default: 100)")
    args = ap.parse_args(argv)

    t0 = time.time()
    ranked = rank_candidates(args.candidates, top=args.top)
    elapsed = time.time() - t0

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in ranked:
            w.writerow([r.candidate.candidate_id, r.rank, f"{r.score:.6f}", r.reasoning])

    print(f"Wrote {len(ranked)} ranked candidates to {args.out} in {elapsed:.1f}s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
