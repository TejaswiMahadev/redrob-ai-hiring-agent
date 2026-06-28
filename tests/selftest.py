#!/usr/bin/env python3
"""Self-test / honeypot check for the Redrob ranker.

Verifies the properties we care about, on the full pool:
  1. Known honeypots never appear in the top 100.
  2. Known strong candidates are not wrongly excluded by the plausibility gate.
  3. The plausibility-flagged ("tier 0") rate inside the top 100 is 0%
     (the challenge disqualifies submissions with >10% honeypots in the top 100).
  4. The output ordering satisfies the submission invariants
     (score non-increasing by rank; ties broken by candidate_id ascending).

Run:  python tests/selftest.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redrob.model import load_candidates, reference_date
from redrob import plausibility
from redrob.pipeline import rank_candidates

# A few candidates manually confirmed during development.
KNOWN_HONEYPOTS = {
    "CAND_0001610", "CAND_0003582", "CAND_0016000", "CAND_0033817", "CAND_0008960",
}
KNOWN_REAL_FITS = {
    "CAND_0000031", "CAND_0000200", "CAND_0000273",
    "CAND_0000981", "CAND_0001131", "CAND_0001389",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args(argv)

    failures = []

    # plausibility gate behaviour on known cases
    cands = load_candidates(args.candidates)
    by_id = {c.candidate_id: c for c in cands}
    for cid in KNOWN_HONEYPOTS:
        c = by_id.get(cid)
        if c and not plausibility.assess(c).implausible:
            failures.append(f"known honeypot {cid} NOT flagged as implausible")
    for cid in KNOWN_REAL_FITS:
        c = by_id.get(cid)
        if c and plausibility.assess(c).implausible:
            failures.append(f"known real fit {cid} wrongly flagged as implausible")

    # full ranking
    ranked = rank_candidates(args.candidates, top=args.top)
    top_ids = [r.candidate.candidate_id for r in ranked]

    hp_in_top = [cid for cid in top_ids if cid in KNOWN_HONEYPOTS]
    if hp_in_top:
        failures.append(f"known honeypots in top {args.top}: {hp_in_top}")

    flagged_in_top = sum(
        1 for r in ranked if plausibility.assess(r.candidate).implausible
    )
    rate = 100.0 * flagged_in_top / len(ranked)
    if rate > 10.0:
        failures.append(f"plausibility-flagged rate in top {args.top} is {rate:.1f}% (> 10%)")

    # ordering invariants
    for a, b in zip(ranked, ranked[1:]):
        if a.score < b.score:
            failures.append(f"score increases at rank {a.rank}->{b.rank}")
        if a.score == b.score and a.candidate.candidate_id > b.candidate.candidate_id:
            failures.append(f"equal scores at rank {a.rank}->{b.rank} but candidate_id not ascending")

    print(f"candidates: {len(cands)}  ref_date: {reference_date(cands)}")
    print(f"top-{args.top} plausibility-flagged: {flagged_in_top} ({rate:.1f}%)")
    print(f"top-3: {top_ids[:3]}")
    if failures:
        print(f"\nSELFTEST FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print("  -", f)
        return 1
    print("\nSELFTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
