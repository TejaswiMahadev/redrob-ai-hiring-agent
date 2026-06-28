"""Stage 5 (part 2) — orchestration: run all stages and produce the ranked top-N.

This is the single place the five stages are composed. It is pure CPU, no
network, and holds the whole pool in memory (~1.8 GB for 100K, well under the
16 GB limit). Ranking the full pool takes ~60s on a laptop CPU.

Tie-breaking: the challenge validator requires score to be non-increasing by
rank and, where scores are equal, candidate_id ascending. We round the score to
6 dp and sort by (-rounded_score, candidate_id), which makes both invariants
true by construction even when two distinct raw scores round to the same value.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Candidate, load_candidates, reference_date
from . import plausibility, rolefit, scoring, behavioral, reasoning

SCORE_DP = 6
# implausible (honeypot/impossible) profiles are forced below every real
# candidate so they cannot enter the shortlist.
TIER0_SCORE = -1.0


@dataclass(slots=True)
class Ranked:
    rank: int
    candidate: Candidate
    score: float
    reasoning: str


def score_all(candidates: list[Candidate], ref_date) -> list[tuple[float, Candidate, object, object]]:
    scored = []
    for c in candidates:
        pl = plausibility.assess(c)
        rf = rolefit.assess(c)
        sc = scoring.score(c, rf)
        bh = behavioral.assess(c.signals, ref_date, c.last_active)
        final = sc.final * bh.modifier
        if pl.implausible:
            final = TIER0_SCORE
        scored.append((round(final, SCORE_DP), c, sc, bh))
    return scored


def rank_from_candidates(candidates: list[Candidate], top: int = 100) -> list[Ranked]:
    """Rank an already-loaded list of candidates. Used by rank_candidates() and
    by the sandbox demo app (which works on an uploaded sample)."""
    ref = reference_date(candidates)
    scored = score_all(candidates, ref)

    # primary: score desc; tie-break: candidate_id asc (validator requirement)
    scored.sort(key=lambda r: (-r[0], r[1].candidate_id))

    out: list[Ranked] = []
    for i, (final, c, sc, bh) in enumerate(scored[:top], start=1):
        out.append(Ranked(
            rank=i,
            candidate=c,
            score=final,
            reasoning=reasoning.generate(c, sc, bh, i),
        ))
    return out


def rank_candidates(path: str, top: int = 100) -> list[Ranked]:
    return rank_from_candidates(load_candidates(path), top=top)
