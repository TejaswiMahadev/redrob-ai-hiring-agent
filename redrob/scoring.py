"""Stage 3 — Multi-Dimensional Feature Score.

Combine independent, explainable dimensions into one composite fit score using
the JD's weight profile. Each dimension is in [0,1]; weights sum to 1, so the
base composite is in [0,1]. Disqualifier flags from Stage 2 apply as
multiplicative down-weights (a stuffer or pure-researcher is scaled down, not
crudely zeroed, preserving a sensible relative order in the long tail).

Dimensions:
  role_fit            (Stage 2) — is this an AI/ML production engineer?
  production_evidence — shipped/at-scale/eval-rigor language
  experience_fit      — YoE vs the JD's 5-9 band (soft)
  trajectory          — product-company applied-ML growth
  domain_fit          — retrieval/ranking/recsys/NLP topical match
  location_fit        — Pune/Noida preferred; major Indian metros welcome
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import rubric
from .model import Candidate
from .rolefit import RoleFit


@dataclass(slots=True)
class Score:
    final: float
    base: float
    dims: dict
    penalties: dict           # name -> multiplier applied
    role: RoleFit


def _saturate(x: float, k: float) -> float:
    return 1.0 - (1.0 / (1.0 + x / k))


def experience_fit(yoe: float) -> float:
    lo, hi = rubric.EXP_IDEAL_LOW, rubric.EXP_IDEAL_HIGH
    if lo <= yoe <= hi:
        return 1.0
    if yoe < lo:
        # decay from 1.0 at lo down toward ~0.25 at EXP_HARD_LOW
        span = lo - rubric.EXP_HARD_LOW
        return max(0.2, 0.25 + 0.75 * (yoe - rubric.EXP_HARD_LOW) / span) if span > 0 else 0.2
    # yoe > hi: mild decay toward ~0.4 at EXP_HARD_HIGH
    span = rubric.EXP_HARD_HIGH - hi
    return max(0.35, 1.0 - 0.6 * (yoe - hi) / span) if span > 0 else 0.4


def trajectory_fit(c: Candidate, role: RoleFit) -> float:
    # duration-weighted fraction of career at product (non-services) companies
    total = 0.0
    product = 0.0
    for s in c.career:
        dur = max(s.duration_months, 1)
        total += dur
        ind = s.industry.lower()
        if any(pi in ind for pi in rubric.PRODUCT_INDUSTRIES):
            product += dur
    product_frac = product / total if total else 0.0
    return round(0.55 * product_frac + 0.45 * role.career_ml_score, 4)


def domain_fit(role: RoleFit) -> float:
    # topical match to the JD's retrieval/ranking/recsys/NLP domain
    return round(_saturate(2.0 * role.domain_core_hits + role.vector_infra_hits
                           + 0.5 * role.ml_general_hits, 3.0), 4)


def production_evidence(role: RoleFit) -> float:
    return round(_saturate(role.production_hits + 1.5 * role.eval_hits
                           + role.domain_core_hits, 3.0), 4)


def location_fit(c: Candidate) -> float:
    loc = c.location.lower()
    country = c.country.lower()
    reloc = bool(c.signals.get("willing_to_relocate"))
    in_india = "india" in country
    if any(p in loc for p in rubric.LOCATION_PREFERRED):
        return 1.0
    if any(w in loc for w in rubric.LOCATION_WELCOME):
        return 0.85
    if in_india:
        return 0.65 if reloc else 0.5
    # outside India: JD is case-by-case, no visa sponsorship
    return 0.4 if reloc else 0.15


def score(c: Candidate, role: RoleFit) -> Score:
    dims = {
        "role_fit": role.role_fit,
        "production_evidence": production_evidence(role),
        "experience_fit": round(experience_fit(c.years_of_experience), 4),
        "trajectory": trajectory_fit(c, role),
        "domain_fit": domain_fit(role),
        "location_fit": location_fit(c),
    }
    base = sum(rubric.WEIGHTS[k] * v for k, v in dims.items())

    penalties: dict = {}
    mult = 1.0
    for dq in role.disqualifiers:
        p = rubric.PENALTY.get(dq, 1.0)
        penalties[dq] = p
        mult *= p

    final = base * mult
    return Score(
        final=round(final, 6),
        base=round(base, 6),
        dims={k: round(v, 4) for k, v in dims.items()},
        penalties=penalties,
        role=role,
    )
