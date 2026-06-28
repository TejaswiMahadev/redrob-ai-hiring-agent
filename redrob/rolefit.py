"""Stage 2 — Role-Fit Gate.

The decisive trap defense. Answers: *is this candidate actually an AI/ML
production engineer?* — by combining current-title prior, career-history ML
trajectory, and production-evidence language in their role descriptions.

The keyword stuffer (off-domain title, AI skills listed, but every role
description is about sales/accounting/design) collapses here, because role_fit
leans on title + career text, never on the skill list. Conversely a
"plain-language Tier-5" — a backend/data engineer who actually built a
recommendation system at a product company — is promoted by career evidence
even with an unremarkable title.

Disqualifier flags follow the JD's "things we explicitly do NOT want", each
guarded so honest self-framing by a genuinely-experienced engineer is not
punished like a dabbler's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import rubric
from .model import Candidate, PROFICIENCY_RANK

# career-stint title category -> ML relevance weight
_STINT_WEIGHT = {
    "bullseye": 1.00, "core_ml": 0.95, "strong_ml": 0.82, "research": 0.60,
    "perception": 0.55, "adjacent": 0.50, "swe": 0.22, "off_domain": 0.0,
}


@dataclass(slots=True)
class RoleFit:
    role_fit: float                       # [0,1], the gate value
    title_category: str
    title_score: float
    career_ml_score: float                # duration-weighted ML-ness of career
    evidence_score: float                 # [0,1] production-ML text evidence
    domain_core_hits: int                 # JD-domain retrieval/ranking/recsys phrases
    ml_general_hits: int
    production_hits: int
    eval_hits: int
    vector_infra_hits: int
    skill_grounding: float                # [0,1] relevant skills corroborated by usage
    disqualifiers: list[str] = field(default_factory=list)


def _saturate(x: float, k: float) -> float:
    """Map a non-negative count to [0,1) with diminishing returns."""
    return 1.0 - (1.0 / (1.0 + x / k))


def _career_ml_score(c: Candidate) -> float:
    total = 0.0
    weighted = 0.0
    for s in c.career:
        cat = rubric.classify_title(s.title.lower())
        dur = max(s.duration_months, 1)
        weighted += _STINT_WEIGHT.get(cat, 0.0) * dur
        total += dur
    return weighted / total if total else 0.0


def _skill_grounding(c: Candidate) -> float:
    """Relevant retrieval/ranking/ML skills, weighted by proficiency, duration and
    on-platform assessment — NOT by mere presence. Caps quickly; a minor signal."""
    relevant = (
        "embeddings", "information retrieval", "vector search", "semantic search",
        "faiss", "pinecone", "qdrant", "milvus", "opensearch", "elasticsearch",
        "pgvector", "recommendation systems", "learning to rank", "ranking",
        "nlp", "machine learning", "deep learning", "sentence transformers",
        "transformers", "pytorch", "mlops", "feature engineering", "fine-tuning llms",
    )
    assess = c.signals.get("skill_assessment_scores") or {}
    score = 0.0
    for sk in c.skills:
        name = sk.name.lower()
        if any(r in name for r in relevant):
            prof = PROFICIENCY_RANK.get(sk.proficiency, 1) / 4.0
            dur = min(sk.duration_months, 60) / 60.0
            a = assess.get(sk.name)
            assess_bonus = (a / 100.0) if isinstance(a, (int, float)) and a >= 0 else 0.0
            score += prof * (0.5 + 0.5 * dur) * (1.0 + assess_bonus)
    return _saturate(score, 3.0)


def assess(c: Candidate) -> RoleFit:
    text = c.career_text
    cat = rubric.classify_title(c.current_title.lower())
    title_score = rubric.TITLE_SCORE[cat]
    career_ml = _career_ml_score(c)

    domain_core = rubric.count_hits(text, rubric.DOMAIN_CORE)
    ml_general = rubric.count_hits(text, rubric.ML_GENERAL)
    production = rubric.count_hits(text, rubric.PRODUCTION)
    eval_hits = rubric.count_hits(text, rubric.EVAL_RIGOR)
    vector_infra = rubric.count_hits(text, rubric.VECTOR_INFRA)

    # Production-ML evidence: JD-domain phrases dominate, general ML supports.
    evidence_score = _saturate(2.0 * domain_core + ml_general + 0.5 * vector_infra, 4.0)
    has_real_evidence = domain_core >= 1 or ml_general >= 3

    skill_grounding = _skill_grounding(c)

    # role_fit blends: who they are now (title), what they've done (career), and
    # whether the text actually shows ML production work.
    role_fit = (
        0.42 * title_score
        + 0.30 * career_ml
        + 0.28 * evidence_score
    )

    # ---- disqualifier / down-weight flags (guarded) ----
    dq: list[str] = []

    if rubric.count_hits(text, rubric.DABBLER_PHRASES) >= 1 and not has_real_evidence:
        dq.append("dabbler")

    if (cat == "research" or rubric.count_hits(text, rubric.RESEARCH_PHRASES) >= 1) \
            and production == 0 and domain_core == 0:
        dq.append("pure_research")

    # consulting-only: every company/industry is services, no product-industry stint
    inds = set(c.industries_lower)
    companies_l = " ".join(c.companies).lower()
    all_services = inds and all(
        any(ci in i for ci in rubric.CONSULTING_INDUSTRIES) for i in inds
    )
    any_product = any(any(pi in i for pi in rubric.PRODUCT_INDUSTRIES) for i in inds)
    if all_services and not any_product:
        dq.append("consulting_only")

    perception = rubric.count_hits(text, rubric.PERCEPTION_PHRASES)
    nlp_ir = rubric.count_hits(text, rubric.NLP_IR_PHRASES)
    if (cat == "perception" or perception >= 3) and nlp_ir <= 1 and domain_core == 0:
        dq.append("perception_only")

    # Title-chasing = switching every ~1.5yr AND never staying anywhere. If the
    # candidate ever held a role >=24mo, a short average is just a normal
    # multi-company product career (often the richest, most evidence-bearing
    # ones), not chasing — don't penalize it.
    if c.n_stints >= 3 and c.years_of_experience >= 4:
        avg_stint = c.sum_career_months / c.n_stints
        if avg_stint < 18 and c.max_stint_months < 24:
            dq.append("title_chaser")

    if cat == "off_domain" and career_ml < 0.15 and not has_real_evidence:
        dq.append("stale_offdomain")

    # big-tech-only: every employer is big-tech AND no smaller/scrappy product stint
    sizes = [s.company_size for s in c.career] + [c.current_company_size]
    all_big_tech = c.companies and all(
        any(bt in comp.lower() for bt in rubric.BIG_TECH_FIRMS) for comp in c.companies
    )
    has_small = any(sz in rubric.SMALL_COMPANY_SIZES for sz in sizes)
    if all_big_tech and not has_small:
        dq.append("big_tech_only")

    return RoleFit(
        role_fit=round(role_fit, 4),
        title_category=cat,
        title_score=title_score,
        career_ml_score=round(career_ml, 4),
        evidence_score=round(evidence_score, 4),
        domain_core_hits=domain_core,
        ml_general_hits=ml_general,
        production_hits=production,
        eval_hits=eval_hits,
        vector_infra_hits=vector_infra,
        skill_grounding=round(skill_grounding, 4),
        disqualifiers=dq,
    )
