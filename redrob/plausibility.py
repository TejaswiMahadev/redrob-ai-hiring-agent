"""Stage 1 — Plausibility / Honeypot Gate.

The dataset seeds ~80 honeypots with subtly impossible profiles. Ranking any of
them in the top 100 hurts us (>10% in top 100 = disqualification), and ranking
them in the top 10 signals our system isn't actually reading profiles.

We do NOT special-case "honeypots". We apply general profile-sanity checks that
any real candidate passes and impossible profiles fail. A candidate that trips
these is forced to relevance tier 0 (pushed to the bottom, excluded from the
top 100 in practice).

Calibrated against real data (see PRD_v2.md / inspect notes):
  * Real strong candidates CAN have a skill duration modestly above their total
    experience (data noise, e.g. Pinecone 88mo on a 6.0yr career). We only flag
    GROSS impossibility, with generous margins, never mild overflow.
  * Honeypot tells observed: skill used for 92mo on a 3.0yr career; "expert" in a
    skill with 0 months; a single 171-month stint on a 10.3yr career; summary
    claiming "5.2 years" while the profile says 3.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Candidate, PROFICIENCY_RANK

# --- tunable thresholds (months unless noted) ------------------------------
SKILL_OVERFLOW_SOFT = 30      # max skill duration this far above YoE -> suspicious
SKILL_OVERFLOW_HARD = 54      # ...this far above -> near-certain impossibility
STINT_OVERFLOW = 24           # a single stint longer than the whole career +2yr
SUM_OVERFLOW = 30             # career months summing well past total experience
# The dataset's duration_months routinely runs a few months over the literal
# start/end span (a systematic quirk affecting ~44% of candidates), so a small
# gap is noise. Real honeypots miss by 100+ months. Only fire on gross gaps.
CALENDAR_GAP = 24             # stint duration exceeding its own date span by >2yr
SUMMARY_YOE_GAP = 2.0         # summary-stated YoE vs profile YoE disagreement (years)
EXPERT_ZERO_SOFT = 1          # one "expert/advanced, 0 months" skill
EXPERT_ZERO_HARD = 2          # two or more is structurally impossible

# points needed to declare a profile implausible
FLAG_THRESHOLD = 3.0


@dataclass(slots=True)
class PlausibilityResult:
    candidate_id: str
    implausible: bool
    score: float
    reasons: list[str] = field(default_factory=list)


def assess(c: Candidate) -> PlausibilityResult:
    score = 0.0
    reasons: list[str] = []
    yoe_months = c.years_of_experience * 12.0

    # 1. Skill used for far longer than the candidate has worked.
    if yoe_months > 0:
        overflow = c.max_skill_duration - yoe_months
        if overflow >= SKILL_OVERFLOW_HARD:
            score += 3.0
            reasons.append(
                f"skill used {c.max_skill_duration}mo but only {c.years_of_experience:.1f}yr total experience"
            )
        elif overflow >= SKILL_OVERFLOW_SOFT:
            score += 1.5
            reasons.append(
                f"skill duration ({c.max_skill_duration}mo) exceeds experience by {overflow/12:.1f}yr"
            )

    # 2. Expert/advanced proficiency with zero months of use — definitionally impossible.
    if c.n_expert_zero_duration >= EXPERT_ZERO_HARD:
        score += 3.0
        reasons.append(f"{c.n_expert_zero_duration} expert/advanced skills claimed with 0 months used")
    elif c.n_expert_zero_duration >= EXPERT_ZERO_SOFT:
        score += 1.0
        reasons.append("an expert/advanced skill claimed with 0 months used")

    # 3. A single stint longer than the entire stated career.
    if yoe_months > 0 and (c.max_stint_months - yoe_months) >= STINT_OVERFLOW:
        score += 2.5
        reasons.append(
            f"a single role lasted {c.max_stint_months}mo but total experience is {c.years_of_experience:.1f}yr"
        )

    # 4. Career durations summing well past total experience.
    if yoe_months > 0 and (c.sum_career_months - yoe_months) >= SUM_OVERFLOW:
        score += 1.5
        reasons.append(
            f"career stints sum to {c.sum_career_months}mo vs {c.years_of_experience:.1f}yr stated"
        )

    # 5. Summary self-reports a different experience level than the profile.
    if c.summary_yoe is not None and abs(c.summary_yoe - c.years_of_experience) >= SUMMARY_YOE_GAP:
        score += 1.5
        reasons.append(
            f"summary says {c.summary_yoe:.1f}yr but profile says {c.years_of_experience:.1f}yr"
        )

    # 6. A stint claiming more months than elapsed between its own start and end.
    for s in c.career:
        if s.start_date:
            end = s.end_date or c.last_active or s.start_date
            calendar = (end.year - s.start_date.year) * 12 + (end.month - s.start_date.month) + 1
            if s.duration_months - calendar > CALENDAR_GAP:
                score += 3.0
                reasons.append(
                    f"a role claims {s.duration_months}mo but its dates span only ~{calendar}mo"
                )
                break

    return PlausibilityResult(
        candidate_id=c.candidate_id,
        implausible=score >= FLAG_THRESHOLD,
        score=round(score, 2),
        reasons=reasons,
    )
