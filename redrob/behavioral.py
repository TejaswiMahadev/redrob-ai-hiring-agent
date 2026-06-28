"""Stage 4 — Behavioral-Availability Modifier.

The JD is explicit: "a perfect-on-paper candidate who hasn't logged in for 6
months and has a 5% recruiter response rate is, for hiring purposes, not
actually available. Down-weight them appropriately."

So this stage is a *multiplier* on the Stage 3 fit score (it modulates fit, it
never manufactures it). It keys mainly on responsiveness + recency + open-to-work,
with notice period, interview reliability and GitHub activity as secondary
signals. Range is roughly [0.5, 1.08]: an unreachable candidate is halved, an
engaged one is held at/just above 1.0.

Reference points from the pool: median recruiter_response_rate 0.44, median days
since last active ~137, median github_activity_score -1 (no GitHub linked, so a
real score is a positive minority signal, and -1 is treated as neutral).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(slots=True)
class Behavioral:
    modifier: float
    availability: float
    notes: list[str] = field(default_factory=list)


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _recency(days: float) -> float:
    # full credit up to ~45 days, decaying to 0 by ~270 days (pool max)
    if days <= 45:
        return 1.0
    return _clip(1.0 - (days - 45) / (270 - 45))


def _notice_mult(days) -> float:
    if days is None:
        return 0.96
    if days <= 30:
        return 1.0
    if days <= 60:
        return 0.98
    if days <= 90:
        return 0.95
    if days <= 120:
        return 0.92
    return 0.88


def assess(signals: dict, ref_date: date, last_active: date | None) -> Behavioral:
    notes: list[str] = []

    resp = signals.get("recruiter_response_rate")
    resp = float(resp) if isinstance(resp, (int, float)) else 0.4

    if last_active is not None:
        days = (ref_date - last_active).days
    else:
        days = 270
    recency = _recency(days)

    otw = 1.0 if signals.get("open_to_work_flag") else 0.0

    icr = signals.get("interview_completion_rate")
    icr = float(icr) if isinstance(icr, (int, float)) and icr >= 0 else 0.7

    availability = _clip(0.40 * resp + 0.35 * recency + 0.15 * otw + 0.10 * icr)

    notice_mult = _notice_mult(signals.get("notice_period_days"))

    gh = signals.get("github_activity_score")
    gh_bonus = (min(float(gh), 100.0) / 100.0) * 0.05 if isinstance(gh, (int, float)) and gh > 0 else 0.0

    modifier = (0.55 + 0.50 * availability) * notice_mult * (1.0 + gh_bonus)

    # human-readable notes for the reasoning stage
    if resp <= 0.15:
        notes.append(f"low recruiter response rate ({resp:.2f})")
    elif resp >= 0.75:
        notes.append(f"highly responsive to recruiters ({resp:.2f})")
    if days >= 150:
        notes.append(f"inactive for ~{days} days")
    elif days <= 45:
        notes.append("recently active")
    if signals.get("open_to_work_flag"):
        notes.append("open to work")
    np_days = signals.get("notice_period_days")
    if isinstance(np_days, (int, float)) and np_days <= 30:
        notes.append(f"short notice period ({int(np_days)}d)")
    elif isinstance(np_days, (int, float)) and np_days >= 120:
        notes.append(f"long notice period ({int(np_days)}d)")

    return Behavioral(
        modifier=round(modifier, 4),
        availability=round(availability, 4),
        notes=notes,
    )
