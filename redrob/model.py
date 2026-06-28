"""Stage 0 — Stream + Normalize.

Parse candidates.jsonl into typed Candidate objects and derive the cheap,
reusable features every later stage depends on. Pure stdlib, no network.

Design notes grounded in the real data (see PRD_v2.md):
  * Skill *presence* is near-noise (every skill appears on ~12% of candidates),
    so we keep skills as structured (name, proficiency, duration, endorsements)
    and lean on usage context, not presence.
  * Career *descriptions* are the strongest discriminator between real ML
    engineers and keyword-stuffers, so we build a searchable career_text blob.
  * Honeypots reveal themselves through numeric impossibility (skill/stint
    durations exceeding total experience, expert proficiency with 0 months,
    summary-stated YoE disagreeing with profile YoE), so we precompute those.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Iterator, Optional

# ---------------------------------------------------------------------------
# Raw record sub-structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Skill:
    name: str
    proficiency: str          # beginner | intermediate | advanced | expert
    endorsements: int
    duration_months: int      # months the candidate has used the skill (0 if absent)


@dataclass(slots=True)
class Stint:
    company: str
    title: str
    industry: str
    company_size: str
    duration_months: int
    is_current: bool
    start_date: Optional[date]
    end_date: Optional[date]
    description: str


# Proficiency -> ordinal, used in several places.
PROFICIENCY_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}

# "X years" / "X+ years" / "X.Y yrs" appearing in a free-text summary.
_SUMMARY_YOE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|year)\b", re.IGNORECASE)


def _parse_date(s) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


@dataclass(slots=True)
class Candidate:
    # --- identity / profile ---
    candidate_id: str
    name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

    # --- structured history ---
    career: list[Stint]
    skills: list[Skill]
    education: list[dict]
    certifications: list[dict]

    # --- redrob behavioral signals (only the ones we use, kept raw) ---
    signals: dict

    # --- derived features (Stage 0 output) ---
    titles_lower: tuple[str, ...] = field(default_factory=tuple)   # current + all career titles
    industries_lower: tuple[str, ...] = field(default_factory=tuple)
    companies: tuple[str, ...] = field(default_factory=tuple)
    career_text: str = ""              # lowercased summary + headline + all descriptions
    skill_names_lower: frozenset[str] = field(default_factory=frozenset)

    sum_career_months: int = 0
    max_stint_months: int = 0
    n_stints: int = 0
    max_skill_duration: int = 0
    n_expert_zero_duration: int = 0    # expert/advanced skills claimed with 0 months used
    summary_yoe: Optional[float] = None
    last_active: Optional[date] = None

    # ---------------------------------------------------------------------
    @classmethod
    def from_record(cls, rec: dict) -> "Candidate":
        p = rec["profile"]
        career = [
            Stint(
                company=h.get("company", ""),
                title=h.get("title", ""),
                industry=h.get("industry", ""),
                company_size=h.get("company_size", ""),
                duration_months=int(h.get("duration_months", 0) or 0),
                is_current=bool(h.get("is_current", False)),
                start_date=_parse_date(h.get("start_date")),
                end_date=_parse_date(h.get("end_date")),
                description=h.get("description", "") or "",
            )
            for h in rec.get("career_history", [])
        ]
        skills = [
            Skill(
                name=s.get("name", ""),
                proficiency=s.get("proficiency", "beginner"),
                endorsements=int(s.get("endorsements", 0) or 0),
                duration_months=int(s.get("duration_months", 0) or 0),
            )
            for s in rec.get("skills", [])
        ]
        sig = rec.get("redrob_signals", {}) or {}

        c = cls(
            candidate_id=rec["candidate_id"],
            name=p.get("anonymized_name", ""),
            headline=p.get("headline", "") or "",
            summary=p.get("summary", "") or "",
            location=p.get("location", "") or "",
            country=p.get("country", "") or "",
            years_of_experience=float(p.get("years_of_experience", 0) or 0),
            current_title=p.get("current_title", "") or "",
            current_company=p.get("current_company", "") or "",
            current_company_size=p.get("current_company_size", "") or "",
            current_industry=p.get("current_industry", "") or "",
            career=career,
            skills=skills,
            education=rec.get("education", []) or [],
            certifications=rec.get("certifications", []) or [],
            signals=sig,
        )
        c._derive()
        return c

    # ---------------------------------------------------------------------
    def _derive(self) -> None:
        titles = [self.current_title] + [s.title for s in self.career]
        self.titles_lower = tuple(t.lower() for t in titles if t)
        self.industries_lower = tuple(
            i.lower() for i in ([self.current_industry] + [s.industry for s in self.career]) if i
        )
        self.companies = tuple(dict.fromkeys(
            [self.current_company] + [s.company for s in self.career]
        ))

        text_parts = [self.summary, self.headline] + [s.description for s in self.career]
        self.career_text = " \n ".join(text_parts).lower()

        self.skill_names_lower = frozenset(s.name.lower() for s in self.skills)

        self.n_stints = len(self.career)
        self.sum_career_months = sum(s.duration_months for s in self.career)
        self.max_stint_months = max((s.duration_months for s in self.career), default=0)
        self.max_skill_duration = max((s.duration_months for s in self.skills), default=0)
        self.n_expert_zero_duration = sum(
            1 for s in self.skills
            if PROFICIENCY_RANK.get(s.proficiency, 0) >= 3 and s.duration_months == 0
        )

        m = _SUMMARY_YOE_RE.search(self.summary)
        self.summary_yoe = float(m.group(1)) if m else None

        self.last_active = _parse_date(self.signals.get("last_active_date"))

    # ---------------------------------------------------------------------
    # convenience accessors used by later stages
    def has_signal(self, key, default=None):
        return self.signals.get(key, default)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def iter_candidates(path: str) -> Iterator[Candidate]:
    """Stream candidates from a .jsonl (or .jsonl.gz) file one at a time."""
    opener = _open_maybe_gzip(path)
    with opener as f:
        for line in f:
            if line.strip():
                yield Candidate.from_record(json.loads(line))


def load_candidates(path: str) -> list[Candidate]:
    return list(iter_candidates(path))


def _open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def reference_date(candidates: Iterable[Candidate]) -> date:
    """Deterministic 'now' for recency features: the latest last_active in the pool.

    Avoids depending on the wall-clock run date, so results are reproducible.
    """
    latest = None
    for c in candidates:
        if c.last_active and (latest is None or c.last_active > latest):
            latest = c.last_active
    return latest or date(2026, 1, 1)
