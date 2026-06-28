"""Stage 5 (part 1) — Grounded reasoning generation.

Produces the 1-2 sentence `reasoning` string for each shortlisted candidate.
Stage-4 manual review checks for: specific facts from the profile, connection to
JD requirements, honest acknowledgement of concerns, NO hallucination, variety
across rows, and tone consistent with the rank.

Every claim here is derived only from fields that exist on the candidate, so it
cannot hallucinate skills or employers. Tone tracks the score: clean positives
up top, explicit hedges toward the cutoff.
"""

from __future__ import annotations

from .model import Candidate
from .scoring import Score
from .behavioral import Behavioral

# JD-domain phrase -> human theme. Scanned in career_text (lowercased).
_THEMES = [
    ("ranking", ("learning-to-rank", "learning to rank", "ranking model",
                 "ranking layer", "ranking system", "re-ranking", "reranking",
                 "discovery feed")),
    ("retrieval", ("retrieval", "dense retrieval", "hybrid retrieval",
                   "embedding-based retrieval", "nearest-neighbor", "nearest neighbor")),
    ("recommendation systems", ("recommendation system", "recommender",
                                "recommendation-style", "collaborative filtering", "recsys")),
    ("search relevance", ("semantic search", "vector search", "search relevance", "hybrid search")),
    ("NLP", ("nlp", "natural language", "transformer", "language model")),
]

_RELEVANT_SKILLS = (
    "pinecone", "faiss", "qdrant", "milvus", "opensearch", "elasticsearch",
    "pgvector", "embeddings", "information retrieval", "vector search",
    "semantic search", "sentence transformers", "learning to rank",
    "recommendation systems", "pytorch", "tensorflow",
)


def _themes_present(text: str) -> list[str]:
    out = []
    for label, phrases in _THEMES:
        if any(p in text for p in phrases):
            out.append(label)
    return out


def _best_relevant_skill(c: Candidate):
    """The relevant skill with the most months of use (>=12mo to be worth citing)."""
    best = None
    for sk in c.skills:
        nl = sk.name.lower()
        if any(r in nl for r in _RELEVANT_SKILLS) and sk.duration_months >= 12:
            if best is None or sk.duration_months > best.duration_months:
                best = sk
    return best


def _upcase_first(s: str) -> str:
    """Capitalize only the first character — unlike str.capitalize(), preserves
    the case of proper nouns (city names, etc.) in the rest of the string."""
    return s[:1].upper() + s[1:] if s else s


def _join(items: list[str]) -> str:
    items = items[:3]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _concerns(c: Candidate, sc: Score, bh: Behavioral) -> list[str]:
    out = []
    role = sc.role
    yoe = c.years_of_experience
    if "big_tech_only" in role.disqualifiers:
        out.append("career entirely at big-tech, where the JD favors product/startup builders")
    if "consulting_only" in role.disqualifiers:
        out.append("career entirely at services firms")
    if "perception_only" in role.disqualifiers:
        out.append("background leans toward vision/speech over NLP/IR")
    if "pure_research" in role.disqualifiers:
        out.append("research-leaning with limited production signal")
    if role.domain_core_hits <= 2:
        out.append("lighter on direct ranking/retrieval evidence")
    if yoe < 5.0:
        out.append(f"slightly under the 5-9yr band ({yoe:.1f}yr)")
    elif yoe > 9.5:
        out.append(f"at the senior end of the band ({yoe:.1f}yr)")
    resp = c.signals.get("recruiter_response_rate")
    if isinstance(resp, (int, float)) and resp <= 0.2:
        out.append(f"limited recruiter responsiveness ({resp:.2f})")
    np_days = c.signals.get("notice_period_days")
    if isinstance(np_days, (int, float)) and np_days >= 120:
        out.append(f"long notice period ({int(np_days)}d)")
    if sc.dims["location_fit"] < 0.6:
        out.append("based outside the preferred metros")
    return out


def _positives(c: Candidate, sc: Score, bh: Behavioral) -> str:
    bits = []
    resp = c.signals.get("recruiter_response_rate")
    if isinstance(resp, (int, float)) and resp >= 0.7:
        bits.append(f"highly responsive ({resp:.2f})")
    if "recently active" in bh.notes:
        bits.append("recently active")
    if c.signals.get("open_to_work_flag"):
        bits.append("open to work")
    np_days = c.signals.get("notice_period_days")
    if isinstance(np_days, (int, float)) and np_days <= 30:
        bits.append(f"short notice ({int(np_days)}d)")
    loc = sc.dims["location_fit"]
    if loc >= 1.0:
        bits.append(f"{c.location.split(',')[0]}-based (preferred)")
    elif loc >= 0.85:
        bits.append(f"{c.location.split(',')[0]}-based")
    return _join(bits)


def generate(c: Candidate, sc: Score, bh: Behavioral, rank: int) -> str:
    yoe = c.years_of_experience
    themes = _themes_present(c.career_text)
    skill = _best_relevant_skill(c)

    # Sentence 1: who they are + what their career actually shows.
    lead = f"{c.current_title} with {yoe:.1f} yrs at {c.current_company} ({c.current_industry})"
    if themes:
        evid = f"; career shows {_join(themes)} work"
        if skill is not None:
            evid += f", with {skill.name} ({skill.duration_months}mo) in their skill set"
    else:
        evid = "; applied-ML background with limited direct retrieval/ranking signal"
    s1 = lead + evid + "."

    # Sentence 2: availability + (honest) concern, weighted by rank.
    pos = _positives(c, sc, bh)
    concerns = _concerns(c, sc, bh)
    # show concerns more readily as rank worsens; always show if a real flag exists
    show_concern = concerns and (rank > 15 or any(
        f in sc.role.disqualifiers for f in
        ("big_tech_only", "consulting_only", "perception_only", "pure_research")
    ) or concerns[0].startswith(("limited recruiter", "long notice")))

    parts = []
    if pos:
        parts.append(_upcase_first(pos))
    if show_concern:
        parts.append("though " + concerns[0] if pos else _upcase_first(concerns[0]))
    s2 = ("" if not parts else " " + "; ".join(parts) + ".")

    # Near the cutoff, be candid that this is a filler-tier pick.
    if rank >= 90 and "lighter on direct ranking/retrieval evidence" in concerns:
        s2 += " Adjacent fit, likely near the cutoff."

    return (s1 + s2).strip()
