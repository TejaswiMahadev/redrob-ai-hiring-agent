"""Redrob Ranker — sandbox / demo app.

A small hosted runner that satisfies submission_spec.md §10.5: it accepts a
candidate sample (<=100), runs the SAME ranking pipeline used by rank.py
end-to-end on CPU, and shows the ranked shortlist with per-dimension
explainability — plus a CSV download.

Important: the ranking itself is identical to the offline pipeline and makes no
network calls. The optional "LLM-polished explanations" toggle is purely a
presentation layer on top of the already-computed ranking — it never influences
the scores or order, is off by default, and degrades gracefully to the
rule-based reasoning when no API key is configured.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud or a HuggingFace Space (CPU, free tier).
"""

from __future__ import annotations

import io
import json
import os

import streamlit as st

from redrob.model import Candidate
from redrob import rolefit, scoring, behavioral, plausibility
from redrob.model import reference_date
from redrob.pipeline import rank_from_candidates

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sandbox", "sample_candidates.jsonl")
MAX_CANDIDATES = 100

st.set_page_config(page_title="Redrob Ranker", page_icon="🎯", layout="wide")


# --------------------------------------------------------------------------- #
# parsing: accept .jsonl (one object per line) or .json (array of objects)
# --------------------------------------------------------------------------- #
def parse_candidates(text: str) -> list[Candidate]:
    text = text.strip()
    records: list[dict] = []
    if text.startswith("["):
        records = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return [Candidate.from_record(r) for r in records]


# --------------------------------------------------------------------------- #
# optional LLM polish — presentation only, key-gated, graceful fallback
# --------------------------------------------------------------------------- #
def llm_polish(facts: str) -> str | None:
    """Rewrite a reasoning string from STRUCTURED FACTS only (no raw profile),
    so it cannot introduce claims absent from the ranking. Returns None on any
    failure so the caller falls back to the rule-based reasoning."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        model = os.environ.get("REDROB_LLM_MODEL", "claude-haiku-4-5-20251001")
        msg = client.messages.create(
            model=model,
            max_tokens=160,
            system=(
                "You are a recruiter writing a one-to-two sentence justification for a "
                "candidate's rank. Use ONLY the facts provided; do not invent skills, "
                "employers, or numbers. Be specific and honest; keep concerns if present."
            ),
            messages=[{"role": "user", "content": f"Facts:\n{facts}\n\nWrite the justification."}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("🎯 Redrob Candidate Ranker")
st.caption(
    "Transparent, CPU-only, network-free ranking for the Senior AI Engineer JD. "
    "Same pipeline as `rank.py` — no LLM, embeddings, or GPU in the ranking step."
)

with st.sidebar:
    st.header("Input")
    src = st.radio("Candidate source", ["Bundled sample (68)", "Upload .jsonl / .json"])
    uploaded = None
    if src.startswith("Upload"):
        uploaded = st.file_uploader("Candidate file (≤100)", type=["jsonl", "json"])
    top_n = st.slider("Show top N", 5, MAX_CANDIDATES, 25)
    polish = st.checkbox(
        "LLM-polished explanations", value=False,
        help="Presentation only — never affects ranking. Requires ANTHROPIC_API_KEY; "
             "falls back to rule-based reasoning otherwise.",
    )
    run = st.button("Rank candidates", type="primary")

if run:
    try:
        if uploaded is not None:
            text = uploaded.getvalue().decode("utf-8")
        else:
            with open(SAMPLE_PATH, encoding="utf-8") as f:
                text = f.read()
        candidates = parse_candidates(text)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not parse candidates: {e}")
        st.stop()

    if len(candidates) > MAX_CANDIDATES:
        st.warning(f"{len(candidates)} candidates provided; using the first {MAX_CANDIDATES}.")
        candidates = candidates[:MAX_CANDIDATES]

    ranked = rank_from_candidates(candidates, top=min(top_n, len(candidates)))
    ref = reference_date(candidates)

    if polish and not os.environ.get("ANTHROPIC_API_KEY"):
        st.info("LLM polish requested but no ANTHROPIC_API_KEY is set — showing rule-based reasoning.")

    # summary metrics
    flagged = sum(1 for r in ranked if plausibility.assess(r.candidate).implausible)
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates scored", len(candidates))
    c2.metric("Shortlisted", len(ranked))
    c3.metric("Implausible in shortlist", flagged)

    # build rows
    table_rows = []
    csv_buf = io.StringIO()
    import csv as _csv
    w = _csv.writer(csv_buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])

    for r in ranked:
        c = r.candidate
        rf = rolefit.assess(c)
        sc = scoring.score(c, rf)
        reason = r.reasoning
        if polish:
            facts = (
                f"rank={r.rank}; title={c.current_title}; company={c.current_company} "
                f"({c.current_industry}); years_experience={c.years_of_experience}; "
                f"location={c.location}; role_fit={sc.dims['role_fit']}; "
                f"domain_evidence_hits={rf.domain_core_hits}; "
                f"response_rate={c.signals.get('recruiter_response_rate')}; "
                f"open_to_work={c.signals.get('open_to_work_flag')}; "
                f"notice_days={c.signals.get('notice_period_days')}; "
                f"disqualifiers={rf.disqualifiers}; base_reasoning={r.reasoning}"
            )
            polished = llm_polish(facts)
            if polished:
                reason = polished

        w.writerow([c.candidate_id, r.rank, f"{r.score:.6f}", reason])
        table_rows.append({
            "rank": r.rank,
            "candidate_id": c.candidate_id,
            "title": c.current_title,
            "company": f"{c.current_company} ({c.current_industry})",
            "yrs": c.years_of_experience,
            "location": c.location,
            "score": round(r.score, 4),
            "reasoning": reason,
        })

    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download ranked CSV", csv_buf.getvalue(),
        file_name="submission_sample.csv", mime="text/csv",
    )

    st.subheader("Per-candidate breakdown")
    for r in ranked:
        c = r.candidate
        rf = rolefit.assess(c)
        sc = scoring.score(c, rf)
        bh = behavioral.assess(c.signals, ref, c.last_active)
        with st.expander(f"#{r.rank} — {c.candidate_id} · {c.current_title} @ {c.current_company}"):
            cols = st.columns(2)
            cols[0].write("**Fit dimensions**")
            cols[0].json(sc.dims)
            cols[1].write("**Signals & flags**")
            cols[1].json({
                "title_category": rf.title_category,
                "domain_evidence_hits": rf.domain_core_hits,
                "disqualifiers": rf.disqualifiers or "none",
                "availability_modifier": bh.modifier,
                "availability_notes": bh.notes or "none",
                "implausible_honeypot": plausibility.assess(c).implausible,
            })
            st.write("**Reasoning:**", r.reasoning)
else:
    st.info("Choose an input on the left and click **Rank candidates**. "
            "The bundled sample includes strong fits, keyword-stuffers, and honeypots "
            "so you can see how each is handled.")
