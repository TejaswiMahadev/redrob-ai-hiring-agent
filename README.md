# Redrob Candidate Ranking System

A transparent, CPU-only, network-free ranker for the **Intelligent Candidate
Discovery & Ranking Challenge**. It ranks the top 100 of 100,000 candidates for
the released Senior AI Engineer job description.

> Design rationale and the reconciliation with the challenge rules live in
> [`PRD_v2.md`](PRD_v2.md).

## Reproduce the submission

No dependencies to install — the ranker uses only the Python standard library
(Python 3.10+).

```bash
python rank.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv
```

Runs end-to-end in **~65 seconds** on a CPU laptop using **~1.8 GB RAM** for the
full 100K pool — comfortably inside the 5 min / 16 GB / CPU-only / no-network
constraints. The output CSV matches `submission_spec.md` (header + 100 rows;
`candidate_id,rank,score,reasoning`; score non-increasing by rank; ties broken
by `candidate_id` ascending).

Validate and self-test:

```bash
python India_runs_data_and_ai_challenge/validate_submission.py submission.csv
python tests/selftest.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl
```

## Why this design

The dataset is built to punish keyword/embedding shortcuts. Three observations
shaped the approach (all verified against the real data):

1. **Skills are noise.** Every skill appears on ~12% of candidates, near-uniform
   across a huge vocabulary — a Marketing Manager is as likely to list "Kafka"
   as a real engineer. So nothing here rewards bare skill presence.
2. **Career descriptions are the signal.** Real ML engineers describe ranking,
   retrieval, recommendation and search work; keyword-stuffers list AI skills but
   their role descriptions are about sales, accounting, or design.
3. **No plain-language "Tier-5" gems exist in this data.** Every one of the
   ~98.8K non-ML-titled candidates has *zero* JD-domain evidence in their career
   text, so semantic embeddings would add nothing — they were intentionally
   omitted (this keeps the system fast, dependency-free, and trap-resistant).

The ranking step therefore makes **no LLM/API calls** (which the rules forbid
anyway) and uses **no embeddings or GPUs** — it is a transparent feature + rule
scorer that can be defended line by line.

## Pipeline (see `redrob/`)

| Stage | Module | What it does |
|---|---|---|
| 0 | `model.py` | Parse `candidates.jsonl`; derive features (tenure, career text, skill grounding, summary-YoE, etc.) |
| 1 | `plausibility.py` | Honeypot / impossible-profile gate → forces tier 0 (skill used longer than the whole career, "expert" with 0 months, a stint outlasting its own dates, …) |
| 2 | `rolefit.py` | The decisive trap defense: *is this an AI/ML production engineer?* — title trajectory × career ML-ness × production-evidence text. Disqualifier flags (dabbler, pure-research, consulting-only, perception-only, title-chaser, big-tech-only) |
| 3 | `scoring.py` | Weighted composite over explainable dimensions (role-fit, production-evidence, experience, trajectory, domain, location) with disqualifier down-weights |
| 4 | `behavioral.py` | Availability multiplier from `redrob_signals` (responsiveness, recency, open-to-work, notice, GitHub) — "perfect on paper but unreachable" candidates drop |
| 5 | `reasoning.py` + `pipeline.py` | Grounded per-candidate reasoning (specific, honest, no hallucination, rank-consistent) and the ranked top-100 |

All weights, keyword families, disqualifier rules and thresholds live in one
place: [`redrob/rubric.py`](redrob/rubric.py) — the JD encoded as typed config.

## Trap handling

- **Honeypots:** the plausibility gate flags impossible profiles via numeric
  inconsistency; the self-test asserts **0 honeypots / 0% flagged in the top 100**
  (the challenge disqualifies submissions with >10%).
- **Keyword stuffers:** suppressed at Stage 2 — an off-domain title with no ML
  career evidence collapses regardless of how many AI skills are listed.
- **Stale/unavailable candidates:** down-weighted at Stage 4 per the JD's
  explicit instruction.
