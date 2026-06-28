# PRD v2: Redrob Candidate Ranking System
### Reconciled with the actual dataset, rules, and compute constraints

**Author:** Yukio (revised after dataset review)
**Status:** Approved direction — feature/rule core + guarded embeddings
**Supersedes:** *AI Hiring Intelligence System* (PRD v1), which was written before the dataset was available.
**Challenge:** India Runs Data & AI Challenge — *Intelligent Candidate Discovery & Ranking* (Redrob)

---

## 0. What changed from v1, and why

PRD v1 was a reasonable pre-dataset guess. The shipped bundle invalidates three of its load-bearing choices. This section is the audit trail; the rest of the doc is the corrected design.

| v1 assumption | Reality in the bundle | Consequence |
|---|---|---|
| Candidate data may be PDFs/resumes → PyMuPDF/pdfplumber | `candidates.jsonl`, 100K records, **fully structured JSON** with a published schema | No parsing layer. Delete §6 parsing stack. |
| Agent 4 = LLM recruiter **at ranking time** (pluggable Claude/OpenAI) | **Network OFF during ranking; no hosted LLM calls; CPU-only; ≤5 min; ≤16 GB** for 100K | LLM at ranking is **forbidden**. LLM allowed only in dev/pre-compute. |
| Section-wise embedding similarity as the scoring **core** | JD states plainly: "the right answer is **not** the candidate with the most AI keywords — that's a trap we built in." | Embedding-centric ranking **amplifies the trap**. Demote embeddings to a guarded secondary signal. |
| `bge-reranker-large` cross-encoder over the pool | CPU-only + 5-min budget for 100K; still semantics-flavored | Cut from the critical path. Reasoning about profiles > re-scoring similarity. |
| Dynamic weights across multiple role archetypes | One fixed JD | Keep weighted multi-dimension scoring; drop multi-archetype switching. |
| No fabricated metrics; graceful no-API-key fallback | Matches the rules' spirit exactly | **Keep.** Now mandatory, not optional. |

**The reframe:** this is not a semantic-search problem. It is a *profile-reasoning* problem the organizers deliberately built to punish keyword/embedding shortcuts. The winning system reads the gap between what a profile *says* and what it *means* — and does it in seconds on CPU.

---

## 1. Problem Statement

Rank the **top 100** of **100,000** candidates for a single, fixed job description (Senior AI Engineer, founding team, Pune/Noida) such that the ranking matches a hidden, tiered ground truth — while avoiding the traps the organizers seeded:

- **Keyword stuffers** — e.g. a *Marketing Manager* whose skills list contains 9 perfect AI terms. High keyword/embedding similarity, zero actual fit.
- **Honeypots (~80)** — subtly impossible profiles (8 yrs tenure at a 3-yr-old company; "expert" in 10 skills with 0 months used). Forced to relevance tier 0. **>10% in your top 100 = disqualification.**
- **Plain-language Tier-5s** — strong fits who *don't* use buzzwords (built a recommendation system at a product company, never wrote "RAG"/"Pinecone"). A keyword scan misses them.
- **Behavioral twins / not-actually-available** — perfect on paper but logged out for 6 months with a 5% recruiter response rate.

A single cosine score cannot separate these. The system must reason over structured career/skill/behavioral evidence and produce an **auditable** ranking with honest, specific per-candidate reasoning.

---

## 2. Goals / Non-Goals

**Goals**
- A **transparent feature + rule scorer** that encodes the JD's explicit rubric (§5) and runs over 100K candidates in **well under 5 minutes on CPU, no network**.
- **Trap resistance** as a first-class objective: title/role-fit gating, plausibility/honeypot checks, and a behavioral-availability modifier — not bolt-ons.
- **Guarded semantic recall** of plain-language fits via precomputed embeddings, structurally prevented from overriding role-fit.
- **Honest, specific, non-templated reasoning** per candidate (Stage-4 review), grounded only in fields that exist in that candidate's profile.
- A **reproducible repo**: single command `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`, passing `validate_submission.py`, plus a small-sample sandbox.

**Non-Goals**
- No LLM calls inside the ranking step (rules forbid it). LLM use is confined to development and offline reasoning, and declared honestly.
- No fabricated evaluation numbers. There is **no ground truth in the bundle**; we will not print a NDCG/MAP figure as if measured. We validate by methodology, ablation, and manual top-K inspection.
- No knowledge-graph layer (same reasoning as v1 §8 — structured fields + rules already capture the "worked at healthcare startup → built medical NLP" inference).
- No multi-archetype weighting — one role.

---

## 3. Users & Use Case

- **Primary user / evaluator:** Redrob engineering, reproducing `rank.py` in a sandboxed CPU container and manually reviewing reasoning + defending-the-work interview.
- **Input:** `candidates.jsonl` (100K structured profiles) + the fixed JD (encoded as a typed rubric, §5).
- **Output:** `submission.csv` — top 100, each row carrying `candidate_id, rank, score, reasoning`, with score non-increasing and ties broken by `candidate_id` ascending. Internally, every candidate also carries a per-dimension breakdown for auditability.

---

## 4. System Architecture

A linear, deterministic pipeline. No orchestration framework. "Stage" = a pure function with a typed input/output contract.

```
                 candidates.jsonl (100K)            job_description (fixed) → typed Rubric (§5)
                        │                                          │
                        ▼                                          │
        ┌──────────────────────────────┐                          │
        │ Stage 0: Stream + Normalize   │  parse JSONL, coerce to typed Candidate;
        └──────────────────────────────┘  derive features (tenure, gaps, company-age, etc.)
                        │                                          │
                        ▼                                          ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ Stage 1: Plausibility / Honeypot Gate                          │
        │   impossible-profile checks → hard tier-0 flag (excluded from top)│
        └──────────────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ Stage 2: Role-Fit Gate                                         │
        │   is this a real AI/ML production engineer? title × career ×   │
        │   production-evidence. Keyword-stuffers fail here.             │
        └──────────────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ Stage 3: Multi-Dimensional Feature Score                       │
        │   skills-in-context · experience-band · production-evidence ·  │
        │   trajectory · domain/IR-NLP · location · (guarded) semantic   │
        │   recall signal. Weighted sum from the Rubric.                 │
        └──────────────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ Stage 4: Behavioral-Availability Modifier                     │
        │   multiplicative: last_active, response_rate, open_to_work,    │
        │   notice_period, interview_completion, github_activity         │
        └──────────────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ Stage 5: Rank + Reason + Emit                                  │
        │   sort (score desc, candidate_id asc); take top 100;          │
        │   generate grounded reasoning from the score breakdown        │
        └──────────────────────────────────────────────────────────────┘
                        │
                        ▼
                 submission.csv  (validator-clean)
```

### 4.0 Stage 0 — Stream + Normalize
Read `candidates.jsonl` line by line (487 MB; full materialization is ~fine within 16 GB but streaming keeps headroom). Coerce each record to a typed `Candidate`. Derive cheap features once: total tenure, per-stint durations, employment gaps, # job switches and their cadence, current-vs-past title trajectory, company-age vs tenure (for honeypots), set of normalized skills with proficiency/endorsements/duration, plain-text blob (summary + all role descriptions) for the semantic stage.

### 4.1 Stage 1 — Plausibility / Honeypot Gate
Hard, explainable consistency checks that force a candidate to **tier 0** (excluded from the top 100 unless the pool is exhausted, which it won't be):
- Tenure at a company exceeding that company's plausible age / the candidate's own total experience.
- Sum of role durations >> `years_of_experience`.
- "expert"/"advanced" proficiency with `duration_months == 0` (or absurdly low) across many skills.
- Endorsements/assessment scores inconsistent with claimed proficiency at scale.
- Dates that don't compose (overlaps that are impossible, end before start, etc.).

These are deliberately *rules*, not a model — they're the thing we can defend in the Stage-5 interview and they directly protect against the >10% honeypot DQ. The org says "you don't need to special-case them"; we treat these checks as general profile-sanity, which *also* catches honeypots.

### 4.2 Stage 2 — Role-Fit Gate
The decisive trap defense. Answer: *is this person actually an AI/ML production engineer?* Combine:
- **Title/headline trajectory** — current + recent titles in the ML/AI/applied-science/relevant-eng family vs. clearly-off families (Marketing, Sales, Accounting, Civil/Mechanical, Graphic Design → the sample_submission.csv decoys).
- **Career evidence** — role descriptions showing built/shipped retrieval, ranking, search, recommendation, embeddings, or production ML at **product** companies.
- **Skill grounding** — AI skills corroborated by proficiency × `duration_months` × endorsements × on-platform assessment scores, not bare presence.

A profile with a perfect skill list but a non-ML title and no corroborating career evidence is heavily suppressed here (not necessarily tier-0, but pushed out of contention). This is exactly the "title is the decisive signal against keyword-stuffer traps" idea the organizers' own methodology example endorses.

### 4.3 Stage 3 — Multi-Dimensional Feature Score
Weighted combination of independent, explainable dimensions (weights are a single tuned vector for this one role, defined in the Rubric, not hardcoded magic numbers scattered in code):

| Dimension | Built from |
|---|---|
| Skills-in-context | AI/IR/NLP skills × proficiency × duration × endorsements × assessment scores |
| Experience band | `years_of_experience` vs JD's 5–9 (soft band; partial credit outside) |
| Production evidence | career descriptions: shipped ranking/search/rec/retrieval systems at scale |
| Trajectory | growth toward applied ML at product companies; penalties for title-chasing cadence |
| Domain / IR-NLP | NLP/IR/search/recsys alignment; penalty for CV/speech/robotics-only |
| Location | Pune/Noida/Hyderabad/Mumbai/Delhi-NCR or `willing_to_relocate` |
| **Semantic recall (guarded)** | cosine(JD-text, candidate plain-text) from **precomputed** embeddings — see §6 |

**Guard on the semantic signal:** it can *promote* a plain-language fit (Tier-5 who never wrote buzzwords) but is **capped** so it cannot lift a candidate who fails the Stage-2 role-fit gate. Embeddings add recall, never override role logic. This is the inverse of PRD v1's emphasis and the single most important correction.

### 4.4 Stage 4 — Behavioral-Availability Modifier
A multiplicative modifier in roughly [0.5, 1.1] applied to the Stage-3 score, from `redrob_signals`: stale `last_active_date`, low `recruiter_response_rate`, `open_to_work_flag` false, long `notice_period_days`, low `interview_completion_rate` pull it down; strong engagement and healthy `github_activity_score` hold it at/above 1.0. "Perfect on paper but unreachable" candidates drop, per the JD's explicit instruction. Multiplicative (not additive) so it modulates fit rather than manufacturing it.

### 4.5 Stage 5 — Rank, Reason, Emit
Sort by `(score desc, candidate_id asc)`; take top 100; assign ranks 1–100; enforce score monotonicity. Generate reasoning **from the structured score breakdown** (template-free phrasing driven by which dimensions fired), so every claim is grounded in that candidate's actual fields — no hallucinated skills, tone matched to rank, concerns surfaced for lower ranks. Write validator-clean CSV.

---

## 5. The Rubric (JD → typed config)

The JD's explicit rubric is encoded as a single typed configuration object — the one place weights, keyword families, disqualifier lists, and thresholds live. (Full extraction stored alongside the repo.) Summary:

- **Ideal:** 6–8 yrs total, 4–5 in applied ML at product companies; shipped ≥1 ranking/search/rec system at scale; opinions on retrieval/eval/LLM-integration; Noida/Pune or relocating; platform-active.
- **Must-haves:** embeddings retrieval, vector DB / hybrid search, strong Python, ranking-eval design.
- **Disqualifiers / strong down-weights:** pure research / no production; recent-only LangChain-on-OpenAI w/o pre-LLM ML; title-chasers; 18mo+ no production code; lifelong consulting (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini); CV/speech/robotics w/o NLP-IR; closed-source 5yr+ no external validation; **title mismatch (keyword stuffer)**.
- **Behavioral:** down-weight stale/low-response/closed candidates; notice <30d best.

---

## 6. Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Language / runtime | Python 3.11, stdlib + numpy | Fast enough; minimal deps eases Stage-3 reproduction |
| Data | `candidates.jsonl` streamed via `json` | No parsing stack needed |
| Core scorer | Pure Python/numpy feature + rule engine | Transparent, CPU-fast (~seconds–low-minutes/100K), interview-defensible |
| Embeddings (guarded signal) | `BAAI/bge-small-en-v1.5` (or e5-small), **precomputed offline**, shipped as artifact keyed to candidate order + a build script | Small model keeps optional in-budget recompute feasible; precompute keeps ranking fast and network-free |
| Vector ops | numpy matmul (JD is one query vector) | No FAISS needed for a single-query top-K over precomputed matrix |
| LLM | **Dev/offline only** (rubric drafting, code review) — declared honestly in metadata | Forbidden at ranking time |
| Validation | provided `validate_submission.py` | Catch format errors pre-submit |
| Sandbox | HuggingFace Space / Streamlit Cloud running on a ≤100-candidate sample | Required by §10.5 |
| Eval (local) | Ablation + manual top-K inspection + honeypot-injection test | No ground truth shipped → no fabricated metrics |

**Reproduction contract:** `rank.py` runs end-to-end within 5 min CPU / 16 GB / no network on the full pool. If the embedding artifact is absent, the system degrades to the pure feature/rule path (still a valid, strong ranking) rather than failing — honest graceful degradation.

---

## 7. Explainability

Each shortlisted candidate carries a per-dimension breakdown (role-fit, skills-in-context, production evidence, experience, trajectory, location, semantic, behavioral modifier) used both for the CSV `reasoning` string and for our own audit. Example reasoning style (grounded, specific, honest, rank-consistent):

> *Rank 7 — "6.4 yrs, ML Engineer at a product company; career history shows a shipped recommendation/search system and embeddings work corroborated by high-duration NLP/IR skills. Noida-based, actively engaged (response rate 0.71). Minor concern: 60-day notice period."*

Lower-rank rows acknowledge gaps explicitly (the Stage-4 reviewer penalizes glowing reasoning on weak candidates).

---

## 8. De-scoped (and why)

- **Knowledge-graph layer** — unchanged from v1: structured fields + rules capture the same inferences without graph-build risk under deadline.
- **In-ranking LLM agent** — forbidden by the rules; would also fail Stage-3 reproduction.
- **Cross-encoder reranker** — cut from the critical path; cost/risk on CPU outweighs benefit given the trap-heavy design.
- **Multi-archetype dynamic weighting** — one fixed role; a single tuned weight vector suffices.

---

## 9. Risks & Honesty Constraints

- **No fabricated metrics** — no ground truth shipped; we report methodology + ablations, never a made-up NDCG.
- **No network/LLM at ranking** — enforced by design; verified by running with networking disabled.
- **Honeypot DQ (>10% in top 100)** — directly mitigated by Stage 1; we will run a self-test injecting known-impossible profiles and confirm they never reach the top 100.
- **Over-suppression risk** — aggressive role-fit gating could bury genuine plain-language Tier-5s; the guarded semantic signal (§4.3) and manual top-K review are the counterbalance.
- **Reproducibility** — minimal deps, pinned `requirements.txt`, single documented command, tested on a clean 16 GB CPU run before submission.

---

## 10. Deliverables

1. **Repo** — `rank.py` + scorer modules, `requirements.txt`, README with the single reproduce command, embedding artifact + its build script, `submission_metadata.yaml`.
2. **`submission.csv`** — top 100, validator-clean.
3. **Sandbox link** — hosted small-sample runner (§10.5).
4. **Methodology summary (≤200 words)** for Stage-4 review + a deck if wanted.

---

## 11. Build Plan (proposed order)

1. Stage 0 normalize + feature derivation; load + profile timing on the full 100K.
2. Stage 1 honeypot/plausibility gate + honeypot-injection self-test.
3. Stage 2 role-fit gate + Stage 3 feature score (rubric-driven); manual top-K inspection.
4. Stage 4 behavioral modifier; ablation on with/without.
5. Stage 5 reasoning generation + CSV emit; run `validate_submission.py`.
6. (Optional) precompute embeddings, wire guarded semantic signal, re-inspect top-K.
7. Sandbox + metadata + README + clean-machine reproduction test.
