"""The JD, encoded as typed configuration.

Every weight, keyword family, disqualifier rule, and threshold the ranker uses
lives here, so the scoring logic stays readable and the JD interpretation is
auditable in one place. Grounded in the released JD (Senior AI Engineer,
founding team) and the observed dataset (see PRD_v2.md and the rubric memory).

Reminder from the data: skill *presence* is near-noise (every skill appears on
~12% of candidates), so nothing here rewards bare skill listing. The signal is
title trajectory + production evidence in career descriptions.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Title families. A candidate's title category is the single strongest cheap
# signal of whether they are actually an AI/ML engineer vs a keyword stuffer.
# Matched by substring against the lowercased title, most specific first.
# ---------------------------------------------------------------------------

# Bullseye: the JD's exact domain — retrieval / ranking / search / recsys.
TITLE_BULLSEYE = (
    "recommendation systems engineer", "recommendation engineer", "recsys",
    "search engineer", "search relevance", "ranking engineer",
)
# Core ML/AI engineering roles.
TITLE_CORE_ML = (
    "machine learning engineer", "ml engineer", "(ml)", "applied ml",
    "applied scientist", "ai engineer", "applied ai", "nlp engineer",
)
# Strong-but-broader ML/data-science roles.
TITLE_STRONG_ML = (
    "data scientist", "ai specialist", "ml scientist", "research scientist",
)
# Research-leaning — fine only with production evidence (JD disqualifies pure research).
TITLE_RESEARCH = ("research engineer", "ai research", "research scientist")
# Perception roles — JD down-weights CV/speech/robotics without NLP/IR.
TITLE_PERCEPTION = (
    "computer vision", "vision engineer", "speech", "robotics", "perception",
)
# Adjacent engineering that can transition into ML (promotable via evidence).
TITLE_ADJACENT = (
    "data engineer", "analytics engineer", "backend engineer", "ml ops",
    "mlops", "platform engineer",
)
# Generic software — weak prior for this role.
TITLE_SWE = (
    "software engineer", "developer", "full stack", "frontend", "front end",
    "devops", "cloud engineer", "qa engineer", "mobile", "java", ".net",
    "sde", "programmer",
)
# Clearly off-domain decoys (the keyword-stuffer carriers).
TITLE_OFF_DOMAIN = (
    "hr manager", "accountant", "mechanical engineer", "marketing manager",
    "sales executive", "civil engineer", "graphic designer", "content writer",
    "operations manager", "project manager", "customer support",
    "business analyst", "recruiter", "finance",
)

# base title score per category (before career-evidence adjustment)
TITLE_SCORE = {
    "bullseye": 1.00,
    "core_ml": 0.90,
    "strong_ml": 0.78,
    "research": 0.62,
    "perception": 0.55,
    "adjacent": 0.50,
    "swe": 0.32,
    "off_domain": 0.05,
}


def classify_title(title_lower: str) -> str:
    t = title_lower
    # Research and perception are checked BEFORE bullseye/core_ml on purpose:
    #  * "research engineer" contains the substring "search engineer", so a plain
    #    substring test would mis-label researchers as bullseye Search Engineers.
    #  * "ai research engineer" contains "ai engineer".
    # The JD is explicitly wary of research-leaning profiles, so the more specific
    # (and more restrictive) category must win.
    if any(k in t for k in TITLE_RESEARCH):
        return "research"
    if any(k in t for k in TITLE_PERCEPTION):
        return "perception"
    if any(k in t for k in TITLE_BULLSEYE):
        return "bullseye"
    if any(k in t for k in TITLE_CORE_ML):
        return "core_ml"
    if any(k in t for k in TITLE_STRONG_ML):
        return "strong_ml"
    if any(k in t for k in TITLE_ADJACENT):
        return "adjacent"
    if any(k in t for k in TITLE_OFF_DOMAIN):
        return "off_domain"
    if any(k in t for k in TITLE_SWE):
        return "swe"
    return "off_domain"


# ---------------------------------------------------------------------------
# Production-evidence phrases scanned in career_text (summary + headline + all
# role descriptions, lowercased). This is what separates a real engineer from a
# stuffer: the stuffer lists AI skills but their descriptions are about sales,
# accounting, brand design. Longer phrases use substring; short acronyms use
# word boundaries to avoid false hits.
# ---------------------------------------------------------------------------

# The JD's exact domain. Highest weight.
DOMAIN_CORE = (
    "learning-to-rank", "learning to rank", "ranking model", "ranking layer",
    "ranking system", "re-ranking", "reranking", "embedding-based retrieval",
    "dense retrieval", "hybrid retrieval", "hybrid search", "semantic search",
    "vector search", "nearest-neighbor", "nearest neighbor", "recommendation system",
    "recommender", "recommendation-style", "collaborative filtering", "recsys",
    "search relevance", "information retrieval", "retrieval", "discovery feed",
)
# General applied-ML work. Medium weight.
ML_GENERAL = (
    "machine learning", "deep learning", "nlp", "natural language",
    "transformer", "sentence-transformer", "embeddings", "fine-tun",
    "classification", "feature engineering", "pytorch", "tensorflow",
    "scikit-learn", "xgboost", "lightgbm", "model training", "mlops",
    "predictive model", "language model",
)
# Production / scale / rigor. Corroborates "shipped to real users".
PRODUCTION = (
    "production", "shipped", "deployed", "serving", "real users", "at scale",
    "latency", "throughput", "a/b test", "offline-online", "offline to online",
    "online metrics", "inference service", "10m+", "100k", "1m+", "million",
)
# Evaluation rigor — the JD calls this out explicitly.
EVAL_RIGOR = ("ndcg", "mrr", " map ", "precision@", "recall@", "offline-online",
              "relevance labeling", "click-through", "a/b test")

# Vector-DB / retrieval infra named tools (JD: "the specific tech doesn't matter").
VECTOR_INFRA = (
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "pgvector", "vespa", "vector database", "vector db",
)

# ---------------------------------------------------------------------------
# Disqualifier / down-weight signals (JD "Things we explicitly do NOT want").
# ---------------------------------------------------------------------------

# Self-taught / dabbler language — the keyword-stuffer summary tell.
DABBLER_PHRASES = (
    "taking online courses", "online courses on", "self-learner", "self learner",
    "experimenting with", "excited about how ai", "excited about how genai",
    "ai tools can augment", "ai tools could augment", "curious about how ai",
    "played with the openai", "keeping up with ai/ml", "side project", "side-project",
    "interested in transitioning", "want to transition", "looking to break into",
    "hobby", "kaggle competitions",
)
# Pure-research tell (only a problem absent production evidence).
RESEARCH_PHRASES = (
    "published", "publications", "papers", "academic", "phd research",
    "research lab", "purely research", "research-only",
)
# Consulting / services firms — JD rejects candidates whose ENTIRE career is here.
CONSULTING_FIRMS = (
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "mindtree", "mphasis", "hcl", "ltimindtree",
    "ltl", "deloitte", "pwc", " key skills",
)
CONSULTING_INDUSTRIES = ("it services", "consulting")

# Big-tech firms. The JD: "if you've spent your career at Google or Meta and you
# want a well-scoped role with a defined ladder, this isn't it." A soft culture
# signal, applied as a mild down-weight only when the ENTIRE career is big-tech
# with no scrappy / startup / smaller-product experience.
BIG_TECH_FIRMS = (
    "google", "meta", "facebook", "apple", "amazon", "microsoft", "netflix",
)

# Perception-only domains (CV/speech/robotics) — problem only without NLP/IR.
PERCEPTION_PHRASES = (
    "computer vision", "image classification", "object detection", "image moderation",
    "resnet", "yolo", "speech recognition", "asr", "text-to-speech", " tts ",
    "robotics", "diffusion model", "gans",
)
NLP_IR_PHRASES = (
    "nlp", "natural language", "information retrieval", "retrieval", "search",
    "ranking", "recommendation", "embeddings", "semantic", "text",
)

# Product-company industry hints (positive — JD wants product over services).
PRODUCT_INDUSTRIES = (
    "software", "fintech", "e-commerce", "food delivery", "saas", "ai/ml",
    "edtech", "healthtech", "gaming", "adtech", "transportation",
    "conversational ai", "insurance tech", "ai services", "healthtech ai",
)

# ---------------------------------------------------------------------------
# Location (JD: Pune/Noida preferred; Hyderabad/Mumbai/Delhi-NCR/Bangalore welcome).
# ---------------------------------------------------------------------------
LOCATION_PREFERRED = ("pune", "noida")
LOCATION_WELCOME = (
    "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "ncr", "bangalore",
    "bengaluru",
)

# ---------------------------------------------------------------------------
# Experience band (JD "5-9 years", soft).
# ---------------------------------------------------------------------------
EXP_IDEAL_LOW = 5.0
EXP_IDEAL_HIGH = 9.0
EXP_HARD_LOW = 2.0     # below this, experience fit decays toward 0
EXP_HARD_HIGH = 16.0

# ---------------------------------------------------------------------------
# Dimension weights for the Stage 3 composite (sum need not be 1; normalized
# by construction since each dimension is in [0,1]).
# ---------------------------------------------------------------------------
WEIGHTS = {
    "role_fit": 0.34,          # is this an AI/ML production engineer? (dominant)
    "production_evidence": 0.24,  # JD-domain retrieval/ranking/recsys evidence
    "experience_fit": 0.12,
    "trajectory": 0.10,        # product-company applied-ML growth
    "domain_fit": 0.10,        # NLP/IR alignment
    "location_fit": 0.10,
}

# Multiplicative down-weights applied for disqualifier hits (Stage 3).
PENALTY = {
    "dabbler": 0.45,           # self-taught-only / keyword stuffer summary
    "pure_research": 0.65,     # research without production
    "consulting_only": 0.70,   # entire career at services firms
    "perception_only": 0.70,   # CV/speech/robotics without NLP/IR
    "title_chaser": 0.80,      # job-hopping <1.5yr for title bumps
    "stale_offdomain": 0.50,   # off-domain title with no ML career evidence
    "big_tech_only": 0.90,     # entire career at big-tech (JD favors product/startup builders)
}

# Company sizes that count as "scrappy / smaller product" experience — presence
# of any such stint exempts a candidate from the big_tech_only down-weight.
SMALL_COMPANY_SIZES = ("1-10", "11-50", "51-200", "201-500")


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def count_hits(text: str, phrases) -> int:
    """Number of distinct phrases from `phrases` present in `text` (lowercased)."""
    return sum(1 for p in phrases if p in text)


_WORD_CACHE: dict[str, re.Pattern] = {}


def has_word(text: str, word: str) -> bool:
    pat = _WORD_CACHE.get(word)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(word) + r"\b")
        _WORD_CACHE[word] = pat
    return pat.search(text) is not None
