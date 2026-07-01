from __future__ import annotations

import json
import math
import re
from datetime import date
from functools import lru_cache
from typing import Any, Iterable


BUNDLE_DATE = date(2026, 6, 30)

RETRIEVAL_TERMS = {
    "retrieval",
    "information retrieval",
    "ranking",
    "ranker",
    "re-ranking",
    "reranking",
    "search",
    "semantic search",
    "hybrid search",
    "vector search",
    "recommendation",
    "recommender",
    "recommendation systems",
    "embeddings",
    "sentence transformers",
    "faiss",
    "milvus",
    "pinecone",
    "qdrant",
    "weaviate",
    "elasticsearch",
    "opensearch",
    "bm25",
}

EVAL_TERMS = {
    "ndcg",
    "mrr",
    "map",
    "offline benchmark",
    "offline-online",
    "a/b",
    "ab test",
    "experimentation",
    "evaluation framework",
    "ranking metrics",
    "metric",
}

PRODUCTION_TERMS = {
    "production",
    "deployed",
    "shipped",
    "real users",
    "scale",
    "on-call",
    "latency",
    "index refresh",
    "drift",
    "regression",
    "pipeline",
    "platform",
    "infrastructure",
}

ML_TERMS = {
    "machine learning",
    "ml",
    "llm",
    "llms",
    "fine-tuning",
    "finetuning",
    "lora",
    "qlora",
    "peft",
    "nlp",
    "deep learning",
    "xgboost",
    "lightgbm",
    "pytorch",
    "tensorflow",
    "scikit-learn",
    "sklearn",
    "mlops",
}

PYTHON_ENGINEERING_TERMS = {
    "python",
    "fastapi",
    "django",
    "flask",
    "docker",
    "kubernetes",
    "spark",
    "airflow",
    "kafka",
    "sql",
    "microservices",
    "backend",
}

PRODUCT_TERMS = {
    "product",
    "startup",
    "marketplace",
    "saas",
    "recruiter",
    "hr-tech",
    "hr tech",
    "pm",
    "user",
    "users",
}

CV_SPEECH_TERMS = {
    "computer vision",
    "image classification",
    "object detection",
    "opencv",
    "yolo",
    "speech",
    "asr",
    "tts",
    "robotics",
}

FRAMEWORK_ONLY_TERMS = {"langchain", "openai api", "prompt engineering", "llamaindex"}

TECHNICAL_TITLES = {
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "applied ml engineer",
    "applied scientist",
    "search engineer",
    "ranking engineer",
    "recommendation systems engineer",
    "recommender systems engineer",
    "nlp engineer",
    "data scientist",
    "senior data scientist",
    "senior software engineer (ml)",
    "ml platform engineer",
    "data engineer",
    "senior data engineer",
    "analytics engineer",
    "backend engineer",
    "software engineer",
    "senior software engineer",
    "full stack developer",
    "cloud engineer",
    "devops engineer",
    "java developer",
    ".net developer",
}

HIGH_SIGNAL_TITLES = {
    "ai engineer",
    "ml engineer",
    "machine learning engineer",
    "applied ml engineer",
    "applied scientist",
    "search engineer",
    "ranking engineer",
    "recommendation systems engineer",
    "recommender systems engineer",
    "nlp engineer",
    "data scientist",
    "senior data scientist",
    "senior software engineer (ml)",
    "ml platform engineer",
}

NONTECH_TITLES = {
    "hr manager",
    "marketing manager",
    "sales executive",
    "accountant",
    "graphic designer",
    "content writer",
    "customer support",
    "civil engineer",
    "mechanical engineer",
    "operations manager",
    "project manager",
    "business analyst",
}

SERVICES_COMPANIES = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
    "hcl",
    "tech mahindra",
    "mphasis",
}

PRODUCT_COMPANIES = {
    "hooli",
    "initech",
    "pied piper",
    "stark industries",
    "wayne enterprises",
    "globex inc",
    "cred",
    "zomato",
    "flipkart",
    "swiggy",
    "razorpay",
    "freshworks",
    "zoho",
    "paytm",
    "phonepe",
    "meesho",
    "inmobi",
    "nykaa",
    "ola",
    "policybazaar",
    "vedantu",
}

TARGET_LOCATIONS = {
    "noida",
    "pune",
    "delhi",
    "gurgaon",
    "mumbai",
    "hyderabad",
    "bangalore",
    "bengaluru",
}

EVIDENCE_TERMS = sorted(
    RETRIEVAL_TERMS
    | EVAL_TERMS
    | PRODUCTION_TERMS
    | ML_TERMS
    | PYTHON_ENGINEERING_TERMS
    | PRODUCT_TERMS
)

ASPECT_QUERIES = {
    "retrieval": "production embeddings retrieval systems vector search faiss milvus pinecone qdrant elasticsearch opensearch hybrid search",
    "ranking": "ranking recommendation systems recommender learning to rank discovery feed xgboost lightgbm ndcg map mrr",
    "evaluation": "evaluation frameworks offline benchmarks online ab tests ndcg mrr map offline online correlation recruiter feedback",
    "production_ml": "production machine learning deployed ml systems mlops latency scale index refresh regression monitoring",
    "python": "strong python backend product engineering shipped production code fastapi django data pipelines",
    "shipper": "startup product engineering scrappy shipping product manager users recruiter workflows marketplace",
    "finetuning": "llm fine tuning lora qlora peft transformers rag",
}

NARRATIVE_RELEVANCE_TERMS = {
    "candidate-jd",
    "discovery",
    "embedding",
    "matching layer",
    "personalization",
    "ranking",
    "ranker",
    "recommendation",
    "relevance",
    "relevant content",
    "relevant matches",
    "retrieval",
    "search",
}

NARRATIVE_PRODUCTION_TERMS = {
    "a/b test",
    "built",
    "deployed",
    "infrastructure",
    "latency",
    "millions",
    "owned",
    "production",
    "rollout",
    "scale",
    "shipped",
}

NARRATIVE_EVALUATION_TERMS = {
    "calibration",
    "evaluation",
    "experiment",
    "human judgments",
    "metrics",
    "ndcg",
    "offline",
    "online",
}

NUMERIC_FEATURES = [
    "years_experience",
    "title_fit",
    "high_signal_title",
    "technical_title",
    "career_system_fit",
    "retrieval_ranking_depth",
    "production_ml_depth",
    "evaluation_experimentation_fit",
    "python_engineering_fit",
    "product_company_fit",
    "experience_fit",
    "location_relocation_fit",
    "behavioral_availability_fit",
    "notice_period_score",
    "response_rate_score",
    "activity_recency_score",
    "open_to_work_score",
    "skill_trust_score",
    "consistency_score",
    "honeypot_risk",
    "keyword_stuffing_penalty",
    "title_mismatch_penalty",
    "services_penalty",
    "nontechnical_ai_stuffer",
    "no_production_evidence",
    "outside_india_penalty",
    "inactive_low_response_penalty",
    "github_score_norm",
    "market_signal_score",
    "career_narrative_strength",
    "career_narrative_band",
    "current_narrative_band",
    "rare_expert_narrative",
    "current_rare_expert_narrative",
    "high_relevance_narrative_count",
    "weighted_formula_score",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def lower_text(value: Any) -> str:
    return clean_text(value).lower()


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def term_hits(text: str, terms: Iterable[str]) -> int:
    low = text.lower()
    return sum(1 for term in terms if term in low)


def term_score(text: str, terms: Iterable[str], scale: float = 5.0) -> float:
    return clamp(term_hits(text, terms) / scale)


def description_frequency_band(frequency: int) -> int:
    """Return the empirical career-narrative frequency band from 0 to 5."""
    if frequency <= 0:
        return 0
    if frequency > 20_000:
        return 0
    if frequency > 5_000:
        return 1
    if frequency > 1_000:
        return 2
    if frequency > 100:
        return 3
    if frequency > 20:
        return 4
    return 5


@lru_cache(maxsize=4096)
def narrative_description_profile(
    description: str,
    frequency: int,
) -> tuple[int, float, bool, bool]:
    band = description_frequency_band(frequency)
    relevance_hits = term_hits(description, NARRATIVE_RELEVANCE_TERMS)
    production_hits = term_hits(description, NARRATIVE_PRODUCTION_TERMS)
    evaluation_hits = term_hits(description, NARRATIVE_EVALUATION_TERMS)
    semantic = clamp(
        0.50 * clamp(relevance_hits / 3.0)
        + 0.28 * clamp(production_hits / 3.0)
        + 0.22 * clamp(evaluation_hits / 2.0)
    )
    band_score = band / 5.0
    strength = band_score * (
        0.62 + 0.38 * semantic if relevance_hits else 0.12 * semantic
    )
    rare_expert = (
        band == 5
        and relevance_hits > 0
        and (production_hits > 0 or evaluation_hits > 0)
    )
    high_relevance = band >= 4 and relevance_hits > 0
    return band, strength, rare_expert, high_relevance


def career_narrative_features(
    histories: list[dict[str, Any]],
    description_frequencies: dict[str, int] | None,
) -> dict[str, float]:
    empty = {
        "career_narrative_strength": 0.0,
        "career_narrative_band": 0.0,
        "current_narrative_band": 0.0,
        "rare_expert_narrative": 0.0,
        "current_rare_expert_narrative": 0.0,
        "high_relevance_narrative_count": 0.0,
    }
    if not description_frequencies:
        return empty

    strongest = 0.0
    max_band = 0
    current_band = 0
    rare_expert = 0.0
    current_rare_expert = 0.0
    high_relevance_count = 0
    for job in histories:
        description = clean_text(job.get("description"))
        frequency = int(description_frequencies.get(description, 0))
        if not description or frequency <= 0:
            continue
        band, strength, is_rare_expert, high_relevance = narrative_description_profile(
            description,
            frequency,
        )
        max_band = max(max_band, band)
        if job.get("is_current"):
            current_band = max(current_band, band)
        strongest = max(strongest, strength)

        if high_relevance:
            high_relevance_count += 1
        if is_rare_expert:
            rare_expert = 1.0
            if job.get("is_current"):
                current_rare_expert = 1.0

    return {
        "career_narrative_strength": strongest,
        "career_narrative_band": max_band / 5.0,
        "current_narrative_band": current_band / 5.0,
        "rare_expert_narrative": rare_expert,
        "current_rare_expert_narrative": current_rare_expert,
        "high_relevance_narrative_count": clamp(high_relevance_count / 3.0),
    }


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        y, m, d = str(value)[:10].split("-")
        return date(int(y), int(m), int(d))
    except (TypeError, ValueError):
        return None


def months_between(start: Any, end: Any) -> int | None:
    start_date = parse_date(start)
    end_date = parse_date(end) or BUNDLE_DATE
    if not start_date:
        return None
    return max(0, (end_date.year - start_date.year) * 12 + end_date.month - start_date.month)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|;\s+", clean_text(text))
    return [p.strip(" -") for p in parts if p.strip()]


def weighted_skill_text(candidate: dict[str, Any]) -> str:
    red = candidate.get("redrob_signals", {})
    assessments = {str(k).lower(): v for k, v in red.get("skill_assessment_scores", {}).items()}
    chunks: list[str] = []
    for skill in candidate.get("skills", []):
        name = clean_text(skill.get("name"))
        if not name:
            continue
        prof = skill.get("proficiency", "beginner")
        duration = int(skill.get("duration_months") or 0)
        endorsements = int(skill.get("endorsements") or 0)
        assess = float(assessments.get(name.lower(), 0.0) or 0.0)
        repeats = 1
        if prof in {"advanced", "expert"}:
            repeats += 1
        if duration >= 24:
            repeats += 1
        if endorsements >= 10:
            repeats += 1
        if assess >= 70:
            repeats += 2
        chunks.extend([name] * min(repeats, 5))
    return " ".join(chunks)


def candidate_text_views(candidate: dict[str, Any]) -> dict[str, str]:
    profile = candidate.get("profile", {})
    histories = candidate.get("career_history", [])
    career_parts = [
        profile.get("current_title"),
        profile.get("current_company"),
        profile.get("current_industry"),
    ]
    for job in histories:
        career_parts.extend(
            [
                job.get("title"),
                job.get("company"),
                job.get("industry"),
                job.get("description"),
            ]
        )
    career_text = clean_text(" ".join(clean_text(p) for p in career_parts if p))
    profile_text = clean_text(f"{profile.get('headline', '')} {profile.get('summary', '')}")
    skills_text = weighted_skill_text(candidate)
    all_text = clean_text(f"{career_text} {profile_text} {skills_text}")
    evidence_parts = []
    for sentence in split_sentences(all_text):
        if term_hits(sentence, EVIDENCE_TERMS):
            evidence_parts.append(sentence)
    if not evidence_parts:
        evidence_parts = split_sentences(career_text)[:2]
    evidence_text = clean_text(" ".join(evidence_parts[:8]))
    return {
        "career_text": career_text,
        "profile_text": profile_text,
        "skills_text": skills_text,
        "evidence_text": evidence_text,
        "all_text": clean_text(f"{career_text} {profile_text} {skills_text} {evidence_text}"),
    }


def _company_sets(candidate: dict[str, Any]) -> tuple[set[str], set[str]]:
    profile = candidate.get("profile", {})
    companies = {lower_text(profile.get("current_company"))}
    industries = {lower_text(profile.get("current_industry"))}
    for job in candidate.get("career_history", []):
        companies.add(lower_text(job.get("company")))
        industries.add(lower_text(job.get("industry")))
    return {c for c in companies if c}, {i for i in industries if i}


def _skill_trust(candidate: dict[str, Any], career_low: str) -> tuple[float, int, int]:
    red = candidate.get("redrob_signals", {})
    assessments = {str(k).lower(): float(v) for k, v in red.get("skill_assessment_scores", {}).items()}
    trusted = 0.0
    ai_skill_count = 0
    suspicious_expert = 0
    for skill in candidate.get("skills", []):
        name = clean_text(skill.get("name"))
        low = name.lower()
        if not name:
            continue
        duration = int(skill.get("duration_months") or 0)
        prof = skill.get("proficiency", "")
        if prof == "expert" and duration <= 1:
            suspicious_expert += 1
        is_jd = any(term in low or low in term for term in RETRIEVAL_TERMS | ML_TERMS | EVAL_TERMS | PYTHON_ENGINEERING_TERMS)
        if not is_jd:
            continue
        ai_skill_count += 1
        endorsements = int(skill.get("endorsements") or 0)
        assess = assessments.get(low, 0.0)
        local = 0.10
        if low in career_low:
            local += 0.45
        if assess >= 70:
            local += 0.35
        elif assess >= 50:
            local += 0.18
        if duration >= 24 and endorsements >= 10:
            local += 0.20
        elif duration >= 12 or endorsements >= 8:
            local += 0.10
        if prof == "expert":
            local += 0.08
        trusted += clamp(local)
    return clamp(trusted / 5.0), ai_skill_count, suspicious_expert


def _experience_fit(years: float) -> float:
    if 5.0 <= years <= 9.0:
        return 1.0
    if 4.0 <= years < 5.0 or 9.0 < years <= 10.5:
        return 0.82
    if 3.0 <= years < 4.0 or 10.5 < years <= 12.0:
        return 0.58
    if years < 3.0:
        return 0.30
    return 0.42


def _activity_score(last_active: Any) -> float:
    dt = parse_date(last_active)
    if not dt:
        return 0.25
    days = max(0, (BUNDLE_DATE - dt).days)
    if days <= 30:
        return 1.0
    if days <= 90:
        return 0.82
    if days <= 180:
        return 0.55
    if days <= 365:
        return 0.25
    return 0.10


def _notice_score(days: Any) -> float:
    try:
        value = float(days)
    except (TypeError, ValueError):
        return 0.40
    if value <= 30:
        return 1.0
    if value <= 60:
        return 0.72
    if value <= 90:
        return 0.48
    if value <= 120:
        return 0.28
    return 0.14


def extract_features(
    candidate: dict[str, Any],
    description_frequencies: dict[str, int] | None = None,
) -> dict[str, Any]:
    profile = candidate.get("profile", {})
    red = candidate.get("redrob_signals", {})
    views = candidate_text_views(candidate)
    career_low = views["career_text"].lower()
    profile_low = views["profile_text"].lower()
    evidence_low = views["evidence_text"].lower()
    all_low = views["all_text"].lower()
    title = lower_text(profile.get("current_title"))
    years = float(profile.get("years_of_experience") or 0.0)
    companies, industries = _company_sets(candidate)
    histories = candidate.get("career_history", [])
    narrative = career_narrative_features(histories, description_frequencies)

    retrieval_depth = clamp(
        0.52 * term_score(career_low + " " + evidence_low, RETRIEVAL_TERMS, 4.0)
        + 0.22 * term_score(views["skills_text"], RETRIEVAL_TERMS, 4.0)
        + 0.26 * term_score(career_low, {"shipped", "production", "built", "deployed", "owned"}, 4.0)
    )
    production_depth = clamp(
        0.50 * term_score(career_low + " " + evidence_low, PRODUCTION_TERMS, 5.0)
        + 0.30 * term_score(career_low + " " + evidence_low, ML_TERMS, 5.0)
        + 0.20 * term_score(career_low, {"built", "owned", "implemented", "designed", "maintained"}, 4.0)
    )
    eval_fit = clamp(0.75 * term_score(career_low + " " + evidence_low, EVAL_TERMS, 3.0) + 0.25 * term_score(all_low, EVAL_TERMS, 5.0))
    python_fit = clamp(0.62 * term_score(career_low + " " + evidence_low, PYTHON_ENGINEERING_TERMS, 6.0) + 0.38 * term_score(views["skills_text"], PYTHON_ENGINEERING_TERMS, 5.0))
    title_fit = 1.0 if title in HIGH_SIGNAL_TITLES else 0.72 if title in TECHNICAL_TITLES else 0.22 if title in NONTECH_TITLES else 0.45
    career_system_fit = clamp(
        0.32 * title_fit
        + 0.34 * retrieval_depth
        + 0.22 * production_depth
        + 0.12 * term_score(career_low, PRODUCT_TERMS, 4.0)
    )
    product_company_fit = 0.0
    if companies & PRODUCT_COMPANIES:
        product_company_fit += 0.55
    if industries & {"software", "saas", "ai/ml", "fintech", "food delivery", "e-commerce", "edtech", "adtech", "healthtech ai", "conversational ai", "ai services"}:
        product_company_fit += 0.28
    product_company_fit += 0.17 * term_score(career_low + " " + profile_low, PRODUCT_TERMS, 5.0)
    product_company_fit = clamp(product_company_fit)

    country = lower_text(profile.get("country"))
    location = lower_text(profile.get("location"))
    willing = bool(red.get("willing_to_relocate"))
    if country == "india":
        location_fit = 0.62
        if any(loc in location for loc in TARGET_LOCATIONS):
            location_fit = 1.0
        elif willing:
            location_fit = 0.78
    else:
        location_fit = 0.42 if willing else 0.18

    response_rate = float(red.get("recruiter_response_rate") or 0.0)
    avg_response_hours = float(red.get("avg_response_time_hours") or 280.0)
    response_speed = 1.0 - clamp(avg_response_hours / 240.0)
    response_score = clamp(0.70 * response_rate + 0.30 * response_speed)
    activity_score = _activity_score(red.get("last_active_date"))
    notice_score = _notice_score(red.get("notice_period_days"))
    open_score = 1.0 if red.get("open_to_work_flag") else 0.35
    verified_score = (0.5 if red.get("verified_email") else 0.0) + (0.3 if red.get("verified_phone") else 0.0) + (0.2 if red.get("linkedin_connected") else 0.0)
    interview_score = float(red.get("interview_completion_rate") or 0.0)
    offer_acceptance = float(red.get("offer_acceptance_rate") if red.get("offer_acceptance_rate", -1) != -1 else 0.45)
    behavioral = clamp(
        0.25 * response_score
        + 0.22 * activity_score
        + 0.18 * open_score
        + 0.13 * notice_score
        + 0.10 * interview_score
        + 0.07 * offer_acceptance
        + 0.05 * verified_score
    )

    skill_trust, ai_skill_count, suspicious_expert = _skill_trust(candidate, career_low)

    total_declared_months = years * 12.0
    career_month_sum = 0
    duration_mismatch_jobs = 0
    short_stints = 0
    for job in histories:
        declared = int(job.get("duration_months") or 0)
        career_month_sum += declared
        computed = months_between(job.get("start_date"), job.get("end_date"))
        if computed is not None and abs(computed - declared) > 3:
            duration_mismatch_jobs += 1
        if declared and declared < 18:
            short_stints += 1
    total_gap_years = abs((career_month_sum - total_declared_months) / 12.0) if histories else 0.0
    consistency_penalty = 0.0
    consistency_penalty += min(0.45, total_gap_years / 8.0)
    consistency_penalty += min(0.25, duration_mismatch_jobs * 0.08)
    consistency_penalty += min(0.20, suspicious_expert * 0.06)
    consistency_penalty += min(0.10, max(0, short_stints - 3) * 0.03)
    consistency_score = clamp(1.0 - consistency_penalty)

    no_career_evidence = retrieval_depth < 0.18 and production_depth < 0.25 and eval_fit < 0.10
    nontechnical_title = title in NONTECH_TITLES
    keyword_stuffing = clamp((ai_skill_count - 4) / 9.0) if no_career_evidence or nontechnical_title else clamp((ai_skill_count - 9) / 12.0)
    title_mismatch = clamp(0.70 * (1.0 if nontechnical_title and ai_skill_count >= 5 else 0.0) + 0.30 * (1.0 if title_fit < 0.35 and retrieval_depth > 0.45 else 0.0))
    all_services = bool(companies) and companies.issubset(SERVICES_COMPANIES)
    services_penalty = 1.0 if all_services and retrieval_depth < 0.35 and product_company_fit < 0.35 else 0.45 if (companies & SERVICES_COMPANIES and product_company_fit < 0.25 and retrieval_depth < 0.25) else 0.0
    cv_speech_without_ir = term_hits(all_low, CV_SPEECH_TERMS) >= 2 and term_hits(all_low, RETRIEVAL_TERMS) == 0
    honeypot_risk = clamp(
        0.34 * (1.0 if total_gap_years > 3.0 else 0.0)
        + 0.26 * (1.0 if suspicious_expert >= 3 else 0.0)
        + 0.17 * keyword_stuffing
        + 0.12 * title_mismatch
        + 0.06 * (1.0 if duration_mismatch_jobs >= 2 else 0.0)
        + 0.05 * (1.0 if cv_speech_without_ir else 0.0)
    )

    github_raw = float(red.get("github_activity_score") or -1)
    github_norm = 0.0 if github_raw < 0 else clamp(github_raw / 100.0)
    market_signal = clamp(
        0.38 * clamp(float(red.get("saved_by_recruiters_30d") or 0) / 20.0)
        + 0.22 * clamp(float(red.get("search_appearance_30d") or 0) / 300.0)
        + 0.18 * clamp(float(red.get("profile_views_received_30d") or 0) / 120.0)
        + 0.12 * clamp(float(red.get("endorsements_received") or 0) / 80.0)
        + 0.10 * github_norm
    )

    no_prod = 1.0 if production_depth < 0.25 and "production" not in career_low and "shipped" not in career_low else 0.0
    outside_india_penalty = 1.0 if country != "india" and not willing else 0.0
    inactive_low_response = 1.0 if activity_score < 0.35 and response_rate < 0.30 else 0.0
    nontechnical_ai_stuffer = 1.0 if nontechnical_title and ai_skill_count >= 5 and retrieval_depth < 0.25 else 0.0

    base = (
        0.24 * career_system_fit
        + 0.16 * retrieval_depth
        + 0.11 * production_depth
        + 0.10 * eval_fit
        + 0.08 * product_company_fit
        + 0.07 * _experience_fit(years)
        + 0.05 * python_fit
        + 0.05 * location_fit
        + 0.08 * behavioral
        + 0.04 * skill_trust
        + 0.02 * market_signal
    )
    multiplier = 1.0
    if honeypot_risk >= 0.72:
        multiplier *= 0.05
    if nontechnical_ai_stuffer:
        multiplier *= 0.35
    if services_penalty >= 1.0:
        multiplier *= 0.45
    if no_prod:
        multiplier *= 0.60
    if outside_india_penalty:
        multiplier *= 0.55
    if inactive_low_response:
        multiplier *= 0.50
    formula_score = clamp(base * multiplier - 0.08 * keyword_stuffing - 0.05 * title_mismatch)

    strongest = strongest_evidence(candidate, views)
    concern = strongest_concern(
        years=years,
        notice=red.get("notice_period_days"),
        response_rate=response_rate,
        activity_score=activity_score,
        honeypot_risk=honeypot_risk,
        services_penalty=services_penalty,
        no_prod=bool(no_prod),
        title=title,
    )

    output = {
        "candidate_id": candidate["candidate_id"],
        "years_experience": years,
        "current_title": clean_text(profile.get("current_title")),
        "current_company": clean_text(profile.get("current_company")),
        "location": clean_text(profile.get("location")),
        "country": clean_text(profile.get("country")),
        "notice_period_days": int(red.get("notice_period_days") or 0),
        "recruiter_response_rate": response_rate,
        "last_active_date": clean_text(red.get("last_active_date")),
        "open_to_work_flag": bool(red.get("open_to_work_flag")),
        "title_fit": title_fit,
        "high_signal_title": 1.0 if title in HIGH_SIGNAL_TITLES else 0.0,
        "technical_title": 1.0 if title in TECHNICAL_TITLES else 0.0,
        "career_system_fit": career_system_fit,
        "retrieval_ranking_depth": retrieval_depth,
        "production_ml_depth": production_depth,
        "evaluation_experimentation_fit": eval_fit,
        "python_engineering_fit": python_fit,
        "product_company_fit": product_company_fit,
        "experience_fit": _experience_fit(years),
        "location_relocation_fit": location_fit,
        "behavioral_availability_fit": behavioral,
        "notice_period_score": notice_score,
        "response_rate_score": response_score,
        "activity_recency_score": activity_score,
        "open_to_work_score": open_score,
        "skill_trust_score": skill_trust,
        "consistency_score": consistency_score,
        "honeypot_risk": honeypot_risk,
        "keyword_stuffing_penalty": keyword_stuffing,
        "title_mismatch_penalty": title_mismatch,
        "services_penalty": services_penalty,
        "nontechnical_ai_stuffer": nontechnical_ai_stuffer,
        "no_production_evidence": no_prod,
        "outside_india_penalty": outside_india_penalty,
        "inactive_low_response_penalty": inactive_low_response,
        "github_score_norm": github_norm,
        "market_signal_score": market_signal,
        **narrative,
        "weighted_formula_score": formula_score,
        "ai_skill_count": ai_skill_count,
        "suspicious_expert_skill_count": suspicious_expert,
        "reason_strongest_evidence": strongest,
        "reason_concern": concern,
        "feature_json": json.dumps(
            {
                "companies": sorted(companies),
                "industries": sorted(industries),
                "ai_skill_count": ai_skill_count,
                "career_month_sum": career_month_sum,
                "declared_months": total_declared_months,
            },
            ensure_ascii=True,
        ),
    }
    output.update({k: views[k] for k in ["career_text", "profile_text", "skills_text", "evidence_text", "all_text"]})
    return output


def strongest_evidence(candidate: dict[str, Any], views: dict[str, str]) -> str:
    evidence = views["evidence_text"]
    career = views["career_text"]
    scored: list[tuple[int, str]] = []
    for sentence in split_sentences(evidence) + split_sentences(career):
        score = (
            3 * term_hits(sentence, RETRIEVAL_TERMS)
            + 2 * term_hits(sentence, EVAL_TERMS)
            + 2 * term_hits(sentence, PRODUCTION_TERMS)
            + term_hits(sentence, ML_TERMS)
        )
        if score > 0:
            scored.append((score, sentence))
    if scored:
        scored.sort(key=lambda x: (-x[0], len(x[1])))
        return clean_text(scored[0][1])[:220]
    skills = [clean_text(s.get("name")) for s in candidate.get("skills", [])[:6]]
    return clean_text("Skills include " + ", ".join(s for s in skills if s))[:220]


def strongest_concern(
    *,
    years: float,
    notice: Any,
    response_rate: float,
    activity_score: float,
    honeypot_risk: float,
    services_penalty: float,
    no_prod: bool,
    title: str,
) -> str:
    if honeypot_risk >= 0.55:
        return "profile consistency risk"
    if no_prod:
        return "limited explicit production ML evidence"
    if services_penalty >= 1.0:
        return "mostly IT services background"
    if title in NONTECH_TITLES:
        return "current title is not a core AI engineering title"
    try:
        if float(notice) > 30:
            return f"{int(float(notice))}-day notice period"
    except (TypeError, ValueError):
        pass
    if response_rate < 0.25:
        return "low recruiter response rate"
    if activity_score < 0.35:
        return "weaker recent activity"
    if years < 5.0 or years > 9.0:
        return "outside the JD's 5-9 year preference"
    return ""


def pseudo_label(row: dict[str, Any]) -> float:
    label = (
        0.28 * row["career_system_fit"]
        + 0.20 * row["retrieval_ranking_depth"]
        + 0.14 * row["production_ml_depth"]
        + 0.12 * row["evaluation_experimentation_fit"]
        + 0.08 * row["product_company_fit"]
        + 0.06 * row["experience_fit"]
        + 0.04 * row["python_engineering_fit"]
        + 0.04 * row["location_relocation_fit"]
        + 0.04 * row["behavioral_availability_fit"]
    )
    label += 0.04 if row["high_signal_title"] else 0.0
    label += 0.16 * row.get("rare_expert_narrative", 0.0)
    label += 0.04 * row.get("career_narrative_strength", 0.0)
    label -= 0.25 * row["honeypot_risk"]
    label -= 0.13 * row["keyword_stuffing_penalty"]
    label -= 0.10 * row["services_penalty"]
    label -= 0.12 * row["title_mismatch_penalty"]
    label -= 0.08 * row["outside_india_penalty"]
    return clamp(label)


def reasoning_for_row(row: dict[str, Any], rank: int) -> str:
    title = clean_text(row.get("current_title")) or "Candidate"
    years = float(row.get("years_experience") or 0.0)
    evidence = clean_text(row.get("reason_strongest_evidence"))
    concern = clean_text(row.get("reason_concern"))
    location = clean_text(row.get("location"))
    response = float(row.get("recruiter_response_rate") or 0.0)
    notice = int(row.get("notice_period_days") or 0)
    aspects = {
        "retrieval and ranking systems": float(row.get("retrieval_ranking_depth") or 0.0),
        "production ML delivery": float(row.get("production_ml_depth") or 0.0),
        "evaluation and experimentation": float(
            row.get("evaluation_experimentation_fit") or 0.0
        ),
        "product engineering": float(row.get("product_company_fit") or 0.0),
    }
    if aspects["retrieval and ranking systems"] >= 0.60:
        theme = "retrieval and ranking systems"
    elif aspects["evaluation and experimentation"] >= 0.65:
        theme = "evaluation and experimentation"
    elif aspects["production ML delivery"] >= 0.60:
        theme = "production ML delivery"
    else:
        theme = max(aspects, key=aspects.get)
    elite = bool(row.get("rare_expert_narrative")) or (
        float(row.get("career_system_fit") or 0.0) >= 0.70
        and aspects["retrieval and ranking systems"] >= 0.60
        and aspects["production ML delivery"] >= 0.50
    )
    variant = rank % 3
    if rank <= 10 and elite:
        openers = [
            f"Top-tier {theme}: {title} with {years:.1f} yrs",
            f"{title} brings {years:.1f} yrs and stands out for {theme}",
            f"At {years:.1f} yrs, {title} is a top-tier {theme} match",
        ]
    elif rank <= 10:
        openers = [
            f"Ranked {theme} option: {title} with {years:.1f} yrs",
            f"{title} offers {years:.1f} yrs with relative {theme} value",
            f"At {years:.1f} yrs, {title} is a supporting {theme} option",
        ]
    elif rank <= 50:
        openers = [
            f"Strong {theme} fit: {title} with {years:.1f} yrs",
            f"{title} offers {years:.1f} yrs with credible {theme}",
            f"At {years:.1f} yrs, {title} shows strong {theme}",
        ]
    elif rank <= 80:
        openers = [
            f"Solid {theme} fit: {title} with {years:.1f} yrs",
            f"{title} adds {years:.1f} yrs of useful {theme}",
            f"At {years:.1f} yrs, {title} remains credible for {theme}",
        ]
    else:
        openers = [
            f"Top-100 {theme} depth: {title} with {years:.1f} yrs",
            f"{title} retains top-100 value through {years:.1f} yrs in {theme}",
            f"At {years:.1f} yrs, {title} provides top-100 supporting {theme}",
        ]
    opener = openers[variant]
    text = f"{opener}; evidence: {evidence.rstrip(' .;')}" if evidence else opener
    signal_bits = []
    if location:
        signal_bits.append(location)
    if response >= 0.65:
        signal_bits.append(f"{response:.2f} recruiter response")
    if notice <= 30:
        signal_bits.append(f"{notice}-day notice")
    else:
        signal_bits.append(f"concern: {notice}-day notice")
    if concern and "notice period" not in concern:
        signal_bits.append(f"concern: {concern}")
    if signal_bits:
        text += ". " + "; ".join(signal_bits)
    if not text.endswith("."):
        text += "."
    return text[:480]


def stable_minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]

