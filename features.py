"""
Feature extraction for Redrob candidate ranking.

Scoring philosophy:
  - Semantic role fit (title, headline, summary, career narrative)
  - Hard-skill depth (not just presence, but production evidence)
  - Behavioral availability signals
  - Disqualifier detection (consulting-only, wrong domain, no production)
  - Honeypot detection (impossible timelines, suspiciously perfect profiles)
"""

from __future__ import annotations
import re
import json
from datetime import date, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Constants derived from careful JD reading
# ---------------------------------------------------------------------------

# Must-have skills per JD
CORE_SKILLS = {
    "embedding", "embeddings", "sentence-transformers", "sentence transformer",
    "bge", "e5", "openai embeddings",
    "vector database", "vector db", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "faiss", "milvus",
    "hybrid search", "dense retrieval", "sparse retrieval", "bm25",
    "retrieval", "ranking", "re-ranking", "reranking",
    "information retrieval", "semantic search",
    "ndcg", "mrr", "map", "a/b test", "evaluation framework",
    "learning to rank", "ltr",
    "rag", "retrieval augmented generation",
    "llm", "large language model", "fine-tuning", "lora", "qlora", "peft",
    "recommendation", "recommender", "search",
}

# Nice-to-have skills
BONUS_SKILLS = {
    "xgboost", "lightgbm",
    "distributed systems", "kafka", "spark",
    "inference optimization", "triton", "onnx", "torchscript",
    "open source", "open-source",
    "hr tech", "hrtech", "recruiting", "talent",
    "marketplace",
}

# Strong positive title signals
GOOD_TITLES = {
    "ml engineer", "machine learning engineer", "applied scientist",
    "applied ml", "ai engineer", "nlp engineer", "search engineer",
    "research engineer", "data scientist", "senior engineer",
    "staff engineer", "principal engineer",
    "software engineer",  # acceptable if career shows ML
}

# Disqualifying title patterns (present career, not past)
BAD_TITLES = {
    "marketing", "sales", "hr manager", "human resources", "recruiter",
    "customer support", "customer success", "finance", "accountant",
    "content writer", "seo", "graphic designer", "product designer",
    "operations manager", "business analyst",
    "computer vision", "cv engineer", "robotics", "speech",
}

# Consulting firms that are JD disqualifiers (entire career)
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "l&t infotech", "ltimindtree", "mindtree",
    "persistent systems", "niit technologies",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram",
    "hyderabad", "mumbai", "bangalore", "bengaluru",
}

REFERENCE_DATE = date(2026, 6, 4)  # Competition date


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _text_lower(candidate: dict) -> str:
    """Flatten all textual fields into one lowercase string for matching."""
    parts = []
    p = candidate.get("profile", {})
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
        parts.append(job.get("company", ""))
    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))
    for c in candidate.get("certifications", []):
        parts.append(c.get("name", ""))
    return " ".join(parts).lower()


def _days_since(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual scoring components
# ---------------------------------------------------------------------------

def score_core_skills(candidate: dict) -> float:
    """
    0-1: How many hard core skills does the candidate have?
    Weight by proficiency level and number of endorsements.
    """
    text = _text_lower(candidate)
    skills_list = candidate.get("skills", [])
    skill_names_lower = {s["name"].lower() for s in skills_list}

    matched = 0
    weighted = 0.0

    for skill in CORE_SKILLS:
        # Match against skill list OR career text
        in_skills = any(skill in sn for sn in skill_names_lower)
        in_text = skill in text

        if in_skills or in_text:
            matched += 1
            # Find actual skill entry for proficiency boost
            prof_bonus = 0.0
            for s in skills_list:
                if skill in s["name"].lower():
                    p = s.get("proficiency", "")
                    if p == "expert":
                        prof_bonus = 1.5
                    elif p == "advanced":
                        prof_bonus = 1.2
                    elif p == "intermediate":
                        prof_bonus = 0.9
                    else:
                        prof_bonus = 0.6
                    break
            weighted += max(prof_bonus, 0.8)  # at least 0.8 if found in text only

    # Normalize - 8+ core skill matches is excellent
    raw = weighted / max(len(CORE_SKILLS), 1)
    return min(raw * 3.5, 1.0)  # scale so ~8 skills → 1.0


def score_production_evidence(candidate: dict) -> float:
    """
    0-1: Does the candidate show evidence of production ML deployment?
    Key signals: actual job descriptions mentioning shipped systems.
    """
    production_signals = [
        "production", "deployed", "shipped", "scaled", "serving",
        "real users", "live", "million", "billion", "latency",
        "inference", "api", "endpoint", "microservice",
        "a/b test", "experiment", "online eval",
        "index", "retrieval pipeline", "embedding pipeline",
    ]
    text = _text_lower(candidate)
    hits = sum(1 for sig in production_signals if sig in text)

    # Extra boost if career history (not just summary) mentions production
    career_text = " ".join(
        (j.get("description", "") + " " + j.get("title", "")).lower()
        for j in candidate.get("career_history", [])
    )
    career_hits = sum(1 for sig in production_signals if sig in career_text)

    combined = hits * 0.5 + career_hits * 0.8
    return min(combined / 8.0, 1.0)


def score_career_trajectory(candidate: dict) -> float:
    """
    0-1: Is this a product-company ML career, not a services/consulting career?
    """
    history = candidate.get("career_history", [])
    if not history:
        return 0.2

    total_months = sum(j.get("duration_months", 0) or 0 for j in history)
    consulting_months = 0
    product_ml_months = 0
    consulting_only = True

    for job in history:
        company = job.get("company", "").lower()
        title = job.get("title", "").lower()
        industry = job.get("industry", "").lower()
        duration = job.get("duration_months", 0) or 0

        is_consulting = any(firm in company for firm in CONSULTING_FIRMS)
        if is_consulting:
            consulting_months += duration
        else:
            consulting_only = False

        is_ml_role = any(
            kw in title for kw in [
                "ml", "machine learning", "ai", "data scientist",
                "nlp", "search", "ranking", "research engineer",
                "applied", "deep learning",
            ]
        )
        if is_ml_role and not is_consulting:
            product_ml_months += duration

    if consulting_only:
        return 0.05  # JD explicitly disqualifies consulting-only careers

    consulting_ratio = consulting_months / max(total_months, 1)
    product_ml_ratio = product_ml_months / max(total_months, 1)

    score = 0.3 + product_ml_ratio * 0.5 - consulting_ratio * 0.3
    return max(0.0, min(1.0, score))


def score_title_fit(candidate: dict) -> float:
    """
    0-1: Does current title match the role? Apply disqualifiers.
    """
    title = candidate.get("profile", {}).get("current_title", "").lower()

    # Hard disqualify on non-technical current roles
    if any(bad in title for bad in BAD_TITLES):
        return 0.05

    # Positive title matches
    if any(good in title for good in GOOD_TITLES):
        return 1.0

    # Partial matches
    if any(kw in title for kw in ["engineer", "developer", "scientist", "analyst"]):
        return 0.6

    return 0.3


def score_experience_range(candidate: dict) -> float:
    """
    0-1: Is experience in the 5-9 year sweet spot?
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0

    if 5 <= yoe <= 9:
        return 1.0
    elif 4 <= yoe < 5:
        return 0.85
    elif 9 < yoe <= 12:
        return 0.75
    elif 3 <= yoe < 4:
        return 0.6
    elif yoe > 12:
        return 0.5
    else:  # < 3 years
        return 0.2


def score_location(candidate: dict) -> float:
    """
    0-1: Location preference for Pune/Noida/NCR/Hyderabad/Mumbai/Bangalore.
    Also considers willing_to_relocate.
    """
    p = candidate.get("profile", {})
    location = (p.get("location", "") + " " + p.get("country", "")).lower()
    signals = candidate.get("redrob_signals", {})
    willing = signals.get("willing_to_relocate", False)

    is_preferred = any(loc in location for loc in PREFERRED_LOCATIONS)
    is_india = "india" in location or any(
        loc in location for loc in PREFERRED_LOCATIONS
    )

    if is_preferred:
        return 1.0
    elif is_india or willing:
        return 0.7
    else:
        return 0.3  # Outside India, not willing to relocate


def score_behavioral_signals(candidate: dict) -> float:
    """
    0-1: Behavioral availability and engagement signals.
    """
    s = candidate.get("redrob_signals", {})

    score = 0.5  # baseline

    # Recency: last active
    days_ago = _days_since(s.get("last_active_date"))
    if days_ago is not None:
        if days_ago <= 30:
            score += 0.2
        elif days_ago <= 90:
            score += 0.1
        elif days_ago > 180:
            score -= 0.2

    # Open to work
    if s.get("open_to_work_flag"):
        score += 0.1

    # Recruiter response rate (key signal)
    rr = s.get("recruiter_response_rate", 0.0) or 0.0
    score += (rr - 0.3) * 0.3  # bonus for high responders

    # Notice period - JD wants sub-30, can buy out 30
    notice = s.get("notice_period_days", 90) or 90
    if notice <= 30:
        score += 0.1
    elif notice <= 60:
        score += 0.0
    elif notice > 90:
        score -= 0.1

    # Interview completion rate
    icr = s.get("interview_completion_rate", 0.5) or 0.5
    score += (icr - 0.5) * 0.15

    # GitHub activity (for an AI engineer this matters)
    github = s.get("github_activity_score", -1)
    if github > 50:
        score += 0.1
    elif github > 20:
        score += 0.05
    elif github == -1:
        score -= 0.05

    # Profile completeness
    completeness = s.get("profile_completeness_score", 50) or 50
    score += (completeness - 50) / 500.0

    return max(0.0, min(1.0, score))


def score_domain_fit(candidate: dict) -> float:
    """
    0-1: Is the candidate in the right domain (NLP/IR/ML vs CV/Speech/Robotics)?
    JD explicitly says CV/Speech/Robotics without NLP is a disqualifier.
    """
    text = _text_lower(candidate)

    # Negative domain signals (without accompanying IR/NLP)
    bad_domain_kw = [
        "computer vision", "image segmentation", "object detection",
        "robotics", "autonomous", "slam", "lidar",
        "speech recognition", "asr", "tts", "text to speech",
        "audio processing",
    ]
    good_domain_kw = [
        "nlp", "natural language", "text", "retrieval", "search",
        "ranking", "embedding", "transformer", "bert", "llm",
        "recommendation", "information retrieval",
    ]

    bad_hits = sum(1 for kw in bad_domain_kw if kw in text)
    good_hits = sum(1 for kw in good_domain_kw if kw in text)

    if bad_hits > 0 and good_hits == 0:
        return 0.1  # Wrong domain per JD
    elif good_hits > 0:
        return min(0.5 + good_hits * 0.08, 1.0)
    else:
        return 0.4


def detect_honeypot(candidate: dict) -> bool:
    """
    Return True if this candidate looks like a honeypot.
    Honeypots: impossible timelines, too-perfect skill counts, etc.
    """
    history = candidate.get("career_history", [])

    for job in history:
        # Check if duration is implausible given company age (we can't know company age)
        # But check if start_date is before reasonable dates
        start = job.get("start_date", "")
        if start:
            try:
                start_year = int(start[:4])
                if start_year < 1990 or start_year > 2026:
                    return True
            except Exception:
                pass

        # Duration vs dates consistency
        dur = job.get("duration_months", 0) or 0
        if dur > 600:  # > 50 years at one company
            return True

    # Too many expert skills (suspicious)
    skills = candidate.get("skills", [])
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count > 15:
        return True

    # Years of experience vs career history mismatch
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    total_career_months = sum(j.get("duration_months", 0) or 0 for j in history)
    # If stated YoE is much larger than career history suggests, suspicious
    if total_career_months > 0 and yoe > 0:
        career_years = total_career_months / 12
        if yoe > career_years * 2.5 and yoe > 10:
            return True

    return False


# ---------------------------------------------------------------------------
# Top-level scoring
# ---------------------------------------------------------------------------

WEIGHTS = {
    "core_skills": 0.28,
    "production_evidence": 0.22,
    "career_trajectory": 0.15,
    "title_fit": 0.10,
    "experience_range": 0.07,
    "location": 0.06,
    "behavioral": 0.07,
    "domain_fit": 0.05,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1"


def score_candidate(candidate: dict) -> tuple[float, dict]:
    """
    Returns (final_score 0-1, breakdown_dict).
    """
    if detect_honeypot(candidate):
        return 0.0, {"honeypot": True}

    components = {
        "core_skills": score_core_skills(candidate),
        "production_evidence": score_production_evidence(candidate),
        "career_trajectory": score_career_trajectory(candidate),
        "title_fit": score_title_fit(candidate),
        "experience_range": score_experience_range(candidate),
        "location": score_location(candidate),
        "behavioral": score_behavioral_signals(candidate),
        "domain_fit": score_domain_fit(candidate),
    }

    # Hard disqualifiers — multiply final score down hard
    multiplier = 1.0
    if components["title_fit"] <= 0.05:  # Clear non-tech role
        multiplier *= 0.2
    if components["career_trajectory"] <= 0.05:  # Consulting-only
        multiplier *= 0.3
    if components["domain_fit"] <= 0.1:  # Wrong domain
        multiplier *= 0.4

    final = sum(v * WEIGHTS[k] for k, v in components.items()) * multiplier
    return round(min(final, 1.0), 4), components
