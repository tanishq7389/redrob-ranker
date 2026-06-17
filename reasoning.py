"""
Generate honest, specific, rank-consistent reasoning for each candidate.
Reasoning must reference actual profile facts — no hallucination.
"""

from __future__ import annotations
from datetime import date, datetime

REFERENCE_DATE = date(2026, 6, 4)


def _days_since(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except Exception:
        return None


def build_reasoning(candidate: dict, score: float, breakdown: dict, rank: int) -> str:
    """
    Build a 1-2 sentence reasoning that is:
    - Specific to the candidate's actual profile
    - Honest about concerns
    - Consistent in tone with the rank
    """
    p = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    title = p.get("current_title", "Unknown role")
    company = p.get("current_company", "")
    yoe = p.get("years_of_experience", 0) or 0
    location = p.get("location", "")
    country = p.get("country", "")

    # Key skill evidence
    ai_skills = [
        s["name"] for s in skills
        if any(kw in s["name"].lower() for kw in [
            "embedding", "vector", "retrieval", "search", "rag", "llm",
            "transformer", "bert", "nlp", "ranking", "faiss", "pinecone",
            "qdrant", "weaviate", "fine-tun", "lora", "sentence",
        ])
    ][:3]

    # Behavioral signals
    rr = signals.get("recruiter_response_rate", 0) or 0
    notice = signals.get("notice_period_days", 90) or 90
    last_active = _days_since(signals.get("last_active_date"))
    open_to_work = signals.get("open_to_work_flag", False)
    github = signals.get("github_activity_score", -1)

    # Production evidence from career
    prod_companies = [
        j["company"] for j in history
        if j.get("company", "").lower() not in {
            "tcs", "infosys", "wipro", "accenture", "cognizant",
            "capgemini", "hcl", "tech mahindra",
        }
        and j.get("is_current") is False
    ][:2]

    loc_str = f"{location}, {country}" if country else location

    # Concerns
    concerns = []
    if notice > 90:
        concerns.append(f"long notice period ({notice}d)")
    if last_active and last_active > 60:
        concerns.append(f"last active {last_active}d ago")
    if rr < 0.3:
        concerns.append(f"low recruiter response rate ({rr:.0%})")
    if breakdown.get("career_trajectory", 0.5) <= 0.15:
        concerns.append("mostly consulting-firm background")
    if breakdown.get("production_evidence", 0.5) < 0.3:
        concerns.append("limited production deployment evidence")
    if country and country not in ("India",) and not signals.get("willing_to_relocate"):
        concerns.append(f"based in {country}, relocation unclear")

    # Build positive part
    positives = []
    if ai_skills:
        positives.append(f"hands-on with {', '.join(ai_skills)}")
    elif breakdown.get("core_skills", 0) > 0.4:
        positives.append("strong ML/IR skill set")

    if breakdown.get("production_evidence", 0) > 0.5:
        positives.append("evidence of production ML deployment")

    if open_to_work:
        positives.append("actively looking")
    elif last_active and last_active <= 14:
        positives.append("active on platform recently")

    if github and github > 40:
        positives.append(f"GitHub score {github:.0f}")

    if notice <= 30:
        positives.append(f"notice {notice}d")

    # Compose sentences
    positive_str = "; ".join(positives) if positives else "general engineering background"

    if rank <= 10:
        # Glowing but honest
        sentence1 = (
            f"{yoe:.1f}yr {title} at {company or 'a product company'} ({loc_str}) — "
            f"{positive_str}."
        )
        if concerns:
            sentence2 = f"Note: {', '.join(concerns[:2])}."
        else:
            sentence2 = "Strong availability and engagement signals."
        return f"{sentence1} {sentence2}"

    elif rank <= 30:
        sentence1 = (
            f"{title} with {yoe:.1f}yr exp ({loc_str}); "
            f"{positive_str}."
        )
        if concerns:
            sentence2 = f"Concern: {', '.join(concerns[:1])}."
        else:
            sentence2 = "Good overall fit."
        return f"{sentence1} {sentence2}"

    elif rank <= 60:
        sentence1 = (
            f"{yoe:.1f}yr {title}; "
            f"{positive_str}."
        )
        if concerns:
            sentence2 = f"Concerns: {'; '.join(concerns[:2])}."
        else:
            sentence2 = "Moderate fit — included for breadth."
        return f"{sentence1} {sentence2}"

    else:
        # Bottom tier — be honest about weakness
        if concerns:
            c_str = "; ".join(concerns[:2])
            return (
                f"{title} ({yoe:.1f}yr); adjacent skills. "
                f"Concerns limit ranking: {c_str}."
            )
        return (
            f"{title} ({yoe:.1f}yr); some relevant skills but limited "
            f"production IR/ML evidence — placed at tail of shortlist."
        )
