"""Match a profile to jobs in two stages.

1. Retrieval: embed a query built from the profile and preferences, and take
   the nearest jobs from the vector index (fast, runs over every job).
2. Reranking: ask the LLM to score the top candidates against the profile and
   say why each fits and what is missing (slower, runs over a dozen jobs).

If the model is unavailable or returns something unusable, the vector order is
used as it is, so matching never fails because of the LLM.
"""
import json

from .index import JobIndex
from .llm import LLM, LLMError
from .schemas import Job, Match, Preferences, Profile

RERANK_PROMPT = """You rank job listings for one candidate.
You get the candidate profile, their preferences, and a list of jobs with short ids like "J1".
Return one JSON object: {"ranking": [{"id": "J1", "score": 0-100, "reasons": [..], "gaps": [..]}, ...]}
- Include every job exactly once, best match first.
- "score": how well the job fits this candidate's skills, seniority and preferences.
- "reasons": 1 to 3 short, specific reasons the job fits, each tied to something in the profile.
- "gaps": 0 to 2 short things the job asks for that the profile does not show.
Use only the ids given. Do not invent jobs or facts."""


def build_query(profile: Profile, prefs: Preferences) -> str:
    parts = [
        prefs.role_focus,
        ", ".join(profile.target_roles),
        profile.headline,
        "Skills: " + ", ".join(profile.skills[:15]) if profile.skills else "",
        profile.summary,
    ]
    return ". ".join(part for part in parts if part)


def _vector_match(job: Job, score: float) -> Match:
    return Match(
        job=job,
        score=max(0, min(100, round(score * 100))),
        vector_score=round(score, 4),
        ranked_by="vector",
    )


def _strings(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:200] for item in value if str(item).strip()][:limit]


def llm_rerank(llm: LLM, profile: Profile, prefs: Preferences, hits: list[tuple[Job, float]]) -> list[Match]:
    """Returns the hits in the model's order. Jobs the model left out keep their vector rank at the end."""
    by_short_id = {f"J{i + 1}": hit for i, hit in enumerate(hits)}
    listing = [
        {
            "id": short_id,
            "title": job.title,
            "company": job.company,
            "location": job.location or ("Remote" if job.remote else ""),
            "remote": job.remote,
            "tags": job.tags[:8],
            "description": job.description[:350],
        }
        for short_id, (job, _) in by_short_id.items()
    ]
    data = llm.json(
        [
            {"role": "system", "content": RERANK_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "profile": profile.model_dump(exclude={"source"}),
                        "preferences": prefs.model_dump(),
                        "jobs": listing,
                    }
                ),
            },
        ]
    )
    ranking = data.get("ranking")
    if not isinstance(ranking, list):
        raise LLMError("rerank response has no 'ranking' list")

    matches: list[Match] = []
    used: set[str] = set()
    for item in ranking:
        if not isinstance(item, dict):
            continue
        short_id = str(item.get("id", "")).strip()
        if short_id not in by_short_id or short_id in used:
            continue  # an id the model made up, or a duplicate
        try:
            score = max(0, min(100, int(round(float(item.get("score", 0))))))
        except (TypeError, ValueError):
            continue
        used.add(short_id)
        job, vector_score = by_short_id[short_id]
        matches.append(
            Match(
                job=job,
                score=score,
                vector_score=round(vector_score, 4),
                reasons=_strings(item.get("reasons"), 3),
                gaps=_strings(item.get("gaps"), 2),
                ranked_by="llm",
            )
        )
    if not matches:
        raise LLMError("rerank response contained no usable items")
    matches.sort(key=lambda m: m.score, reverse=True)
    matches += [_vector_match(job, score) for sid, (job, score) in by_short_id.items() if sid not in used]
    return matches


def match_jobs(
    profile: Profile,
    prefs: Preferences,
    index: JobIndex,
    llm: LLM | None,
    *,
    k: int = 8,
    pool: int = 12,
    exclude_ids: set[str] | frozenset[str] = frozenset(),
) -> list[Match]:
    query = build_query(profile, prefs)
    hits = index.search(
        query,
        limit=pool * 2,
        remote_only=prefs.remote_only,
        location=prefs.location,
        exclude_ids=exclude_ids,
    )
    avoid = [word.lower() for word in prefs.avoid_keywords]
    if avoid:
        hits = [
            (job, score)
            for job, score in hits
            if not any(word in f"{job.title} {job.company} {' '.join(job.tags)}".lower() for word in avoid)
        ]
    hits = hits[:pool]
    if not hits:
        return []
    if llm is not None:
        try:
            return llm_rerank(llm, profile, prefs, hits)[:k]
        except LLMError:
            pass
    return [_vector_match(job, score) for job, score in hits][:k]
