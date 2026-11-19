"""Fetch job listings from public job-board APIs and normalise them to Job.

Each source is fetched on its own. If one fails, the others still load and the
failed source is reported, so the index keeps that source's older jobs.
"""
import html
import re
from datetime import datetime, timezone

import httpx

from .schemas import Job

USER_AGENT = "job-match-agent/0.1"
MAX_DESCRIPTION = 1200


def clean_html(text: str | None, limit: int = MAX_DESCRIPTION) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def to_date(value) -> str:
    """Unix timestamps and common date strings -> YYYY-MM-DD ('' if unknown)."""
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d")
    match = re.match(r"(\d{4}-\d{2}-\d{2})", str(value))
    return match.group(1) if match else ""


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in (value or []) if item]


def parse_arbeitnow(payload: dict) -> list[Job]:
    return [
        Job(
            id=f"arbeitnow:{item['slug']}",
            source="arbeitnow",
            title=item.get("title", ""),
            company=item.get("company_name", ""),
            location=item.get("location", ""),
            remote=bool(item.get("remote")),
            tags=_strings(item.get("tags")) + _strings(item.get("job_types")),
            description=clean_html(item.get("description")),
            url=item.get("url", ""),
            posted_at=to_date(item.get("created_at")),
        )
        for item in payload.get("data", [])
        if item.get("slug") and item.get("title")
    ]


def parse_remotive(payload: dict) -> list[Job]:
    return [
        Job(
            id=f"remotive:{item['id']}",
            source="remotive",
            title=item.get("title", ""),
            company=item.get("company_name", ""),
            location=item.get("candidate_required_location", ""),
            remote=True,
            tags=_strings(item.get("tags")) + _strings(item.get("category")),
            description=clean_html(item.get("description")),
            url=item.get("url", ""),
            posted_at=to_date(item.get("publication_date")),
        )
        for item in payload.get("jobs", [])
        if item.get("id") and item.get("title")
    ]


def parse_jobicy(payload: dict) -> list[Job]:
    return [
        Job(
            id=f"jobicy:{item['id']}",
            source="jobicy",
            title=html.unescape(item.get("jobTitle", "")),
            company=html.unescape(item.get("companyName", "")),
            location=item.get("jobGeo", ""),
            remote=True,
            tags=[html.unescape(t) for t in _strings(item.get("jobIndustry")) + _strings(item.get("jobType"))],
            description=clean_html(item.get("jobDescription") or item.get("jobExcerpt")),
            url=item.get("url", ""),
            posted_at=to_date(item.get("pubDate")),
        )
        for item in payload.get("jobs", [])
        if item.get("id") and item.get("jobTitle")
    ]


def parse_himalayas(payload: dict) -> list[Job]:
    return [
        Job(
            id=f"himalayas:{item['guid']}",
            source="himalayas",
            title=item.get("title", ""),
            company=item.get("companyName", ""),
            location=", ".join(_strings(item.get("locationRestrictions"))),
            remote=True,
            tags=[t.replace("-", " ") for t in _strings(item.get("categories"))[:6]],
            description=clean_html(item.get("description") or item.get("excerpt")),
            url=item.get("applicationLink") or item.get("guid", ""),
            posted_at=to_date(item.get("pubDate")),
        )
        for item in payload.get("jobs", [])
        if item.get("guid") and item.get("title")
    ]


# name -> (list of URLs to fetch, parser)
SOURCES = {
    "arbeitnow": (
        [f"https://www.arbeitnow.com/api/job-board-api?page={page}" for page in (1, 2)],
        parse_arbeitnow,
    ),
    "remotive": (["https://remotive.com/api/remote-jobs"], parse_remotive),
    "jobicy": (["https://jobicy.com/api/v2/remote-jobs?count=50"], parse_jobicy),
    "himalayas": (
        [f"https://himalayas.app/jobs/api?limit=20&offset={offset}" for offset in range(0, 200, 20)],
        parse_himalayas,
    ),
}


def fetch_all(client: httpx.Client | None = None) -> tuple[list[Job], set[str], dict[str, str]]:
    """Returns (jobs, sources that loaded completely, errors by source)."""
    own_client = client is None
    client = client or httpx.Client(timeout=30, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    jobs: list[Job] = []
    complete: set[str] = set()
    errors: dict[str, str] = {}
    try:
        for name, (urls, parse) in SOURCES.items():
            try:
                found: list[Job] = []
                for url in urls:
                    response = client.get(url)
                    response.raise_for_status()
                    found.extend(parse(response.json()))
                jobs.extend(found)
                complete.add(name)
            except Exception as exc:  # one bad source must not stop the others
                errors[name] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        if own_client:
            client.close()
    return jobs, complete, errors
