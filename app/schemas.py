import hashlib
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _clean_list(value, limit: int) -> list[str]:
    """Accept a list (or a comma-separated string) and return unique, trimmed strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    seen, out = set(), []
    for item in value:
        text = str(item).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text[:80])
    return out[:limit]


class Job(BaseModel):
    id: str  # "<source>:<id on that source>"
    source: str
    title: str
    company: str = ""
    location: str = ""
    remote: bool = False
    tags: list[str] = []
    description: str = ""
    url: str = ""
    posted_at: str = ""  # YYYY-MM-DD when known

    def text(self) -> str:
        """The text that gets embedded for semantic search.

        The company name is left out on purpose: with it, a query that mentions
        a tool (say Figma) pulls in unrelated jobs at the company of that name.
        """
        parts = [self.title, self.location, "remote" if self.remote else ""]
        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags))
        parts.append(self.description[:800])
        return ". ".join(p for p in parts if p)

    def content_hash(self) -> str:
        """Changes only when the embedded text changes, so unchanged jobs are not re-embedded."""
        return hashlib.sha1(self.text().encode("utf-8")).hexdigest()

    def brief(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "remote": self.remote,
            "tags": self.tags[:6],
            "url": self.url,
        }


class Profile(BaseModel):
    headline: str = Field("", max_length=160)
    years_experience: float = Field(0, ge=0, le=60)
    seniority: Literal["intern", "junior", "mid", "senior", "lead"] = "junior"
    skills: list[str] = []
    target_roles: list[str] = []
    locations: list[str] = []
    remote_ok: bool = True
    summary: str = Field("", max_length=600)
    # "llm" when the model produced it, "fallback" when keyword matching did.
    source: Literal["llm", "fallback"] = "llm"

    @field_validator("skills", mode="before")
    @classmethod
    def _skills(cls, v):
        return _clean_list(v, 30)

    @field_validator("target_roles", "locations", mode="before")
    @classmethod
    def _short_lists(cls, v):
        return _clean_list(v, 6)


class Match(BaseModel):
    job: Job
    score: int = Field(ge=0, le=100)
    vector_score: float
    reasons: list[str] = []
    gaps: list[str] = []
    ranked_by: Literal["llm", "vector"]


class Preferences(BaseModel):
    """What the agent remembers about a session. Stored as key/value rows."""

    remote_only: bool = False
    location: str = ""
    avoid_keywords: list[str] = []
    role_focus: str = ""
    notes: str = ""

    @field_validator("avoid_keywords", mode="before")
    @classmethod
    def _avoid(cls, v):
        return _clean_list(v, 10)

    @field_validator("remote_only", mode="before")
    @classmethod
    def _bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "y"}
        return bool(v)


PREFERENCE_KEYS = tuple(Preferences.model_fields.keys())
