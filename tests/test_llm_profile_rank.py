import json

import httpx
import pytest

from app.llm import ChatLLM, LLMError
from app.profile import extract_profile, fallback_profile
from app.rank import build_query, match_jobs
from app.schemas import Preferences, Profile

from .conftest import FakeLLM

RESUME = "Software intern. Built REST APIs in Python with FastAPI and PostgreSQL. Used Docker and Redis."
GOOD_PROFILE = {
    "headline": "Backend intern",
    "years_experience": 0.7,
    "seniority": "intern",
    "skills": ["Python", "FastAPI", "PostgreSQL", "python"],
    "target_roles": ["Backend Engineer"],
    "locations": ["Bengaluru"],
    "remote_ok": True,
    "summary": "Builds APIs.",
}


def make_llm(handler, models=("big", "small")):
    slept = []
    llm = ChatLLM("https://llm.test/v1", "key", list(models), transport=httpx.MockTransport(handler), sleep=slept.append)
    return llm, slept


def ok(content):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


# --- ChatLLM: retries, backoff, model fallback ---

def test_llm_retries_with_backoff_then_succeeds():
    attempts = []

    def handler(request):
        attempts.append(json.loads(request.content)["model"])
        return httpx.Response(429) if len(attempts) < 3 else ok('{"a": 1}')

    llm, slept = make_llm(handler)
    assert llm.json([{"role": "user", "content": "hi"}]) == {"a": 1}
    assert attempts == ["big", "big", "big"]
    assert slept == [1.0, 2.0]  # exponential backoff


def test_llm_waits_as_long_as_a_short_retry_after_says():
    attempts = []

    def handler(request):
        attempts.append(json.loads(request.content)["model"])
        return httpx.Response(429, headers={"retry-after": "2"}) if len(attempts) < 2 else ok('{"a": 1}')

    llm, slept = make_llm(handler)
    assert llm.json([{"role": "user", "content": "hi"}]) == {"a": 1}
    assert attempts == ["big", "big"] and slept == [2.0]


def test_llm_moves_to_the_next_model_when_retry_after_is_long():
    attempts = []

    def handler(request):
        model = json.loads(request.content)["model"]
        attempts.append(model)
        return httpx.Response(429, headers={"retry-after": "23"}) if model == "big" else ok('{"from": "small"}')

    llm, slept = make_llm(handler)
    assert llm.json([{"role": "user", "content": "hi"}]) == {"from": "small"}
    assert attempts == ["big", "small"] and slept == []  # no pointless retries, no sleeping


def test_llm_asks_for_enough_output_room():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return ok("{}")

    llm, _ = make_llm(handler)
    llm.json([{"role": "user", "content": "hi"}])
    llm.chat([{"role": "user", "content": "hi"}])
    assert [payload["max_completion_tokens"] for payload in sent] == [6000, 6000]


def test_llm_falls_back_to_next_model():
    def handler(request):
        model = json.loads(request.content)["model"]
        return httpx.Response(503) if model == "big" else ok('{"from": "small"}')

    llm, _ = make_llm(handler)
    assert llm.json([{"role": "user", "content": "hi"}]) == {"from": "small"}


def test_llm_skips_a_rejected_model_without_retrying():
    seen = []

    def handler(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        return httpx.Response(404, text="model not found") if model == "big" else ok("{}")

    llm, slept = make_llm(handler)
    llm.json([{"role": "user", "content": "hi"}])
    assert seen == ["big", "small"] and slept == []


def test_llm_raises_when_everything_fails_or_json_is_invalid():
    llm, _ = make_llm(lambda request: httpx.Response(500))
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "hi"}])
    llm, _ = make_llm(lambda request: ok("this is not json"))
    with pytest.raises(LLMError):
        llm.json([{"role": "user", "content": "hi"}])


def test_llm_sends_json_mode_and_tools():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return ok("{}")

    llm, _ = make_llm(handler)
    llm.json([{"role": "user", "content": "x"}])
    llm.chat([{"role": "user", "content": "x"}], tools=[{"type": "function", "function": {"name": "t"}}])
    assert bodies[0]["response_format"] == {"type": "json_object"}
    assert bodies[1]["tool_choice"] == "auto" and bodies[1]["tools"][0]["function"]["name"] == "t"


# --- profile extraction ---

def test_profile_from_valid_json_dedupes_skills():
    profile = extract_profile(FakeLLM(json_replies=[GOOD_PROFILE]), RESUME)
    assert profile.source == "llm" and profile.seniority == "intern"
    assert profile.skills == ["Python", "FastAPI", "PostgreSQL"]


def test_profile_repairs_invalid_json_once():
    bad = {**GOOD_PROFILE, "seniority": "wizard", "years_experience": "a few"}
    llm = FakeLLM(json_replies=[bad, GOOD_PROFILE])
    profile = extract_profile(llm, RESUME)
    assert profile.source == "llm"
    assert len(llm.json_calls) == 2
    assert "failed validation" in llm.json_calls[1][-1]["content"]


def test_profile_falls_back_to_keywords():
    for llm in (None, FakeLLM(json_replies=[LLMError("down")]), FakeLLM(json_replies=[{"seniority": "x"}] * 2)):
        profile = extract_profile(llm, RESUME)
        assert profile.source == "fallback"
        assert {"Python", "FastAPI", "PostgreSQL", "Docker", "Redis"} <= set(profile.skills)


def test_fallback_does_not_match_inside_other_words():
    skills = fallback_profile("Going to the Javanese restaurant; I know JavaScript and C++.").skills
    assert "Java" not in skills and "Go" not in skills
    assert "JavaScript" in skills and "C++" in skills


# --- matching ---

PROFILE = Profile(headline="Backend intern", skills=["Python", "FastAPI", "PostgreSQL"], target_roles=["Backend Engineer"])


def test_build_query_uses_role_focus_and_skills():
    query = build_query(PROFILE, Preferences(role_focus="machine learning"))
    assert query.startswith("machine learning") and "FastAPI" in query


def test_match_uses_llm_order_and_reasons(index):
    def rerank(messages):
        jobs = json.loads(messages[-1]["content"])["jobs"]
        ml = next(job["id"] for job in jobs if "Machine Learning" in job["title"])
        py = next(job["id"] for job in jobs if "Python Backend" in job["title"])
        return {"ranking": [
            {"id": ml, "score": 91, "reasons": ["Uses Python"], "gaps": ["PyTorch"]},
            {"id": py, "score": 80, "reasons": ["FastAPI and PostgreSQL"], "gaps": []},
            {"id": "J99", "score": 100, "reasons": ["made up"]},
        ]}

    class Reranker(FakeLLM):
        def json(self, messages):
            return rerank(messages)

    matches = match_jobs(PROFILE, Preferences(), index, Reranker(), k=6)
    assert [m.job.id for m in matches[:2]] == ["t:ml", "t:py"]
    assert matches[0].ranked_by == "llm" and matches[0].gaps == ["PyTorch"]
    assert all(m.job.id.startswith(("t:", "u:")) for m in matches)  # the invented id was dropped
    assert {m.ranked_by for m in matches[2:]} == {"vector"}  # jobs the model skipped keep vector rank


def test_match_falls_back_to_vector_order(index):
    for llm in (None, FakeLLM(json_replies=[LLMError("down")]), FakeLLM(json_replies=[{"ranking": "nope"}])):
        matches = match_jobs(PROFILE, Preferences(), index, llm, k=3)
        assert matches[0].job.id == "t:py"
        assert all(m.ranked_by == "vector" for m in matches)


def test_match_applies_preferences_and_exclusions(index):
    prefs = Preferences(remote_only=True, avoid_keywords=["devops"])
    ids = [m.job.id for m in match_jobs(PROFILE, prefs, index, None, k=10, exclude_ids={"t:data"})]
    assert ids and set(ids) <= {"t:ml"}
