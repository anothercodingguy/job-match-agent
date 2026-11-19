import json

import pytest
from fastapi.testclient import TestClient

from app.agent import run_agent
from app.llm import LLMError
from app.main import create_app
from app.memory import Memory
from app.schemas import Profile
from app.sources import clean_html, parse_arbeitnow, parse_himalayas, parse_jobicy, parse_remotive, to_date

from .conftest import FakeLLM, tool_call
from .test_llm_profile_rank import GOOD_PROFILE, RESUME


# --- memory ---

def test_preferences_upsert_and_types(memory):
    sid = memory.create_session()
    memory.set_preference(sid, "remote_only", "true")
    memory.set_preference(sid, "avoid_keywords", "sales, Marketing")
    memory.set_preference(sid, "location", "India")
    memory.set_preference(sid, "location", "Germany")  # overwrites
    prefs = memory.get_preferences(sid)
    assert prefs.remote_only is True and prefs.location == "Germany"
    assert prefs.avoid_keywords == ["sales", "Marketing"]
    with pytest.raises(ValueError):
        memory.set_preference(sid, "favourite_colour", "blue")


def test_messages_come_back_in_order_and_sessions_are_separate(memory):
    a, b = memory.create_session(), memory.create_session()
    for i in range(4):
        memory.add_message(a, "user" if i % 2 == 0 else "assistant", f"m{i}")
    memory.add_message(b, "user", "other")
    assert [m["content"] for m in memory.recent_messages(a, limit=3)] == ["m1", "m2", "m3"]
    assert [m["content"] for m in memory.recent_messages(b)] == ["other"]


def test_profile_round_trip_and_feedback(memory, tmp_path):
    sid = memory.create_session()
    assert memory.get_profile(sid) is None
    memory.save_profile(sid, Profile(skills=["Python"]))
    assert memory.get_profile(sid).skills == ["Python"]
    memory.set_feedback(sid, "t:py", "dismissed")
    memory.set_feedback(sid, "t:py", "saved")  # changing your mind replaces the row
    assert memory.job_ids(sid, "saved") == {"t:py"} and memory.job_ids(sid, "dismissed") == set()
    on_disk = Memory(str(tmp_path / "sub" / "app.db"))
    assert on_disk.session_exists(on_disk.create_session())


# --- agent loop ---

def test_agent_saves_preference_then_searches_with_it(index, memory):
    sid = memory.create_session()
    llm = FakeLLM(chat_replies=[
        tool_call("save_preference", key="remote_only", value="true"),
        tool_call("search_jobs", call_id="call_2", query="python engineer"),
        {"role": "assistant", "content": "Try the Machine Learning Engineer role at Deepfield."},
    ])
    result = run_agent(llm, memory, index, sid, "Only remote jobs please. Anything in Python?")
    assert result["steps"] == ["save_preference", "search_jobs"]
    assert memory.get_preferences(sid).remote_only is True
    assert result["jobs"] and all(job["remote"] for job in result["jobs"])
    assert "Deepfield" in result["reply"]
    # The tool results were fed back to the model, and the turn was stored.
    assert llm.chat_calls[-1][-1]["role"] == "tool"
    assert [m["role"] for m in memory.recent_messages(sid)] == ["user", "assistant"]


def test_agent_remembers_across_turns(index, memory):
    sid = memory.create_session()
    run_agent(FakeLLM(chat_replies=[
        tool_call("dismiss_job", job_id="t:py"),
        {"role": "assistant", "content": "Hidden."},
    ]), memory, index, sid, "Not interested in the Acme job")
    llm = FakeLLM(chat_replies=[
        tool_call("search_jobs", query="python fastapi backend"),
        {"role": "assistant", "content": "Here you go."},
    ])
    result = run_agent(llm, memory, index, sid, "show me backend jobs")
    assert "t:py" not in {job["id"] for job in result["jobs"]}
    system_and_history = llm.chat_calls[0]
    assert "Not interested in the Acme job" in [m["content"] for m in system_and_history]


def test_agent_returns_tool_errors_to_the_model(index, memory):
    sid = memory.create_session()
    llm = FakeLLM(chat_replies=[
        tool_call("save_preference", key="shoe_size", value="42"),
        tool_call("get_job", call_id="c2", job_id="t:nope"),
        tool_call("teleport", call_id="c3"),
        {"role": "assistant", "content": "Sorry, I could not do that."},
    ])
    result = run_agent(llm, memory, index, sid, "do odd things")
    tool_messages = [json.loads(m["content"]) for m in llm.chat_calls[-1] if m["role"] == "tool"]
    assert all("error" in message for message in tool_messages) and len(tool_messages) == 3
    assert result["reply"].startswith("Sorry")


def test_agent_stops_after_max_steps(index, memory):
    sid = memory.create_session()
    llm = FakeLLM(chat_replies=[tool_call("search_jobs", query="python")] * 10)
    result = run_agent(llm, memory, index, sid, "loop forever")
    assert len(result["steps"]) == 5 and "ran out of steps" in result["reply"]


def test_agent_without_llm_falls_back_to_plain_search(index, memory):
    sid = memory.create_session()
    for llm in (None, FakeLLM(chat_replies=[LLMError("down")])):
        result = run_agent(llm, memory, index, sid, "python fastapi backend")
        assert result["jobs"][0]["id"] == "t:py" and "unavailable" in result["reply"]


# --- HTTP API ---

@pytest.fixture
def client(index, memory):
    llm = FakeLLM(json_replies=[GOOD_PROFILE])
    with TestClient(create_app(llm=llm, index=index, memory=memory)) as test_client:
        yield test_client


def test_api_flow(client):
    assert client.get("/api/health").json()["jobs"] == 6
    sid = client.post("/api/session").json()["session_id"]
    assert client.get("/api/matches", params={"session_id": sid}).status_code == 400  # no profile yet
    profile = client.post("/api/profile", json={"session_id": sid, "resume_text": RESUME}).json()
    assert profile["skills"][:2] == ["Python", "FastAPI"]
    matches = client.get("/api/matches", params={"session_id": sid, "k": 3}).json()["matches"]
    assert matches[0]["job"]["id"] == "t:py" and matches[0]["ranked_by"] == "vector"  # rerank reply not scripted
    assert client.post("/api/feedback", json={"session_id": sid, "job_id": "t:py", "verdict": "dismissed"}).status_code == 200
    again = client.get("/api/matches", params={"session_id": sid}).json()["matches"]
    assert "t:py" not in {m["job"]["id"] for m in again}
    chat = client.post("/api/chat", json={"session_id": sid, "message": "data analyst sql"}).json()
    assert chat["jobs"] and "preferences" in chat


def test_api_rejects_bad_input(client):
    assert client.post("/api/profile", json={"session_id": "nope", "resume_text": RESUME}).status_code == 404
    sid = client.post("/api/session").json()["session_id"]
    assert client.post("/api/profile", json={"session_id": sid, "resume_text": "short"}).status_code == 422
    assert client.post("/api/feedback", json={"session_id": sid, "job_id": "t:py", "verdict": "love"}).status_code == 422
    assert client.post("/api/feedback", json={"session_id": sid, "job_id": "x", "verdict": "saved"}).status_code == 404
    assert client.get("/").status_code == 200


# --- source parsers ---

def test_parsers_normalise_each_source():
    a = parse_arbeitnow({"data": [{"slug": "dev-1", "title": "Dev", "company_name": "A", "location": "Berlin",
                                   "remote": True, "tags": ["x"], "job_types": ["full time"],
                                   "description": "<p>Hello&nbsp;<b>world</b></p>", "url": "u", "created_at": 1700000000},
                                  {"title": "no slug"}]})
    assert len(a) == 1 and a[0].id == "arbeitnow:dev-1" and a[0].description == "Hello world"
    assert a[0].posted_at == "2023-11-14" and a[0].tags == ["x", "full time"]
    r = parse_remotive({"jobs": [{"id": 7, "title": "T", "company_name": "C", "category": "Software",
                                  "candidate_required_location": "Worldwide", "tags": ["py"],
                                  "publication_date": "2026-11-01T10:00:00"}]})
    assert r[0].id == "remotive:7" and r[0].remote and r[0].posted_at == "2026-11-01"
    j = parse_jobicy({"jobs": [{"id": 9, "jobTitle": "R&amp;D Lead", "companyName": "J", "jobGeo": "USA",
                                "jobIndustry": ["Tech"], "jobType": ["Full-Time"], "jobExcerpt": "x", "pubDate": "2026-11-02 08:00:00"}]})
    assert j[0].title == "R&D Lead" and j[0].tags == ["Tech", "Full-Time"]
    h = parse_himalayas({"jobs": [{"guid": "g1", "title": "T", "companyName": "H", "locationRestrictions": ["India", "UK"],
                                   "categories": ["Data-Engineer"], "excerpt": "e", "applicationLink": "l", "pubDate": 1700000000}]})
    assert h[0].location == "India, UK" and h[0].tags == ["Data Engineer"]
    assert clean_html(None) == "" and to_date("garbage") == "" and to_date(None) == ""
