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


def test_old_database_without_jobs_column_is_upgraded(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, profile_json TEXT);"
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
        "role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);"
    )
    conn.close()
    store = Memory(path)
    sid = store.create_session()
    store.add_message(sid, "assistant", "one job", jobs=[{"id": "t:py", "title": "T", "company": "C"}])
    store.add_message(sid, "user", "thanks")
    assert store.last_shown_jobs(sid) == [{"id": "t:py", "title": "T", "company": "C"}]
    assert store.recent_messages(sid)[-1] == {"role": "user", "content": "thanks"}


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


def test_stored_reply_records_what_the_turn_changed(index, memory):
    sid = memory.create_session()
    result = run_agent(FakeLLM(chat_replies=[
        tool_call("save_preference", key="remote_only", value="true"),
        tool_call("dismiss_job", call_id="c2", job_id="t:py"),
        tool_call("dismiss_job", call_id="c3", job_id="t:nope"),  # fails, so it is not recorded
        {"role": "assistant", "content": "Done."},
    ]), memory, index, sid, "remote only, and hide the Acme job")
    assert result["reply"] == "Done."  # the user sees the plain reply
    # The next turn's history shows the request was already carried out.
    assert memory.recent_messages(sid)[-1]["content"] == (
        "(Done this turn: saved preference remote_only = true; hid Junior Python Backend Engineer at Acme.)\nDone."
    )
    # A turn that changes nothing is stored as it is.
    run_agent(FakeLLM(chat_replies=[{"role": "assistant", "content": "Hello."}]), memory, index, sid, "hi")
    assert memory.recent_messages(sid)[-1]["content"] == "Hello."


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


def test_agent_must_answer_without_tools_when_steps_run_out(index, memory):
    sid = memory.create_session()
    llm = FakeLLM(chat_replies=[tool_call("search_jobs", query="python")] * 5 + [
        {"role": "assistant", "content": "Best I found: the Acme backend role."},
    ])
    result = run_agent(llm, memory, index, sid, "keep searching")
    assert len(result["steps"]) == 5 and result["reply"].startswith("Best I found")
    # Five calls offered the tools. The sixth offered none and told the model to answer.
    assert all(llm.chat_tools[:5]) and llm.chat_tools[5] is None
    assert "Do not call any tool" in llm.chat_calls[5][-1]["content"]


def test_agent_says_so_when_the_forced_answer_also_fails(index, memory):
    sid = memory.create_session()
    llm = FakeLLM(chat_replies=[tool_call("search_jobs", query="python")] * 5 + [LLMError("tool_use_failed")])
    result = run_agent(llm, memory, index, sid, "keep searching")
    # Not the "model unavailable" fallback: the searches did run, so their jobs are returned.
    assert "ran out of steps" in result["reply"] and result["jobs"] and len(result["steps"]) == 5


def test_agent_is_told_the_ids_of_jobs_it_showed_last(index, memory):
    sid = memory.create_session()
    run_agent(FakeLLM(chat_replies=[
        tool_call("search_jobs", query="python fastapi backend"),
        {"role": "assistant", "content": "Try the Junior Python Backend Engineer role at Acme."},
    ]), memory, index, sid, "backend jobs please")
    assert "t:py" in {job["id"] for job in memory.last_shown_jobs(sid)}

    # Next turn: the id is in the model's context, so it can dismiss without searching again.
    llm = FakeLLM(chat_replies=[
        tool_call("dismiss_job", job_id="t:py"),
        {"role": "assistant", "content": "Hidden."},
    ])
    result = run_agent(llm, memory, index, sid, "hide the Acme one")
    system_prompt = llm.chat_calls[0][0]["content"]
    assert "t:py | Junior Python Backend Engineer | Acme" in system_prompt
    assert result["steps"] == ["dismiss_job"] and "t:py" in memory.job_ids(sid, "dismissed")

    # A dismissed job is no longer offered as something to act on.
    llm = FakeLLM(chat_replies=[{"role": "assistant", "content": "Anything else?"}])
    run_agent(llm, memory, index, sid, "thanks")
    assert "t:py |" not in llm.chat_calls[0][0]["content"]


def test_jobs_named_in_the_reply_come_first_and_are_marked(index, memory):
    sid = memory.create_session()
    llm = FakeLLM(chat_replies=[
        tool_call("search_jobs", query="python engineer data"),
        # Typographic hyphen and different case, as real models write them.
        {"role": "assistant", "content": "Best fit: data analyst \u2013 NUMBERS. It uses SQL."},
    ])
    result = run_agent(llm, memory, index, sid, "what fits me?")
    assert len(result["jobs"]) > 1
    assert result["jobs"][0]["id"] == "t:data" and result["jobs"][0]["recommended"] is True
    assert [job["recommended"] for job in result["jobs"][1:]] == [False] * (len(result["jobs"]) - 1)


def test_search_results_for_the_model_leave_out_urls(index, memory):
    sid = memory.create_session()
    llm = FakeLLM(chat_replies=[
        tool_call("search_jobs", query="python"),
        {"role": "assistant", "content": "ok"},
    ])
    result = run_agent(llm, memory, index, sid, "python jobs")
    tool_result = json.loads(llm.chat_calls[-1][-1]["content"])
    assert tool_result["jobs"] and all("url" not in job and "id" in job for job in tool_result["jobs"])
    assert all("url" in job for job in result["jobs"])  # the UI still gets links


def test_agent_search_limit_is_enforced_in_code(index, memory):
    sid = memory.create_session()
    llm = FakeLLM(chat_replies=[
        tool_call("search_jobs", query="python backend"),
        tool_call("search_jobs", call_id="c2", query="python backend developer"),
        tool_call("search_jobs", call_id="c3", query="python backend engineer"),
        {"role": "assistant", "content": "The Acme role is the closest."},
    ])
    result = run_agent(llm, memory, index, sid, "backend jobs")
    tool_results = [json.loads(m["content"]) for m in llm.chat_calls[-1] if m["role"] == "tool"]
    assert "jobs" in tool_results[0] and "jobs" in tool_results[1]
    assert "Search limit" in tool_results[2]["error"]  # the third search did not run
    assert result["reply"].startswith("The Acme role")


def test_agent_keeps_finished_tool_calls_when_the_model_fails_midway(index, memory):
    sid = memory.create_session()
    llm = FakeLLM(chat_replies=[tool_call("dismiss_job", job_id="t:py"), LLMError("rate limited")])
    result = run_agent(llm, memory, index, sid, "hide the Acme one")
    assert result["steps"] == ["dismiss_job"]  # no plain search on "hide the Acme one"
    assert "t:py" in memory.job_ids(sid, "dismissed")
    assert "partway" in result["reply"] and result["jobs"] == []


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
