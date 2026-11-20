from evals.agent import SCENARIOS, Turn, report, run

from .conftest import FakeLLM, tool_call


def test_runner_checks_what_the_agent_actually_did(index):
    scenarios = {
        "remote": [
            Turn(
                "remote only, python jobs",
                [
                    ("saves remote_only", lambda a: a.prefs.remote_only),
                    ("searched", lambda a: "search_jobs" in a.steps),
                    ("never true", lambda a: a.reply == "something else"),
                ],
                remember=lambda a: {"job": a.jobs[0]},
            ),
            Turn(
                lambda notes: f"hide {notes['job']['title']}",
                [("hides that job", lambda a: a.dismissed == {a.notes["job"]["id"]})],
            ),
        ]
    }
    llm = FakeLLM(chat_replies=[
        tool_call("save_preference", key="remote_only", value="true"),
        tool_call("search_jobs", call_id="c2", query="python"),
        {"role": "assistant", "content": "Here."},
        tool_call("dismiss_job", call_id="c3", job_id="t:ml"),
        {"role": "assistant", "content": "Hidden."},
    ])
    rows = run(llm, index, scenarios)
    assert [(row["turn"], row["check"], row["passed"]) for row in rows] == [
        (1, "saves remote_only", True),
        (1, "searched", True),
        (1, "never true", False),
        (2, "hides that job", True),  # t:ml is the first remote python job the search returned
    ]
    text = report(rows, "m", 6)
    assert "**3 of 4 checks passed (75%).**" in text and "**FAIL**" in text


def test_a_turn_that_depends_on_a_missing_job_fails_its_checks(index):
    scenarios = {"x": [Turn(lambda notes: notes["job"]["title"], [("a", lambda a: True), ("b", lambda a: True)])]}
    rows = run(FakeLLM(), index, scenarios)
    assert [row["passed"] for row in rows] == [False, False] and rows[0]["message"] == "(not asked)"


def test_every_scenario_has_checks():
    assert len(SCENARIOS) >= 8 and all(turn.checks for turns in SCENARIOS.values() for turn in turns)
