"""Agent eval: does the chat agent do the right thing, measured as a pass rate?

    python -m evals.agent          # needs an LLM key in .env; writes evals/agent_results.md

Each scenario is a short scripted conversation run against the real model, on
the frozen snapshot of listings and a fresh in-memory database. After every
turn a few checks look at what actually happened: which tools were called,
what was saved to memory, which jobs came back, what the reply says. A check
is code, so there is no judging by another model.

What the checks cover:
  memory      a stated preference is saved, and one that was not stated is not
  tools       the right tool for the request (search, get_job, dismiss_job)
  results     later searches respect what was saved or hidden
  answers     a written reply that names a job it retrieved, in plain text

One run is one sample. The model does not answer the same way every time, so
expect the rate to move by a few checks between runs.
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.agent import OUT_OF_STEPS_REPLY, run_agent
from app.config import settings
from app.embed import FastEmbedder
from app.index import JobIndex
from app.llm import LLM, ChatLLM
from app.memory import Memory
from app.schemas import Preferences

from .run import load_snapshot

HERE = Path(__file__).parent
DEFAULTS = Preferences()


@dataclass
class After:
    """What a check can look at after a turn."""

    reply: str
    steps: list[str]
    jobs: list[dict]
    prefs: Preferences
    dismissed: set[str]
    notes: dict = field(default_factory=dict)  # carried from turn to turn within a scenario


Check = tuple[str, Callable[[After], bool]]


@dataclass
class Turn:
    message: str | Callable[[dict], str]
    checks: list[Check]
    remember: Callable[[After], dict] | None = None  # what to keep for later turns


def _text(job: dict) -> str:
    return f"{job['title']} {job['company']} {' '.join(job['tags'])}".lower()


# Checks used in more than one place.
WROTE_AN_ANSWER: Check = (
    "writes an answer",
    lambda a: bool(a.reply) and a.reply != OUT_OF_STEPS_REPLY and not a.reply.startswith("The language model"),
)
PLAIN_TEXT: Check = ("answers in plain text", lambda a: "**" not in a.reply and "\n#" not in "\n" + a.reply)
SEARCHED: Check = ("searches the index", lambda a: "search_jobs" in a.steps)
NAMES_A_RETRIEVED_JOB: Check = ("names a job it retrieved", lambda a: any(job["recommended"] for job in a.jobs))
SAVED_NOTHING: Check = ("saves no preference that was not stated", lambda a: a.prefs == DEFAULTS)
FIRST_JOB = lambda a: {"job": a.jobs[0]} if a.jobs else {}  # noqa: E731

SCENARIOS: dict[str, list[Turn]] = {
    "remote only": [
        Turn(
            "I only want remote jobs. Find me data analyst roles.",
            [
                ("saves remote_only", lambda a: a.prefs.remote_only),
                SEARCHED,
                ("every job shown is remote", lambda a: bool(a.jobs) and all(job["remote"] for job in a.jobs)),
                WROTE_AN_ANSWER,
                NAMES_A_RETRIEVED_JOB,
                PLAIN_TEXT,
            ],
        ),
        Turn(
            "What about marketing roles?",
            [
                ("still only remote jobs, a turn later", lambda a: bool(a.jobs) and all(job["remote"] for job in a.jobs)),
                WROTE_AN_ANSWER,
            ],
        ),
    ],
    "avoid a keyword": [
        Turn(
            "Nothing in sales please. Which account or customer roles do you have?",
            [
                ("saves 'sales' as something to avoid", lambda a: "sales" in [w.lower() for w in a.prefs.avoid_keywords]),
                ("no job shown mentions sales", lambda a: bool(a.jobs) and not any("sales" in _text(job) for job in a.jobs)),
                WROTE_AN_ANSWER,
            ],
        )
    ],
    "two preferences at once": [
        Turn(
            "Remote only, and avoid anything about crypto. Show me product manager roles.",
            [
                ("saves remote_only", lambda a: a.prefs.remote_only),
                ("saves 'crypto' as something to avoid", lambda a: any("crypto" in w.lower() for w in a.prefs.avoid_keywords)),
                SEARCHED,
                NAMES_A_RETRIEVED_JOB,
            ],
        )
    ],
    "location": [
        Turn(
            "I can only work in Germany. Find software developer jobs.",
            [
                ("saves the location", lambda a: "german" in a.prefs.location.lower() or "deutsch" in a.prefs.location.lower()),
                SEARCHED,
                WROTE_AN_ANSWER,
            ],
        )
    ],
    "a preference is changed": [
        Turn("Remote jobs only, please.", [("saves remote_only", lambda a: a.prefs.remote_only)]),
        Turn(
            "Actually, on-site is fine too. Forget the remote-only rule.",
            [("turns remote_only off again", lambda a: not a.prefs.remote_only)],
        ),
    ],
    "no preference stated": [
        Turn(
            "What frontend developer jobs do you have?",
            [SAVED_NOTHING, SEARCHED, WROTE_AN_ANSWER, NAMES_A_RETRIEVED_JOB, PLAIN_TEXT],
        )
    ],
    "hide a job by name": [
        Turn("Find devops jobs.", [SEARCHED, NAMES_A_RETRIEVED_JOB], remember=FIRST_JOB),
        Turn(
            lambda notes: f"Not interested in the {notes['job']['title']} one at {notes['job']['company']}. Hide it.",
            [
                ("hides exactly that job", lambda a: a.dismissed == {a.notes["job"]["id"]}),
                ("hides it without searching for it first", lambda a: a.steps[:1] == ["dismiss_job"]),
            ],
        ),
        Turn(
            "Search for devops jobs again.",
            [
                ("the hidden job does not come back", lambda a: a.notes["job"]["id"] not in [job["id"] for job in a.jobs]),
                ("hides nothing else", lambda a: a.dismissed == {a.notes["job"]["id"]}),
                WROTE_AN_ANSWER,
            ],
        ),
    ],
    "details of one job": [
        Turn("Find security jobs.", [SEARCHED], remember=FIRST_JOB),
        Turn(
            lambda notes: f"Tell me more about the {notes['job']['title']} job at {notes['job']['company']}.",
            [
                ("reads that job with get_job", lambda a: "get_job" in a.steps),
                ("does not search again for it", lambda a: "search_jobs" not in a.steps),
                WROTE_AN_ANSWER,
            ],
        ),
    ],
    "off topic": [
        Turn(
            "What is the capital of France?",
            [SAVED_NOTHING, ("hides nothing", lambda a: not a.dismissed), ("replies", lambda a: bool(a.reply))],
        )
    ],
}


def run(llm: LLM, index: JobIndex, scenarios: dict[str, list[Turn]] = SCENARIOS) -> list[dict]:
    """One row per check: {scenario, turn, check, passed, message, steps}."""
    rows = []
    for name, turns in scenarios.items():
        memory = Memory(":memory:")
        session = memory.create_session()
        notes: dict = {}
        for number, turn in enumerate(turns, start=1):
            try:
                message = turn.message(notes) if callable(turn.message) else turn.message
            except KeyError:
                # An earlier turn returned no job to refer to, so this turn cannot be asked.
                rows += [dict(scenario=name, turn=number, check=c, passed=False, message="(not asked)", steps=[]) for c, _ in turn.checks]
                continue
            result = run_agent(llm, memory, index, session, message)
            after = After(
                reply=result["reply"],
                steps=result["steps"],
                jobs=result["jobs"],
                prefs=memory.get_preferences(session),
                dismissed=memory.job_ids(session, "dismissed"),
                notes=notes,
            )
            if turn.remember:
                notes.update(turn.remember(after))
            for check, passes in turn.checks:
                try:
                    passed = bool(passes(after))
                except (KeyError, IndexError):
                    passed = False
                rows.append(dict(scenario=name, turn=number, check=check, passed=passed, message=message, steps=result["steps"]))
    return rows


def report(rows: list[dict], model: str, jobs: int) -> str:
    passed = sum(row["passed"] for row in rows)
    turns = len({(row["scenario"], row["turn"]) for row in rows})
    lines = [
        "# Agent eval",
        "",
        f"{len({row['scenario'] for row in rows})} scripted conversations, {turns} turns, {len(rows)} checks, "
        f"model `{model}`, {jobs} listings (frozen snapshot). One run.",
        "",
        f"**{passed} of {len(rows)} checks passed ({passed / len(rows):.0%}).**",
        "",
        "| Conversation | Turn | Check | Result | Tools called |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['turn']} | {row['check']} | {'pass' if row['passed'] else '**FAIL**'} "
            f"| {' → '.join(row['steps']) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=settings.models[0], help="one model, so the result is not a mix")
    parser.add_argument("--only", default="", help="run one conversation by name (for a quick check)")
    args = parser.parse_args()
    if not settings.llm_api_key:
        raise SystemExit("No LLM key found. Put GROQ_API_KEY in .env first.")
    jobs, _, _, _ = load_snapshot()
    index = JobIndex(FastEmbedder())
    index.sync(jobs)
    # One model and patient retries: a rate limit should be waited out, not turned into a fallback.
    llm = ChatLLM(
        settings.llm_base_url, settings.llm_api_key, [args.model], timeout=60, max_retries=6, backoff=5.0, max_wait=90
    )
    scenarios = {args.only: SCENARIOS[args.only]} if args.only else SCENARIOS
    text = report(run(llm, index, scenarios), args.model, len(jobs))
    if not args.only:
        (HERE / "agent_results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
