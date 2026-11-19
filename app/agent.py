"""The chat agent: an LLM in a loop with four tools.

Each turn the model sees the candidate profile, the saved preferences and the
recent conversation, including the ids of the jobs it showed last. It can
call tools to search the job index, read one job, save a preference, or
dismiss a job. The loop ends when the model answers in plain text. After
MAX_STEPS tool rounds it gets one last call with no tools and an instruction
to answer with what it has found.

Memory is what makes later turns better: preferences and dismissed jobs are
written to SQLite and applied to every later search in the session.
"""
import json

from .index import JobIndex
from .llm import LLM, LLMError
from .memory import Memory
from .schemas import PREFERENCE_KEYS, Job

MAX_STEPS = 5
# Asking the model to search less was not enough: with weak results it kept
# rewording the query. So the limit is enforced here, not only in the prompt.
MAX_SEARCHES = 2
MAX_REMEMBERED_JOBS = 12

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Semantic search over the live job index. Saved preferences are applied automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, e.g. 'junior python backend'"},
                    "remote_only": {"type": "boolean"},
                    "location": {"type": "string", "description": "City or country to restrict to"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job",
            "description": "Full details of one job by its id.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_preference",
            "description": "Remember a lasting preference the user stated, so later searches respect it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "enum": list(PREFERENCE_KEYS)},
                    "value": {
                        "type": "string",
                        "description": "remote_only: 'true'/'false'. avoid_keywords: comma-separated words.",
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dismiss_job",
            "description": "Hide a job the user is not interested in from all later results.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a job search assistant. You help one candidate find jobs from a live job index.
- Use search_jobs to find jobs. Never describe a job you have not retrieved with a tool.
- Put only the role and skills in the search query. Remote and location go in remote_only and location, not in the query text.
- You get at most two searches per message. The results are the closest jobs the index has, so rewording the query returns much the same jobs. Then answer with the best of them.
- When the user states lasting preferences (remote only, a location, things to avoid, a role focus), call save_preference once for each of them, before searching.
- When the user rejects a job, call dismiss_job with its id. Ids of the jobs shown last are listed below.
- Keep answers short. For each job you recommend, give the title, the company and one reason it fits this candidate.
- Write plain text. No Markdown: no asterisks, no headings.
- If the index has nothing suitable, say so plainly."""


# Sent with the last call. Leaving the tools out is not enough on its own: the
# model still tries to call one and the API rejects the reply (seen with Groq).
OUT_OF_STEPS = (
    "You have used all your tool calls for this message. Do not call any tool. "
    "Answer the user now, using only the jobs and results already in this conversation."
)
OUT_OF_STEPS_REPLY = "I ran out of steps before finishing. Here are the jobs I found so far."


class ToolBox:
    """Runs the tool calls for one chat turn and records which jobs were shown."""

    def __init__(self, memory: Memory, index: JobIndex, session_id: str):
        self.memory = memory
        self.index = index
        self.session_id = session_id
        self.shown: dict[str, Job] = {}
        self.searches = 0
        self.done: list[str] = []  # changes made this turn, in plain words

    def run(self, name: str, raw_arguments: str) -> dict:
        try:
            args = json.loads(raw_arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
            handler = getattr(self, f"_{name}", None)
            if handler is None:
                return {"error": f"unknown tool '{name}'"}
            return handler(**args)
        except (ValueError, TypeError) as exc:
            # Bad arguments go back to the model as an error it can correct.
            return {"error": str(exc)}

    def _search_jobs(self, query: str, remote_only: bool | None = None, location: str | None = None) -> dict:
        if self.searches >= MAX_SEARCHES:
            return {"error": "Search limit for this message reached. Answer now with the jobs already found."}
        self.searches += 1
        prefs = self.memory.get_preferences(self.session_id)
        hits = self.index.search(
            str(query),
            limit=6,
            remote_only=prefs.remote_only if remote_only is None else bool(remote_only),
            location=prefs.location if location is None else str(location),
            exclude_ids=self.memory.job_ids(self.session_id, "dismissed"),
        )
        avoid = [word.lower() for word in prefs.avoid_keywords]
        jobs = [
            job
            for job, _ in hits
            if not any(word in f"{job.title} {job.company} {' '.join(job.tags)}".lower() for word in avoid)
        ]
        for job in jobs:
            self.shown[job.id] = job
        # The model does not need the url: the UI gets it from self.shown.
        return {
            "jobs": [
                {
                    "id": job.id,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "remote": job.remote,
                    "tags": job.tags[:6],
                    "snippet": job.description[:200],
                }
                for job in jobs
            ]
        }

    def _get_job(self, job_id: str) -> dict:
        job = self.index.get(str(job_id))
        if job is None:
            return {"error": f"no job with id '{job_id}'"}
        self.shown[job.id] = job
        return job.model_dump()

    def _save_preference(self, key: str, value: str) -> dict:
        self.memory.set_preference(self.session_id, key, value)
        saved = getattr(self.memory.get_preferences(self.session_id), key)
        self.done.append(f"saved preference {key} = {json.dumps(saved)}")
        return {"saved": {key: saved}}

    def _dismiss_job(self, job_id: str) -> dict:
        job = self.index.get(str(job_id))
        if job is None:
            return {"error": f"no job with id '{job_id}'"}
        self.memory.set_feedback(self.session_id, job.id, "dismissed")
        self.shown.pop(job.id, None)
        self.done.append(f"hid {job.title} at {job.company}")
        return {"dismissed": job_id}


def _context(memory: Memory, session_id: str) -> str:
    profile = memory.get_profile(session_id)
    prefs = memory.get_preferences(session_id)
    dismissed = memory.job_ids(session_id, "dismissed")
    shown = [job for job in memory.last_shown_jobs(session_id) if job["id"] not in dismissed]
    return (
        "Candidate profile: "
        + (profile.model_dump_json(exclude={"source"}) if profile else "not provided yet")
        + "\nSaved preferences: "
        + prefs.model_dump_json()
        # Without this the model only has the text of earlier replies, and no ids to act on.
        + "\nJobs shown to the user last (id | title | company): "
        + ("; ".join(f"{job['id']} | {job['title']} | {job['company']}" for job in shown) or "none yet")
    )


def run_agent(llm: LLM | None, memory: Memory, index: JobIndex, session_id: str, user_message: str) -> dict:
    """One chat turn. Returns {"reply": str, "jobs": [job briefs], "steps": [tool names]}."""
    toolbox = ToolBox(memory, index, session_id)
    history = memory.recent_messages(session_id, limit=10)
    memory.add_message(session_id, "user", user_message)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + _context(memory, session_id)},
        *history,
        {"role": "user", "content": user_message},
    ]
    steps: list[str] = []
    reply = ""
    try:
        if llm is None:
            raise LLMError("no LLM configured")
        for _ in range(MAX_STEPS):
            message = llm.chat(messages, TOOLS)
            calls = message.get("tool_calls") or []
            if not calls:
                reply = (message.get("content") or "").strip()
                break
            messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
            for call in calls:
                name = call.get("function", {}).get("name", "")
                steps.append(name)
                result = toolbox.run(name, call.get("function", {}).get("arguments", ""))
                messages.append(
                    {"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(result)}
                )
        else:
            # Every round was a tool call. Ask once more for a written answer, with no tools.
            try:
                message = llm.chat([*messages, {"role": "system", "content": OUT_OF_STEPS}], None)
                reply = (message.get("content") or "").strip()
            except LLMError:
                reply = ""
        if not reply:
            reply = OUT_OF_STEPS_REPLY
    except LLMError:
        if steps:
            # The model failed partway. Its tool calls already took effect, so say that
            # instead of running a plain search on a message like "hide that one".
            reply = "The language model stopped responding partway through. What it had already done was kept" + (
                ", and these are the jobs it found." if toolbox.shown else "."
            )
        else:
            # No model at all: still do something useful with a plain search on the message.
            steps.append("search_jobs")
            found = toolbox.run("search_jobs", json.dumps({"query": user_message}))
            reply = (
                "The language model is unavailable right now, so these are plain search results for your message."
                if found.get("jobs")
                else "The language model is unavailable right now, and a plain search found nothing for that."
            )
    shown = list(toolbox.shown.values())
    # Tool calls are not kept in the history, only the text of each turn. So the stored
    # copy of the reply starts with what was changed. Without it a later turn can read
    # "hide that job" as still open and act on it again. The user sees the plain reply.
    note = f"(Done this turn: {'; '.join(toolbox.done)}.)\n" if toolbox.done else ""
    memory.add_message(
        session_id,
        "assistant",
        note + reply,
        jobs=[{"id": job.id, "title": job.title, "company": job.company} for job in shown[:MAX_REMEMBERED_JOBS]],
    )
    return {"reply": reply, "jobs": [job.brief() for job in shown], "steps": steps}
