"""The chat agent: an LLM in a loop with four tools.

Each turn the model sees the candidate profile, the saved preferences and the
recent conversation. It can call tools to search the job index, read one job,
save a preference, or dismiss a job. The loop ends when the model answers in
plain text, or after MAX_STEPS tool rounds.

Memory is what makes later turns better: preferences and dismissed jobs are
written to SQLite and applied to every later search in the session.
"""
import json

from .index import JobIndex
from .llm import LLM, LLMError
from .memory import Memory
from .schemas import PREFERENCE_KEYS, Job

MAX_STEPS = 5

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
- When the user states a lasting preference (remote only, a location, things to avoid, a role focus), call save_preference.
- When the user rejects a job, call dismiss_job.
- Keep answers short. For each job you recommend, give the title, the company and one reason it fits this candidate.
- If the index has nothing suitable, say so plainly."""


class ToolBox:
    """Runs the tool calls for one chat turn and records which jobs were shown."""

    def __init__(self, memory: Memory, index: JobIndex, session_id: str):
        self.memory = memory
        self.index = index
        self.session_id = session_id
        self.shown: dict[str, Job] = {}

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
        return {"jobs": [{**job.brief(), "snippet": job.description[:200]} for job in jobs]}

    def _get_job(self, job_id: str) -> dict:
        job = self.index.get(str(job_id))
        if job is None:
            return {"error": f"no job with id '{job_id}'"}
        self.shown[job.id] = job
        return job.model_dump()

    def _save_preference(self, key: str, value: str) -> dict:
        self.memory.set_preference(self.session_id, key, value)
        return {"saved": {key: getattr(self.memory.get_preferences(self.session_id), key)}}

    def _dismiss_job(self, job_id: str) -> dict:
        if self.index.get(str(job_id)) is None:
            return {"error": f"no job with id '{job_id}'"}
        self.memory.set_feedback(self.session_id, str(job_id), "dismissed")
        self.shown.pop(str(job_id), None)
        return {"dismissed": job_id}


def _context(memory: Memory, session_id: str) -> str:
    profile = memory.get_profile(session_id)
    prefs = memory.get_preferences(session_id)
    return (
        "Candidate profile: "
        + (profile.model_dump_json(exclude={"source"}) if profile else "not provided yet")
        + "\nSaved preferences: "
        + prefs.model_dump_json()
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
        if not reply:
            reply = "I ran out of steps before finishing. Here are the jobs I found so far."
    except LLMError:
        # No model: still do something useful with a plain search on the message.
        steps.append("search_jobs")
        found = toolbox.run("search_jobs", json.dumps({"query": user_message}))
        reply = (
            "The language model is unavailable right now, so these are plain search results for your message."
            if found.get("jobs")
            else "The language model is unavailable right now, and a plain search found nothing for that."
        )
    memory.add_message(session_id, "assistant", reply)
    return {"reply": reply, "jobs": [job.brief() for job in toolbox.shown.values()], "steps": steps}
