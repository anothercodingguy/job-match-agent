import json

import pytest

from app.embed import HashEmbedder
from app.index import JobIndex
from app.llm import LLMError
from app.memory import Memory
from app.schemas import Job


class FakeLLM:
    """Returns scripted replies in order. A reply that is an Exception is raised instead."""

    def __init__(self, json_replies=(), chat_replies=()):
        self.json_replies = list(json_replies)
        self.chat_replies = list(chat_replies)
        self.json_calls: list[list[dict]] = []
        self.chat_calls: list[list[dict]] = []

    def json(self, messages):
        self.json_calls.append(list(messages))
        if not self.json_replies:
            raise LLMError("no scripted JSON reply left")
        reply = self.json_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def chat(self, messages, tools=None):
        self.chat_calls.append(list(messages))
        if not self.chat_replies:
            raise LLMError("no scripted chat reply left")
        reply = self.chat_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def tool_call(name: str, call_id: str = "call_1", **arguments) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


JOBS = [
    Job(id="t:py", source="t", title="Junior Python Backend Engineer", company="Acme", location="Bengaluru, India",
        remote=False, tags=["python", "fastapi", "postgresql"], description="Build REST APIs with FastAPI and PostgreSQL."),
    Job(id="t:react", source="t", title="React Frontend Developer", company="Pixel", location="Berlin, Germany",
        remote=False, tags=["react", "typescript", "css"], description="Build user interfaces with React and TypeScript."),
    Job(id="t:ml", source="t", title="Machine Learning Engineer", company="Deepfield", location="",
        remote=True, tags=["python", "pytorch", "llm"], description="Train and evaluate LLM systems with PyTorch and Python."),
    Job(id="t:mkt", source="t", title="Social Media Marketing Manager", company="Brandly", location="Munich, Germany",
        remote=False, tags=["marketing", "social media"], description="Plan campaigns and manage social media channels."),
    Job(id="t:data", source="t", title="Data Analyst", company="Numbers", location="Worldwide",
        remote=True, tags=["sql", "excel", "tableau"], description="Analyse data with SQL and build Tableau dashboards."),
    Job(id="u:devops", source="u", title="DevOps Engineer", company="Cloudy", location="Remote, India",
        remote=True, tags=["kubernetes", "docker", "aws"], description="Run Kubernetes clusters and CI/CD pipelines on AWS."),
]


@pytest.fixture
def index():
    idx = JobIndex(HashEmbedder())
    idx.sync(JOBS)
    return idx


@pytest.fixture
def memory():
    return Memory(":memory:")
