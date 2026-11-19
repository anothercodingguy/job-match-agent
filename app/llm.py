"""A small client for OpenAI-compatible chat APIs (Groq by default).

Two calls are used in this project:
- json():  JSON mode, for structured outputs (profile extraction, reranking)
- chat():  normal chat with optional tool definitions, for the agent loop

Failure handling: each model gets a few attempts with exponential backoff on
rate limits, server errors and network errors. If a model keeps failing, or
rejects the request outright, the next model in the list is tried.

Rate limits (HTTP 429) usually come with a Retry-After header saying how many
seconds to wait. A short wait is honoured. A long one means this model will
not answer in time for a web request, so the next model is tried at once.
"""
import json
import time
from typing import Callable, Protocol

import httpx


class LLMError(Exception):
    pass


class LLM(Protocol):
    def json(self, messages: list[dict]) -> dict: ...

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict: ...


# Reasoning models spend output tokens on thinking before they write. With a provider's
# small default limit the answer can be cut off (seen on Groq: 1,924 of 2,048 tokens went
# to reasoning and only a few jobs got scored), so ask for enough room every time.
MAX_OUTPUT_TOKENS = 6000


class _Retryable(Exception):
    def __init__(self, message: str, wait: float | None = None):
        super().__init__(message)
        self.wait = wait  # seconds the server asked us to wait, if it said


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return max(0.0, float(response.headers.get("retry-after", "")))
    except ValueError:
        return None


class ChatLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        models: list[str],
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        backoff: float = 1.0,
        max_wait: float = 5.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not models:
            raise ValueError("at least one model is required")
        self.models = models
        self.max_retries = max_retries
        self.backoff = backoff
        self.max_wait = max_wait  # longest Retry-After worth sleeping through
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def _complete(self, payload: dict) -> dict:
        last_error: Exception | None = None
        for model in self.models:
            for attempt in range(self.max_retries + 1):
                try:
                    response = self._client.post("/chat/completions", json={**payload, "model": model})
                    if response.status_code == 429:
                        wait = _retry_after(response)
                        if wait is not None and wait > self.max_wait:
                            # Retrying sooner cannot work, and waiting is too slow: next model.
                            last_error = LLMError(f"{model}: rate limited, retry after {wait:.0f}s")
                            break
                        raise _Retryable(f"{model}: HTTP 429", wait)
                    if response.status_code >= 500:
                        raise _Retryable(f"{model}: HTTP {response.status_code}")
                    if response.status_code >= 400:
                        # Bad request or unknown model: retrying won't help, try the next model.
                        last_error = LLMError(f"{model}: HTTP {response.status_code}: {response.text[:200]}")
                        break
                    return response.json()["choices"][0]["message"]
                except (_Retryable, httpx.TransportError) as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        wait = getattr(exc, "wait", None)
                        self._sleep(wait if wait is not None else self.backoff * (2**attempt))
                except (KeyError, IndexError, ValueError) as exc:
                    last_error = LLMError(f"{model}: unexpected response shape: {exc}")
                    break
        raise LLMError(f"all models failed: {last_error}")

    def json(self, messages: list[dict]) -> dict:
        message = self._complete(
            {
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_completion_tokens": MAX_OUTPUT_TOKENS,
            }
        )
        try:
            data = json.loads(message.get("content") or "")
        except json.JSONDecodeError as exc:
            raise LLMError(f"model did not return valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMError("model returned JSON that is not an object")
        return data

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload: dict = {"messages": messages, "temperature": 0.2, "max_completion_tokens": MAX_OUTPUT_TOKENS}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return self._complete(payload)
