# Job Match Agent

Paste a resume, get jobs ranked for you from a live index of real listings, then refine the results by chatting with an agent that remembers what you told it.

## What it does

- **Structured profile from a resume.** The resume text goes to an LLM in JSON mode. The reply is validated with Pydantic; if it fails validation the errors are sent back once for a repair. The raw resume text is not stored.
- **Semantic search over a changing job index.** Listings from four public job APIs are embedded with `all-MiniLM-L6-v2` and stored in Qdrant. A sync step adds new jobs, re-embeds changed ones, skips unchanged ones (by content hash) and removes jobs that disappeared.
- **Two-stage ranking.** Vector search picks candidates from the whole index. The LLM then scores the top dozen against the profile and returns reasons and gaps for each.
- **An agent with tools and memory.** The chat model can call `search_jobs`, `get_job`, `save_preference` and `dismiss_job`. Preferences and dismissed jobs are saved in SQLite and applied to every later search in the session.
- **Works without the LLM.** If the model is down or no key is set, the profile falls back to keyword matching, ranking falls back to vector order, and chat falls back to a plain search.
- **A retrieval eval.** 20 cases on a frozen snapshot of real listings compare semantic search with a keyword baseline.

## How it works

```text
resume text ──> profile.py ──(JSON mode + validation)──> Profile ──> SQLite
                                                           │
job APIs ──> sources.py ──> index.py (embed + Qdrant sync) │
                                  │                        v
                                  └──> rank.py: vector search ──> LLM rerank ──> matches
                                  │
chat message ──> agent.py: LLM <──> tools (search, get job, save preference, dismiss)
                                          │
                                          └──> memory.py (SQLite: messages, preferences, feedback)
```

| File | Role |
|---|---|
| `app/llm.py` | Client for OpenAI-compatible chat APIs: JSON mode, tool calls, retries with backoff, model fallback |
| `app/profile.py` | Resume to `Profile`, with one repair attempt and a keyword fallback |
| `app/index.py` | Qdrant index: `sync()` and filtered `search()` |
| `app/sources.py` | Fetches and normalises listings from Arbeitnow, Remotive, Jobicy and Himalayas |
| `app/rank.py` | Retrieval, then LLM reranking with reasons and gaps |
| `app/agent.py` | Tool-calling loop, capped at 5 tool rounds per turn |
| `app/memory.py` | SQLite schema and queries: sessions, messages, preferences, feedback |
| `app/main.py` | FastAPI routes and the static UI |

## Run locally

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env            # add GROQ_API_KEY (optional: the app runs without it, in fallback mode)
.venv/bin/python -m app.ingest  # download current listings to data/jobs.json
.venv/bin/uvicorn app.main:app --port 8000
```

Open http://localhost:8000. The first start embeds every job, which takes about half a minute; later starts only embed what changed.

Any OpenAI-compatible endpoint can be used by setting `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODELS`. Set `QDRANT_URL` to use a Qdrant server instead of the local on-disk index.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/session` | Start a session |
| POST | `/api/profile` | `{session_id, resume_text}` to a structured profile |
| GET | `/api/matches?session_id=` | Ranked matches with scores, reasons and gaps |
| POST | `/api/chat` | `{session_id, message}` to a reply, the jobs shown and the tools called |
| POST | `/api/feedback` | Save or dismiss a job |
| POST | `/api/refresh` | Re-download the sources and sync the index (at most every 30 minutes) |
| GET | `/api/stats`, `/api/health` | Index size by source, last sync, whether an LLM is configured |

## Evaluation

```bash
.venv/bin/python -m evals.run   # writes evals/results.md
```

Each case describes a candidate in plain English with no job title, and a job counts as relevant if its title matches the case's regex. Those labels are rough, so the useful reading is the comparison between the two methods on the same 866 listings:

| Mean over 20 cases | Semantic (MiniLM) | Keyword baseline |
|---|---|---|
| Precision@5 | 0.54 | 0.17 |
| Recall@10 | 0.45 | 0.12 |
| MRR | 0.72 | 0.29 |

The eval also caught a bug: company names were part of the embedded text, so a query mentioning Figma returned sales jobs at Figma. Removing the company name raised precision@5 from 0.48 to 0.54. Per-case numbers are in [`evals/results.md`](evals/results.md). The LLM reranking stage is not covered by this eval yet.

## Tests

```bash
.venv/bin/python -m pytest -q   # 34 tests, no network or API key needed
```

The tests use a scripted fake LLM and a hashing embedder. They cover index sync, search filters, LLM retries and model fallback, profile repair and fallback, reranking (including ids the model invents), the agent loop, memory, the HTTP API and the source parsers.

## Limits

- Sessions have no login; a session id is a random token kept in the browser.
- The embedding model is English-only, and many Arbeitnow listings are in German.
- Location filtering is a substring match on the listing's location text.
- The eval labels come from title regexes, not human judgement.

## Data sources

Listings come from the public APIs of [Arbeitnow](https://www.arbeitnow.com/api/job-board-api), [Remotive](https://remotive.com/api/remote-jobs), [Jobicy](https://jobicy.com/jobs-rss-feed) and [Himalayas](https://himalayas.app/api). Every listing links back to its source. `evals/jobs_snapshot.json` is a frozen sample with shortened descriptions, used only for the eval.
