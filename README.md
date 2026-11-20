# Job Match Agent

Paste a resume, get jobs ranked for you from a live index of real listings, then refine the results by chatting with an agent that remembers what you told it.

**Live:** https://job-match-agent-seven.vercel.app (add `?demo` to run it on a made-up sample resume)

## What it does

- **Structured profile from a resume.** The resume text goes to an LLM in JSON mode. The reply is validated with Pydantic; if it fails validation the errors are sent back once for a repair. The raw resume text is not stored.
- **Semantic search over a changing job index.** Listings from four public job APIs are embedded with `all-MiniLM-L6-v2` and stored in Qdrant. A sync step adds new jobs, re-embeds changed ones, skips unchanged ones (by content hash) and removes jobs that disappeared.
- **Two-stage ranking.** Vector search picks candidates from the whole index. The LLM then scores the top dozen against the profile and returns reasons and gaps for each.
- **An agent with tools and memory.** The chat model can call `search_jobs`, `get_job`, `save_preference` and `dismiss_job`. Preferences and dismissed jobs are saved in SQLite and applied to every later search in the session. Each turn the model is also given the ids of the jobs it showed last, so "hide the second one" needs no new search.
- **Limits the model cannot talk its way past.** At most 2 searches and 5 tool rounds per message, enforced in code. When the rounds run out, the model gets one last call with no tools and has to answer with what it found.
- **Works without the LLM.** If the model is down or no key is set, the profile falls back to keyword matching, ranking falls back to vector order, and chat falls back to a plain search. On a rate limit the client reads the server's `Retry-After`: a short wait is honoured, a long one sends the request to the next model at once.
- **Feedback that changes ranking.** Jobs you save pull the next search towards them and jobs you hide push it away (the query vector is shifted), and the reranker is told about both.
- **Memory in Postgres or SQLite.** The same SQL runs on both. With a `DATABASE_URL` the sessions, preferences and feedback survive restarts.
- **Four evals.** Retrieval, reranking, feedback and the agent's behaviour, each with numbers below.

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
| `app/llm.py` | Client for OpenAI-compatible chat APIs: JSON mode, tool calls, retries with backoff, `Retry-After`, model fallback |
| `app/profile.py` | Resume to `Profile`, with one repair attempt and a keyword fallback |
| `app/index.py` | Qdrant index: `sync()` and filtered `search()` |
| `app/embed.py` | Embedders: MiniLM through fastembed, a hashing baseline, and a wrapper that reuses vectors saved at build time |
| `app/prebuild.py` | Build step: download listings, embed them, save the vectors and the model files |
| `app/sources.py` | Fetches and normalises listings from Arbeitnow, Remotive, Jobicy and Himalayas |
| `app/rank.py` | Retrieval, then LLM reranking with reasons and gaps |
| `app/agent.py` | Tool-calling loop, capped at 5 tool rounds and 2 searches per turn |
| `app/memory.py` | Schema and queries on SQLite or Postgres: sessions, messages (with the jobs each reply showed), preferences, feedback |
| `app/main.py` | FastAPI routes and the static UI |

## Run locally

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env            # add GROQ_API_KEY (optional: the app runs without it, in fallback mode)
.venv/bin/python -m app.ingest  # download current listings to data/jobs.json
.venv/bin/uvicorn app.main:app --port 8000
```

Open http://localhost:8000. The first start embeds every job, which takes about half a minute; later starts only embed what changed.

`python -m app.prebuild` is optional locally. It does the download and the embedding ahead of time, so the app starts in about a second instead of half a minute.

Any OpenAI-compatible endpoint can be used by setting `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODELS`. Set `QDRANT_URL` to use a Qdrant server instead of the local on-disk index.

## Deployment (Vercel)

The app runs as one Vercel Function. Vercel's disk is read-only apart from `/tmp`, and instances start cold, so three things differ from a local run:

- **The index is built at deploy time.** [`vercel.json`](vercel.json) runs `python -m app.prebuild` as the build command. It fetches the listings, embeds them and saves the vectors and the model files into the bundle. At startup the index is filled from the saved vectors, with no model run. In a local simulation of these conditions startup took 1.0 s, against about 30 s when every job is embedded at startup.
- **The vector index is in memory**, rebuilt from those vectors on each cold start.
- **Memory needs a database outside the instance.** Set `DATABASE_URL` to a Postgres database (for example Neon, through Vercel's marketplace) and sessions survive restarts. Without it SQLite is used in `/tmp`, and a session lasts only as long as the instance that holds it.

Listings are as fresh as the last deploy, and the page shows how old they are. [`.github/workflows/refresh.yml`](.github/workflows/refresh.yml) asks Vercel for a new deploy once a day through a deploy hook, if the `VERCEL_DEPLOY_HOOK` repository secret is set. `POST /api/refresh` is turned off on Vercel, because it would only change one short-lived instance.

Set `GROQ_API_KEY` in the Vercel project's environment variables. Without it the app runs in fallback mode.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/session` | Start a session |
| POST | `/api/profile` | `{session_id, resume_text}` to a structured profile |
| GET | `/api/matches?session_id=` | Ranked matches with scores, reasons and gaps |
| POST | `/api/chat` | `{session_id, message}` to a reply, the tools called and the jobs found, with the ones the reply names marked and listed first |
| POST | `/api/feedback` | Save or dismiss a job |
| POST | `/api/refresh` | Re-download the sources and sync the index (at most every 30 minutes) |
| GET | `/api/stats`, `/api/health` | Index size by source, when the listings were fetched, which database holds the memory, whether an LLM is configured |

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

The eval also caught a bug: company names were part of the embedded text, so a query mentioning Figma returned sales jobs at Figma. Removing the company name raised precision@5 from 0.48 to 0.54. Per-case numbers are in [`evals/results.md`](evals/results.md).

A second eval checks the reranking stage. For each case it takes the 12 jobs vector search retrieves, lets the LLM reorder them with the app's own rerank code, and scores both orders with the same labels:

```bash
.venv/bin/python -m evals.rerank   # needs an LLM key; writes evals/rerank_results.md
```

| Mean over 20 cases | Vector order | After LLM rerank | Best possible order of the same 12 |
|---|---|---|---|
| Precision@5 | 0.54 | 0.73 | 0.79 |
| MRR | 0.72 | 0.91 | - |

Reranking raised precision@5 in 11 cases, lowered it in 1 (machine learning, 1.0 to 0.6) and left 8 unchanged. This is one run with `openai/gpt-oss-120b` on 2026-11-19. Two cautions: the labels are title regexes and the model reads titles, so the labels suit it; and reranking cannot fix retrieval, since it only reorders the 12 jobs it is given. Per-case numbers are in [`evals/rerank_results.md`](evals/rerank_results.md).

### The agent

```bash
.venv/bin/python -m evals.agent   # needs an LLM key; writes evals/agent_results.md
```

9 scripted conversations (14 turns) run against the real model on the frozen snapshot. After each turn, code checks what happened: which tools were called, what was saved, which jobs came back, what the reply says. No model judges another model.

| Run | Checks passed |
|---|---|
| First run | 37 of 39 (95%) |
| After one fix | 38 of 39 (97%) |

The first run found that "Remote jobs only, please." on its own got a reply but no `save_preference` call. One line in the prompt fixed it (3 of 3 replays, then the full run). The check that still fails is a reply that named none of the jobs it retrieved. Each line is a single run with `openai/gpt-oss-120b` on 2026-11-21; the model does not answer the same way every time. Every check and the tools called are in [`evals/agent_results.md`](evals/agent_results.md).

### Feedback

```bash
.venv/bin/python -m evals.feedback   # no LLM needed; writes evals/feedback_results.md
```

A simulated user looks at the first 5 results of each case, saves up to 2 relevant jobs and hides up to 2 irrelevant ones, and searches again.

| Mean over 20 cases | Search again, feedback ignored | Search shifted by feedback |
|---|---|---|
| Precision@5 on the next page | 0.41 | 0.52 |

Better in 9 cases, worse in 4, the same in 7. The shift uses the textbook Rocchio weights (0.75 towards saved, 0.15 away from hidden), not weights tuned on these cases. The user is simulated from the same title-regex labels, so this shows the mechanism works, not how real users behave.

### Things tried and not adopted

```bash
.venv/bin/python -m evals.variants   # writes evals/variants_results.md
```

| Retrieval method | Precision@5 | Recall@10 | MRR |
|---|---|---|---|
| all-MiniLM-L6-v2 (what the app uses) | 0.54 | 0.45 | 0.72 |
| bge-small-en-v1.5 | 0.48 | 0.42 | 0.62 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.56 | 0.42 | 0.79 |
| BM25 keyword search | 0.39 | 0.29 | 0.64 |
| Hybrid (MiniLM + BM25, reciprocal rank fusion), equal weight | 0.50 | 0.43 | 0.77 |
| Hybrid, vector counted twice | 0.54 | 0.46 | 0.81 |

Hybrid search finds the first relevant job sooner (MRR) but does not put more relevant jobs in the top 5, and the better hybrid setting was picked after seeing these results. With 20 cases a difference of 0.02 is two jobs. None of this was enough to justify a larger model or a second index, so retrieval stayed as it was.

## Tests

```bash
.venv/bin/python -m pytest -q   # 61 tests, no network or API key needed
```

The tests use a scripted fake LLM and a hashing embedder. They cover index sync, search filters, vectors reused from the build step, feedback-shifted search, LLM retries and model fallback, profile repair and fallback, reranking (including ids the model invents), the agent loop and its limits, memory, the HTTP API, the source parsers and the scoring code of the evals.

Set `TEST_DATABASE_URL` to a throwaway Postgres database and every memory test also runs there (77 tests; checked against a local Postgres 15).

## Checked against a real model

On 2026-11-19 the app was run against Groq (`openai/gpt-oss-120b`, with `openai/gpt-oss-20b` as the fallback) on the free tier:

- Profile extraction returned valid JSON that passed validation on the first call.
- Reranking returned a score, reasons and gaps for all 12 candidates and changed the order.
- A chat turn saved two preferences, searched, and answered with jobs it had retrieved. The next turn hid a job by id without searching for it again.
- The full flow was also run in a browser: resume, LLM-ranked matches, chat.

Running it against a real model found problems the scripted tests could not, each now fixed and covered by a test:

- The default models had been retired by Groq. The defaults were replaced.
- The model kept rewording its search until the round limit and never wrote an answer. Asking it in the prompt to search less did not work, so the limit is now in code.
- Leaving the tools out of the last call was not enough: the model still tried to call one and Groq rejected the request. The last call now also carries an instruction to answer.
- The client retried a rate-limited request after 1 and 2 seconds when the server had said to wait 12. It now reads `Retry-After`.
- The model could not hide a job it had shown in an earlier turn without searching for it again, because the history held only text. The ids of the jobs shown last are now part of its context.
- After hiding a job, a later turn sometimes tried to hide it again, because nothing in the history said it was done. The stored copy of each reply now starts with what the turn changed. Listing the hidden jobs in the system prompt was tried first and made it worse (4 of 4 replays went looking for the job, against 0 of 4 with the note).

- The fallback model's rerank was cut off: 1,924 of its default 2,048 output tokens went to reasoning, so only a few jobs got a score. Every call now asks for enough output room.

These checks are a handful of manual runs, not a measured success rate.

## Limits

- Groq's free tier allows 8,000 tokens per minute per model. A resume plus its rerank uses about 5,000 and a chat turn 2,000 to 8,000, so two quick requests in a row can exhaust both models. The app then falls back as described above.
- The jobs a chat reply recommends are found by matching titles and company names in the reply's text. A job the model describes in other words is listed with the other search results.
- Most listings are mid-level or senior roles in Europe and the US. A junior profile gets a weak pool, and the reranker shows it with low scores.
- Search always returns its 6 nearest jobs, even when none is a good match. Narrow filters (remote only, one city) on 841 listings often leave few good ones.

- Sessions have no login; a session id is a random token kept in the browser.
- The embedding model is English-only, and many Arbeitnow listings are in German.
- Location filtering is a substring match on the listing's location text.
- The eval labels come from title regexes, not human judgement.

## Data sources

Listings come from the public APIs of [Arbeitnow](https://www.arbeitnow.com/api/job-board-api), [Remotive](https://remotive.com/api/remote-jobs), [Jobicy](https://jobicy.com/jobs-rss-feed) and [Himalayas](https://himalayas.app/api). Every listing links back to its source. `evals/jobs_snapshot.json` is a frozen sample with shortened descriptions, used only for the eval.
