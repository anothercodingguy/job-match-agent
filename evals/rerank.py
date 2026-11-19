"""Rerank eval: does the LLM reranking stage improve on plain vector order?

    python -m evals.rerank         # needs an LLM key in .env; writes evals/rerank_results.md

For each case, vector search retrieves a pool of jobs (12, the pool size the
app uses). The LLM then reorders that pool with the same prompt and code the
app uses (app.rank.llm_rerank). Both orders are scored against the same
title-regex labels as the retrieval eval.

How to read it:
- Reranking only reorders the pool, so it cannot do better than the "ceiling"
  column: the score of a perfect ordering of the same 12 jobs.
- The labels are title regexes. The model also reads descriptions, so it can
  be marked wrong for a sensible choice. Treat small differences as noise.
- A case where the LLM call fails is left out of all columns and counted.
"""
import argparse
import time
from pathlib import Path

from app.config import settings
from app.embed import FastEmbedder
from app.index import JobIndex
from app.llm import LLM, ChatLLM, LLMError
from app.rank import build_query, llm_rerank
from app.schemas import Preferences, Profile

from .run import load_snapshot

HERE = Path(__file__).parent
POOL = 12  # same as the default in app.rank.match_jobs
K = 5


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(job_id in relevant for job_id in ranked[:k]) / k


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for position, job_id in enumerate(ranked, start=1):
        if job_id in relevant:
            return 1 / position
    return 0.0


def compare(
    index: JobIndex,
    cases: list[dict],
    relevant_ids: dict[str, set[str]],
    llm: LLM,
    *,
    pool: int = POOL,
    k: int = K,
    pause: float = 0.0,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Returns (rows per case, errors per failed case)."""
    rows: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for number, case in enumerate(cases):
        name = case["name"]
        relevant = relevant_ids[name]
        # The case text is the only thing the model knows about the candidate: no job title.
        profile = Profile(summary=case["query"])
        prefs = Preferences()
        hits = index.search(build_query(profile, prefs), limit=pool)
        vector_order = [job.id for job, _ in hits]
        if pause and number:
            time.sleep(pause)
        try:
            matches = llm_rerank(llm, profile, prefs, hits)
        except LLMError as exc:
            errors[name] = str(exc)[:200]
            continue
        llm_order = [match.job.id for match in matches]
        in_pool = sum(job_id in relevant for job_id in vector_order)
        rows[name] = {
            "relevant_in_pool": in_pool,
            "p_vector": precision_at_k(vector_order, relevant, k),
            "p_llm": precision_at_k(llm_order, relevant, k),
            "p_ceiling": min(k, in_pool) / k,
            "mrr_vector": reciprocal_rank(vector_order, relevant),
            "mrr_llm": reciprocal_rank(llm_order, relevant),
            "ranked_by_llm": sum(match.ranked_by == "llm" for match in matches),
        }
    return rows, errors


def report(rows: dict[str, dict], errors: dict[str, str], *, model: str, jobs: int, pool: int, k: int) -> str:
    def mean(key: str) -> float:
        return sum(row[key] for row in rows.values()) / len(rows)

    better = sum(row["p_llm"] > row["p_vector"] for row in rows.values())
    worse = sum(row["p_llm"] < row["p_vector"] for row in rows.values())
    lines = [
        "# Rerank eval",
        "",
        f"{jobs} real job listings (frozen snapshot), {len(rows)} cases scored, model `{model}`. "
        f"Vector search retrieves {pool} jobs per case; the LLM reorders them. "
        "Labels are rule-based (title regex), so small differences are noise.",
        "",
        "| Metric (mean over cases) | Vector order | After LLM rerank | Ceiling for this pool |",
        "|---|---|---|---|",
        f"| Precision@{k} | {mean('p_vector'):.2f} | {mean('p_llm'):.2f} | {mean('p_ceiling'):.2f} |",
        f"| MRR | {mean('mrr_vector'):.2f} | {mean('mrr_llm'):.2f} | - |",
        "",
        f"Precision@{k} per case: rerank better in {better}, worse in {worse}, "
        f"the same in {len(rows) - better - worse}.",
        "",
        f"| Case | Relevant in pool | P@{k} vector | P@{k} LLM | Ceiling | Jobs the LLM ranked |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['relevant_in_pool']} | {row['p_vector']:.1f} | {row['p_llm']:.1f} "
            f"| {row['p_ceiling']:.1f} | {row['ranked_by_llm']}/{pool} |"
        )
    if errors:
        lines += ["", f"Left out because the LLM call failed ({len(errors)}): " + ", ".join(errors) + "."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=settings.models[0], help="one model, so the numbers are not a mix")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to wait between cases (rate limits)")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N cases (for a quick check)")
    args = parser.parse_args()
    if not settings.llm_api_key:
        raise SystemExit("No LLM key found. Put GROQ_API_KEY in .env first.")

    jobs, cases, relevant_ids, _ = load_snapshot()
    if args.limit:
        cases = cases[: args.limit]
    index = JobIndex(FastEmbedder())
    index.sync(jobs)
    # One model and patient retries: an eval should wait out a rate limit, not fall back.
    llm = ChatLLM(
        settings.llm_base_url, settings.llm_api_key, [args.model], timeout=60, max_retries=6, backoff=5.0, max_wait=90
    )
    rows, errors = compare(index, cases, relevant_ids, llm, pause=args.pause)
    for name, error in errors.items():
        print(f"FAILED {name}: {error}")
    if not rows:
        raise SystemExit("Every LLM call failed, so there is nothing to report.")
    text = report(rows, errors, model=args.model, jobs=len(jobs), pool=POOL, k=K)
    if not args.limit:
        (HERE / "rerank_results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
