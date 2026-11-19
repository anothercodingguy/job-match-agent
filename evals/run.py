"""Retrieval eval: does semantic search find the right kind of job?

    python -m evals.run            # writes evals/results.md

For each case, a plain-English description of a candidate is used as the
search query, with no job title in it. A job counts as relevant if its TITLE
matches the case's regex. That is a rough, rule-based label (it misses
relevant jobs with unusual titles), so read the numbers as a comparison
between methods on the same data, not as absolute quality.

Two methods are compared on the same frozen snapshot of real listings:
  semantic  all-MiniLM-L6-v2 embeddings (what the app uses)
  keyword   hashed bag-of-words vectors (shared words only)
"""
import json
import re
from pathlib import Path

from app.embed import FastEmbedder, HashEmbedder
from app.index import JobIndex
from app.schemas import Job

HERE = Path(__file__).parent
MIN_RELEVANT = 3


def evaluate(index: JobIndex, cases: list[dict], relevant_ids: dict[str, set[str]]) -> dict[str, dict]:
    out = {}
    for case in cases:
        relevant = relevant_ids[case["name"]]
        ranked = [job.id for job, _ in index.search(case["query"], limit=10)]
        hits = [job_id in relevant for job_id in ranked]
        first = next((i for i, hit in enumerate(hits) if hit), None)
        out[case["name"]] = {
            "p5": sum(hits[:5]) / 5,
            "r10": sum(hits) / min(10, len(relevant)),
            "mrr": 0.0 if first is None else 1 / (first + 1),
        }
    return out


def load_snapshot() -> tuple[list[Job], list[dict], dict[str, set[str]], list[str]]:
    """The frozen listings, the usable cases, the relevant job ids per case, and the skipped cases."""
    jobs = [Job.model_validate(item) for item in json.loads((HERE / "jobs_snapshot.json").read_text())["jobs"]]
    all_cases = json.loads((HERE / "cases.json").read_text())
    relevant_ids, cases, skipped = {}, [], []
    for case in all_cases:
        pattern = re.compile(case["relevant_title"], re.IGNORECASE)
        ids = {job.id for job in jobs if pattern.search(job.title)}
        if len(ids) < MIN_RELEVANT:
            skipped.append(f"{case['name']} ({len(ids)} relevant jobs)")
            continue
        relevant_ids[case["name"]] = ids
        cases.append(case)
    return jobs, cases, relevant_ids, skipped


def main() -> None:
    jobs, cases, relevant_ids, skipped = load_snapshot()

    results = {}
    for label, embedder in (("semantic", FastEmbedder()), ("keyword", HashEmbedder(1024))):
        index = JobIndex(embedder)
        index.sync(jobs)
        results[label] = evaluate(index, cases, relevant_ids)

    def mean(label: str, metric: str) -> float:
        return sum(row[metric] for row in results[label].values()) / len(cases)

    lines = [
        "# Retrieval eval",
        "",
        f"{len(jobs)} real job listings (frozen snapshot), {len(cases)} cases. "
        "Labels are rule-based (title regex), so compare the two columns rather than reading them as absolute quality.",
        "",
        "| Metric (mean over cases) | Semantic (MiniLM) | Keyword baseline |",
        "|---|---|---|",
        f"| Precision@5 | {mean('semantic', 'p5'):.2f} | {mean('keyword', 'p5'):.2f} |",
        f"| Recall@10 | {mean('semantic', 'r10'):.2f} | {mean('keyword', 'r10'):.2f} |",
        f"| MRR | {mean('semantic', 'mrr'):.2f} | {mean('keyword', 'mrr'):.2f} |",
        "",
        "| Case | Relevant jobs | P@5 semantic | P@5 keyword |",
        "|---|---|---|---|",
    ]
    for case in cases:
        name = case["name"]
        lines.append(
            f"| {name} | {len(relevant_ids[name])} | {results['semantic'][name]['p5']:.1f} | {results['keyword'][name]['p5']:.1f} |"
        )
    if skipped:
        lines += ["", "Skipped (fewer than 3 relevant jobs in the snapshot): " + ", ".join(skipped) + "."]
    text = "\n".join(lines) + "\n"
    (HERE / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
