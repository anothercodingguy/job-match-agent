"""Feedback eval: do saved and hidden jobs improve the next search?

    python -m evals.feedback        # no LLM needed; writes evals/feedback_results.md

A simulated user runs each case's query and looks at the top 5. They save the
relevant jobs they see (up to 2) and hide the irrelevant ones (up to 2). The
query is then run again, leaving out every job they acted on, in two ways:

  without  the same search again (hidden and saved jobs only removed)
  with     the search shifted by that feedback (what the app does)

Both are scored on precision@5 over the jobs the user has not seen acted on.
Labels are the same title regexes as the other evals. Cases where the user
had nothing to save or nothing to hide still count: feedback is whatever the
first page allowed.
"""
from pathlib import Path

from app.embed import FastEmbedder
from app.index import JobIndex

from .run import load_snapshot

HERE = Path(__file__).parent


def simulate(index: JobIndex, query: str, relevant: set[str]) -> dict:
    first_page = [job.id for job, _ in index.search(query, limit=5)]
    saved = [job_id for job_id in first_page if job_id in relevant][:2]
    hidden = [job_id for job_id in first_page if job_id not in relevant][:2]
    acted = set(saved) | set(hidden)

    def precision(**feedback) -> float:
        hits = index.search(query, limit=5, exclude_ids=acted, **feedback)
        return sum(job.id in relevant for job, _ in hits) / 5

    return {
        "saved": len(saved),
        "hidden": len(hidden),
        "without": precision(),
        "with": precision(liked_ids=set(saved), disliked_ids=set(hidden)),
    }


def main() -> None:
    jobs, cases, relevant_ids, _ = load_snapshot()
    index = JobIndex(FastEmbedder())
    index.sync(jobs)
    rows = {case["name"]: simulate(index, case["query"], relevant_ids[case["name"]]) for case in cases}
    mean = lambda key: sum(row[key] for row in rows.values()) / len(rows)  # noqa: E731
    better = sum(row["with"] > row["without"] for row in rows.values())
    worse = sum(row["with"] < row["without"] for row in rows.values())
    lines = [
        "# Feedback eval",
        "",
        f"{len(jobs)} listings, {len(rows)} cases, simulated feedback on the first 5 results "
        "(save up to 2 relevant, hide up to 2 irrelevant), then the same query again.",
        "",
        "| Mean over cases | Search again without feedback | Search shifted by feedback |",
        "|---|---|---|",
        f"| Precision@5 on the next page | {mean('without'):.2f} | {mean('with'):.2f} |",
        "",
        f"Per case: better in {better}, worse in {worse}, the same in {len(rows) - better - worse}.",
        "",
        "| Case | Saved | Hidden | P@5 without | P@5 with |",
        "|---|---|---|---|---|",
        *[f"| {name} | {r['saved']} | {r['hidden']} | {r['without']:.1f} | {r['with']:.1f} |" for name, r in rows.items()],
    ]
    text = "\n".join(lines) + "\n"
    (HERE / "feedback_results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
