"""Download job listings and save them to a JSON file the server loads.

    python -m app.ingest                 # fetch all sources -> data/jobs.json
    python -m app.ingest --out some.json

The server syncs its index from this file at startup, and can also refresh
itself through POST /api/refresh.
"""
import argparse
import json
from pathlib import Path

from .config import settings
from .schemas import Job
from .sources import fetch_all


def save_jobs(path: str, jobs: list[Job], complete: set[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"complete_sources": sorted(complete), "jobs": [job.model_dump() for job in jobs]}, indent=1)
    )


def load_jobs(path: str) -> tuple[list[Job], set[str]]:
    data = json.loads(Path(path).read_text())
    return [Job.model_validate(item) for item in data["jobs"]], set(data.get("complete_sources", []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=settings.jobs_path)
    args = parser.parse_args()

    jobs, complete, errors = fetch_all()
    for source, error in errors.items():
        print(f"  {source}: FAILED ({error})")
    by_source: dict[str, int] = {}
    for job in jobs:
        by_source[job.source] = by_source.get(job.source, 0) + 1
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count} jobs")
    save_jobs(args.out, jobs, complete)
    print(f"saved {len(jobs)} jobs to {args.out}")


if __name__ == "__main__":
    main()
