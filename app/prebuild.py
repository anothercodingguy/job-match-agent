"""Build step: do the slow work once, before the app starts.

    python -m app.prebuild

1. Download current listings to data/jobs.json. If every source fails, fall
   back to the frozen eval snapshot so the app never starts empty.
2. Embed every job and save the vectors to data/vectors.json.
3. Copy the embedding model files to data/model.

A host with a read-only disk and cold starts (Vercel) runs this as its build
command. The app then starts in about a second: it loads the saved vectors
instead of embedding every job, and loads the model from disk instead of
downloading it.
"""
import json
import shutil
from pathlib import Path

from .config import ROOT, settings
from .embed import FastEmbedder, save_vectors
from .ingest import save_jobs
from .schemas import Job
from .sources import fetch_all

SNAPSHOT = ROOT / "evals" / "jobs_snapshot.json"


def copy_model(embedder: FastEmbedder, target: str) -> None:
    """Copy the downloaded model out of the download cache as plain files (no symlinks)."""
    embedder.embed(["warm up"])  # makes sure the model is downloaded and loaded
    source = Path(embedder._model.model._model_dir)
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target, symlinks=False)


def main() -> None:
    jobs, complete, errors = fetch_all()
    for source, error in errors.items():
        print(f"  {source}: FAILED ({error})")
    if not jobs:
        print("  no source answered; using the frozen snapshot instead")
        jobs = [Job.model_validate(item) for item in json.loads(SNAPSHOT.read_text())["jobs"]]
        complete = set()
    save_jobs(settings.jobs_path, jobs, complete)
    print(f"saved {len(jobs)} jobs to {settings.jobs_path}")

    embedder = FastEmbedder(settings.embed_model)
    texts = sorted({job.text() for job in jobs})
    count = save_vectors(settings.vectors_path, embedder, texts, settings.embed_model)
    print(f"saved {count} vectors to {settings.vectors_path}")

    copy_model(embedder, settings.model_dir)
    size = sum(f.stat().st_size for f in Path(settings.model_dir).rglob("*") if f.is_file()) / 1e6
    print(f"copied the model to {settings.model_dir} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
