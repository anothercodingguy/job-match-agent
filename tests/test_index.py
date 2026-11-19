from app.embed import HashEmbedder
from app.index import JobIndex
from app.schemas import Job

from .conftest import JOBS


def test_sync_adds_then_skips_unchanged(index):
    assert index.count() == len(JOBS)
    stats = index.sync(JOBS)
    assert stats["added"] == 0 and stats["updated"] == 0 and stats["unchanged"] == len(JOBS)


def test_sync_reembeds_only_changed_jobs(index):
    changed = [job.model_copy() for job in JOBS]
    changed[0] = changed[0].model_copy(update={"description": "Now a Django role with Celery workers."})
    stats = index.sync(changed)
    assert stats["updated"] == 1 and stats["unchanged"] == len(JOBS) - 1
    assert "Django" in index.get("t:py").description


def test_sync_removes_stale_jobs_only_for_complete_sources(index):
    remaining = [job for job in JOBS if job.id not in ("t:mkt", "u:devops")]
    # Source "u" failed to download this time, so only "t" is complete.
    stats = index.sync(remaining, complete_sources={"t"})
    assert stats["removed"] == 1
    assert index.get("t:mkt") is None
    assert index.get("u:devops") is not None  # kept: its source was not complete


def test_sync_ignores_duplicate_ids():
    idx = JobIndex(HashEmbedder())
    stats = idx.sync([JOBS[0], JOBS[0]])
    assert stats["added"] == 1 and idx.count() == 1


def test_search_ranks_the_relevant_job_first(index):
    hits = index.search("python fastapi backend apis", limit=3)
    assert hits[0][0].id == "t:py"
    assert hits[0][1] >= hits[-1][1]


def test_search_remote_only(index):
    hits = index.search("python engineer", limit=10, remote_only=True)
    assert hits and all(job.remote for job, _ in hits)


def test_search_location_keeps_matching_and_open_remote_jobs(index):
    ids = {job.id for job, _ in index.search("engineer developer analyst", limit=10, location="India")}
    assert "t:py" in ids and "u:devops" in ids  # located in India
    assert "t:ml" in ids and "t:data" in ids  # remote with no location restriction
    assert "t:react" not in ids and "t:mkt" not in ids  # on-site in Germany


def test_search_excludes_ids(index):
    hits = index.search("python fastapi backend", limit=3, exclude_ids={"t:py"})
    assert "t:py" not in {job.id for job, _ in hits}


def test_empty_query_returns_nothing(index):
    assert index.search("   ") == []


def test_content_hash_ignores_fields_that_are_not_embedded():
    job = JOBS[0]
    assert job.content_hash() == job.model_copy(update={"url": "https://example.com/x"}).content_hash()
    assert job.content_hash() != job.model_copy(update={"title": "Senior Python Engineer"}).content_hash()
    assert isinstance(Job.model_validate(job.model_dump()), Job)
