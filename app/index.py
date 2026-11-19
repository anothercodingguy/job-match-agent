"""Semantic job index on Qdrant.

The job list changes all the time, so the index is kept in step with sync():
new jobs are added, changed jobs are re-embedded, unchanged jobs are skipped
(compared by a hash of the embedded text), and jobs that disappeared from a
source are removed.
"""
import uuid
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from .embed import Embedder
from .schemas import Job

ANYWHERE = ("", "worldwide", "anywhere", "global", "remote")


def point_id(job_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, job_id))


class JobIndex:
    def __init__(self, embedder: Embedder, client: QdrantClient | None = None, collection: str = "jobs"):
        self.embedder = embedder
        self.client = client or QdrantClient(":memory:")
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
            )

    def _existing(self) -> dict[str, tuple[str, str]]:
        """point id -> (content hash, source) for everything already indexed."""
        found: dict[str, tuple[str, str]] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=500,
                offset=offset,
                with_payload=["content_hash", "source"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                found[str(point.id)] = (payload.get("content_hash", ""), payload.get("source", ""))
            if offset is None:
                return found

    def sync(self, jobs: list[Job], complete_sources: set[str] | None = None) -> dict:
        """Bring the index in line with `jobs`.

        Only sources listed in `complete_sources` have their missing jobs
        removed. A source that failed to download is left out of that set, so
        an outage does not wipe its jobs from the index.
        """
        existing = self._existing()
        stats = Counter(added=0, updated=0, unchanged=0, removed=0)
        seen: set[str] = set()
        to_embed: list[tuple[str, Job, str]] = []
        for job in jobs:
            pid = point_id(job.id)
            if pid in seen:
                continue
            seen.add(pid)
            digest = job.content_hash()
            if pid not in existing:
                stats["added"] += 1
            elif existing[pid][0] != digest:
                stats["updated"] += 1
            else:
                stats["unchanged"] += 1
                continue
            to_embed.append((pid, job, digest))

        for start in range(0, len(to_embed), 64):
            batch = to_embed[start : start + 64]
            vectors = self.embedder.embed([job.text() for _, job, _ in batch])
            self.client.upsert(
                self.collection,
                points=[
                    PointStruct(id=pid, vector=vector, payload={**job.model_dump(), "content_hash": digest})
                    for (pid, job, digest), vector in zip(batch, vectors)
                ],
            )

        stale = [
            pid
            for pid, (_, source) in existing.items()
            if pid not in seen and source in (complete_sources or set())
        ]
        if stale:
            self.client.delete(self.collection, points_selector=PointIdsList(points=stale))
            stats["removed"] = len(stale)
        return {**stats, "total": self.count()}

    def search(
        self,
        query: str,
        limit: int = 10,
        *,
        remote_only: bool = False,
        location: str = "",
        exclude_ids: set[str] | frozenset[str] = frozenset(),
    ) -> list[tuple[Job, float]]:
        if not query.strip():
            return []
        vector = self.embedder.embed([query])[0]
        query_filter = None
        if remote_only:
            query_filter = Filter(must=[FieldCondition(key="remote", match=MatchValue(value=True))])
        # Location and exclusions are applied after retrieval, so fetch extra candidates.
        fetch = limit * 5 if (location or exclude_ids) else limit
        points = self.client.query_points(
            self.collection, query=vector, limit=min(fetch, 200), query_filter=query_filter, with_payload=True
        ).points
        wanted = location.strip().lower()
        results: list[tuple[Job, float]] = []
        for point in points:
            job = Job.model_validate(point.payload)
            if job.id in exclude_ids:
                continue
            if wanted:
                where = job.location.lower()
                open_remote = job.remote and where.strip() in ANYWHERE
                if wanted not in where and not open_remote:
                    continue
            results.append((job, float(point.score)))
            if len(results) == limit:
                break
        return results

    def get(self, job_id: str) -> Job | None:
        points = self.client.retrieve(self.collection, ids=[point_id(job_id)], with_payload=True)
        return Job.model_validate(points[0].payload) if points else None

    def count(self) -> int:
        return self.client.count(self.collection).count

    def sources(self) -> dict[str, int]:
        return dict(Counter(source for _, source in self._existing().values()))
