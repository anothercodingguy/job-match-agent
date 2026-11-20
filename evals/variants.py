"""Retrieval experiments: would another retrieval method beat what the app uses?

    python -m evals.variants        # writes evals/variants_results.md (takes a few minutes)

Same snapshot, cases and title-regex labels as evals/run.py. Compared:
  - the app's embedding model (all-MiniLM-L6-v2)
  - two other small embedding models
  - BM25 keyword search
  - hybrid: BM25 and vector results merged with reciprocal rank fusion (RRF)

None of these is used by the app. The file exists so the decision not to
switch is backed by numbers anyone can reproduce.
"""
import collections
import math
import re
from pathlib import Path

from app.index import JobIndex
from app.schemas import Job

from .run import load_snapshot

HERE = Path(__file__).parent
STOP = set("a an the and or of to in for with on at by is are as be this that you we our your will from it".split())
MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9+#.]+", text.lower()) if t not in STOP]


class BM25:
    """Okapi BM25 over title, tags and the start of the description."""

    def __init__(self, jobs: list[Job], k1: float = 1.4, b: float = 0.75):
        self.ids = [job.id for job in jobs]
        self.docs = [tokens(f"{job.title} {' '.join(job.tags)} {job.description[:800]}") for job in jobs]
        self.counts = [collections.Counter(doc) for doc in self.docs]
        self.df = collections.Counter(term for doc in self.docs for term in set(doc))
        self.average = sum(map(len, self.docs)) / len(self.docs)
        self.k1, self.b = k1, b

    def search(self, query: str, limit: int) -> list[str]:
        n = len(self.docs)
        scored = []
        for i, counts in enumerate(self.counts):
            score = 0.0
            for term in tokens(query):
                tf = counts.get(term, 0)
                if tf:
                    idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    norm = tf + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.average)
                    score += idf * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self.ids[i] for _, i in scored[:limit]]


def rrf(rankings: list[tuple[list[str], float]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion: each list adds weight / (k + rank) to a job's score."""
    scores: dict[str, float] = collections.defaultdict(float)
    for ranked, weight in rankings:
        for rank, job_id in enumerate(ranked, start=1):
            scores[job_id] += weight / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def score(rank_fn, cases: list[dict], relevant: dict[str, set[str]]) -> tuple[float, float, float]:
    p5 = r10 = mrr = 0.0
    for case in cases:
        wanted = relevant[case["name"]]
        hits = [job_id in wanted for job_id in rank_fn(case["query"])[:10]]
        first = next((i for i, hit in enumerate(hits) if hit), None)
        p5 += sum(hits[:5]) / 5
        r10 += sum(hits) / min(10, len(wanted))
        mrr += 0.0 if first is None else 1 / (first + 1)
    return p5 / len(cases), r10 / len(cases), mrr / len(cases)


class _Model:
    def __init__(self, name: str):
        from fastembed import TextEmbedding

        self.dim = next(m["dim"] for m in TextEmbedding.list_supported_models() if m["model"] == name)
        self._model = TextEmbedding(name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]


def main() -> None:
    jobs, cases, relevant, _ = load_snapshot()
    rows: list[tuple[str, tuple[float, float, float]]] = []
    app_index = None
    for name in MODELS:
        index = JobIndex(_Model(name))
        index.sync(jobs)
        app_index = app_index or index  # the first model is the app's
        label = name.split("/")[-1] + (" (what the app uses)" if name == MODELS[0] else "")
        rows.append((label, score(lambda q: [job.id for job, _ in index.search(q, limit=10)], cases, relevant)))

    bm25 = BM25(jobs)
    vector = lambda q: [job.id for job, _ in app_index.search(q, limit=50)]  # noqa: E731
    rows.append(("BM25 keyword search", score(lambda q: bm25.search(q, 50), cases, relevant)))
    rows.append(("Hybrid: MiniLM + BM25, RRF, equal weight", score(lambda q: rrf([(vector(q), 1), (bm25.search(q, 50), 1)]), cases, relevant)))
    rows.append(("Hybrid: MiniLM + BM25, RRF, vector counted twice", score(lambda q: rrf([(vector(q), 2), (bm25.search(q, 50), 1)]), cases, relevant)))

    lines = [
        "# Retrieval experiments",
        "",
        f"{len(jobs)} listings, {len(cases)} cases, title-regex labels (see evals/run.py). "
        "One relevant job more or less in a case's top 5 moves mean precision@5 by 0.01, "
        "so differences of a few hundredths are noise.",
        "",
        "| Method | Precision@5 | Recall@10 | MRR |",
        "|---|---|---|---|",
        *[f"| {label} | {p5:.2f} | {r10:.2f} | {mrr:.2f} |" for label, (p5, r10, mrr) in rows],
    ]
    text = "\n".join(lines) + "\n"
    (HERE / "variants_results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
