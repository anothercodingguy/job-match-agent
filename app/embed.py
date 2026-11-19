import hashlib
import json
import math
import re
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """all-MiniLM-L6-v2 through fastembed (ONNX, CPU). The model loads on first use."""

    dim = 384

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model_dir: str = ""):
        self.model_name = model_name
        # A folder that already holds the model files (made by app.prebuild). When it
        # exists nothing is downloaded, which matters on hosts with a read-only disk.
        self.model_dir = model_dir if model_dir and Path(model_dir).is_dir() else ""
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from fastembed import TextEmbedding

            extra = {"specific_model_path": self.model_dir} if self.model_dir else {}
            self._model = TextEmbedding(model_name=self.model_name, **extra)
        return [vector.tolist() for vector in self._model.embed(texts)]


def text_key(text: str) -> str:
    """Same value as Job.content_hash() when the text is Job.text()."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class CachedEmbedder:
    """Looks a text up in vectors computed earlier, and only embeds what is missing.

    The build step embeds every job once and saves the vectors. At startup the
    index is filled from that file, so the model only runs for search queries
    and for jobs that are new since the build.
    """

    def __init__(self, inner: Embedder, vectors: dict[str, list[float]]):
        self.inner = inner
        self.dim = inner.dim
        self.vectors = vectors
        self.hits = 0
        self.misses = 0

    @classmethod
    def from_file(cls, inner: Embedder, path: str) -> "CachedEmbedder":
        data = json.loads(Path(path).read_text())
        if data.get("dim") != inner.dim:
            return cls(inner, {})  # made with another model: ignore it
        return cls(inner, data["vectors"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [text_key(text) for text in texts]
        missing = [text for text, key in zip(texts, keys) if key not in self.vectors]
        fresh = dict(zip(missing, self.inner.embed(missing))) if missing else {}
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        return [self.vectors[key] if key in self.vectors else fresh[text] for text, key in zip(texts, keys)]


def save_vectors(path: str, embedder: Embedder, texts: list[str], model_name: str = "") -> int:
    """Embed the texts and write {hash of text: vector} for CachedEmbedder to load."""
    vectors = embedder.embed(texts)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            {
                "model": model_name,
                "dim": embedder.dim,
                "vectors": {text_key(t): [round(x, 6) for x in v] for t, v in zip(texts, vectors)},
            }
        )
    )
    return len(vectors)


class HashEmbedder:
    """Bag-of-words hashed into a fixed-size vector.

    No model download and fully deterministic. Used by the tests and as the
    keyword baseline in the evals. It matches shared words, not meaning.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9+#.]+", text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]
