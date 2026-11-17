"""A small vector database written from scratch with NumPy.

No Pinecone/Chroma on purpose: the whole idea fits in ~100 lines, and writing it
yourself shows you understand what those tools do.

* Vector search: cosine similarity (vectors are L2-normalized, so it's a dot product).
* Keyword search: BM25, the classic search-engine ranking formula.
* Hybrid search: merge both rankings with Reciprocal Rank Fusion (RRF).
  Vectors catch meaning ("can I send it back" ~ "returns"), BM25 catches exact
  words like product names and numbers. Together they beat either one alone.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder, tokenize


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity of the chunk to the query (0-1, higher is closer)
    rank: int


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q_terms = tokenize(query)
        out = np.zeros(len(self.docs), dtype=np.float32)
        for i, (doc, length) in enumerate(zip(self.docs, self.lengths)):
            s = 0.0
            for term in q_terms:
                tf = doc.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                s += self.idf.get(term, 0) * tf * (self.k1 + 1) / denom
            out[i] = s
        return out


class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self._bm25: BM25 | None = None

    # ---- building -------------------------------------------------------
    def add(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        new_vectors = [
            self.embedder.embed([c.text for c in chunks[i : i + batch_size]])
            for i in range(0, len(chunks), batch_size)
        ]
        stacked = np.vstack(new_vectors) if new_vectors else np.zeros((0, 0), dtype=np.float32)
        self.vectors = stacked if self.vectors.size == 0 else np.vstack([self.vectors, stacked])
        self.chunks.extend(chunks)
        self._bm25 = None  # rebuild lazily

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([c.text for c in self.chunks])
        return self._bm25

    # ---- searching ------------------------------------------------------
    def search(self, query: str, k: int = 4, mode: str = "hybrid", rrf_k: int = 60) -> list[SearchResult]:
        """mode: 'vector', 'keyword' or 'hybrid'."""
        if not self.chunks:
            return []
        q_vec = self.embedder.embed([query])[0]
        cosine = self.vectors @ q_vec

        if mode == "vector":
            order = np.argsort(-cosine)
        elif mode == "keyword":
            order = np.argsort(-self.bm25.scores(query), kind="stable")
        else:
            vec_rank = np.argsort(-cosine)
            kw_rank = np.argsort(-self.bm25.scores(query), kind="stable")
            fused = np.zeros(len(self.chunks))
            for ranking in (vec_rank, kw_rank):
                for position, idx in enumerate(ranking):
                    fused[idx] += 1.0 / (rrf_k + position + 1)
            order = np.argsort(-fused)

        return [
            SearchResult(chunk=self.chunks[i], score=float(cosine[i]), rank=r + 1)
            for r, i in enumerate(order[:k])
        ]

    # ---- persistence ----------------------------------------------------
    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        meta = {"embedder": self.embedder.name, "chunks": [asdict(c) for c in self.chunks]}
        (folder / "chunks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path, embedder: Embedder) -> "VectorStore":
        folder = Path(folder)
        meta = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.name:
            raise ValueError(
                f"Index was built with '{meta['embedder']}' but you are using '{embedder.name}'. "
                "Re-run `ingest` with the same embedder."
            )
        store = cls(embedder)
        store.chunks = [Chunk(**c) for c in meta["chunks"]]
        store.vectors = np.load(folder / "vectors.npy")
        return store
