"""Text -> vector.

Two embedders share one tiny interface (`embed(list[str]) -> np.ndarray`):

* FastEmbedEmbedder  - a real semantic model (BAAI/bge-small-en-v1.5, ~70MB, runs on CPU,
                       no API key). Downloaded the first time you use it.
* HashingEmbedder    - a no-download fallback that hashes words and word pairs into a
                       fixed-size vector. It only matches shared words (no real "meaning"),
                       but it is deterministic and great for tests and offline demos.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i if in is isn it its me my of on or "
    "our so t than that the their them then there these this to us was we what when where which "
    "who why will with you your".split()
)


def stem(word: str) -> str:
    """A tiny stemmer so word forms match: notes -> note, synced -> sync, syncing -> sync."""
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 5 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        if w in STOPWORDS:
            continue
        out.append(stem(w))
    return out


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        # Bigger dim = fewer hash collisions (512 gave noticeably worse retrieval in eval).
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _index(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 - not used for security
        return int.from_bytes(digest[:4], "little") % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feat in features:
                matrix[row, self._index(feat)] += 1.0
        # log scaling so one repeated word doesn't dominate
        matrix = np.log1p(matrix)
        return _normalize(matrix)


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError("fastembed is not installed. Run: pip install fastembed") from exc
        self.name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return _normalize(vectors)


def get_embedder(kind: str = "auto") -> Embedder:
    """kind: 'fastembed', 'hashing', or 'auto' (fastembed if it loads, else hashing)."""
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder()
    try:
        return FastEmbedEmbedder()
    except Exception as exc:  # noqa: BLE001 - fall back on any download/import problem
        print(f"[embeddings] fastembed unavailable ({exc.__class__.__name__}); using offline hashing embedder")
        return HashingEmbedder()
