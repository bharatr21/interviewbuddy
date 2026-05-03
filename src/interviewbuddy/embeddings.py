from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for text."""


@dataclass(frozen=True)
class HashEmbeddingProvider:
    dimensions: int = 128

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        return _normalize(vector)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]
