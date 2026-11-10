"""In-memory vector backend.

Useful for short-lived projects, tests, and machines where running Valkey or
Qdrant is not worth it. Cosine similarity is computed brute-force; there is no
persistence, so memories vanish with the process.
"""

from __future__ import annotations

import math
from typing import Any

from goneby.vector.base import VectorBackend
from goneby.vector.filtering import matches_filter


class MemoryBackend(VectorBackend):
    """Process-local vector store with cosine similarity."""

    def __init__(self):
        self._data: dict[str, tuple[list[float], dict[str, Any]]] = {}

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def upsert(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self._data[id] = (list(vector), dict(metadata))

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        scored = []
        for key, (vector, metadata) in self._data.items():
            if filter and not matches_filter(metadata, filter):
                continue

            dot = sum(a * b for a, b in zip(query_vector, vector))
            norm_a = math.sqrt(sum(a * a for a in query_vector))
            norm_b = math.sqrt(sum(b * b for b in vector))
            if norm_a == 0 or norm_b == 0:
                continue

            scored.append(
                {
                    "id": key,
                    "score": dot / (norm_a * norm_b),
                    "metadata": metadata,
                }
            )

        scored.sort(key=lambda hit: hit["score"], reverse=True)
        return scored[:top_k]

    async def delete(self, id: str) -> None:
        self._data.pop(id, None)

    async def clear(self) -> None:
        self._data.clear()

    async def count(self) -> int:
        return len(self._data)
