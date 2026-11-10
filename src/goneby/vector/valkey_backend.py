"""Valkey vector backend — default for small installs."""

from __future__ import annotations

import json
from typing import Any

from goneby.vector.base import VectorBackend
from goneby.vector.filtering import matches_filter


class ValkeyBackend(VectorBackend):
    """Valkey vector backend using Valkey's vector search capabilities."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self._host = host
        self._port = port
        self._db = db
        self._client = None

    async def connect(self) -> None:
        """Initialize Valkey connection."""
        import valkey

        self._client = valkey.Valkey(
            host=self._host,
            port=self._port,
            db=self._db,
            decode_responses=True,
        )
        await self._client.ping()

    async def disconnect(self) -> None:
        """Close Valkey connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def upsert(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """Insert or update a vector with metadata."""
        if not self._client:
            raise RuntimeError("Not connected to Valkey")

        # Store as hash with vector and metadata
        key = f"goneby:{id}"
        await self._client.hset(
            key,
            mapping={
                "vector": json.dumps(vector),
                "metadata": json.dumps(metadata),
            },
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors using brute-force cosine similarity.

        Metadata filtering happens here rather than natively: this backend scans
        and scores every key, so dropping non-matching entries before the
        top-k cut is both correct and free.
        """
        if not self._client:
            raise RuntimeError("Not connected to Valkey")

        # Valkey doesn't have native vector search in all versions
        # Fall back to brute-force scan with cosine similarity
        import math

        results = []
        async for key in self._client.scan_iter(match="goneby:*"):
            data = await self._client.hgetall(key)
            if not data:
                continue

            metadata = json.loads(data.get("metadata", "{}"))

            if filter and not matches_filter(metadata, filter):
                continue

            stored_vec = json.loads(data.get("vector", "[]"))
            if not stored_vec:
                continue

            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_vector, stored_vec))
            norm_a = math.sqrt(sum(a * a for a in query_vector))
            norm_b = math.sqrt(sum(b * b for b in stored_vec))

            if norm_a == 0 or norm_b == 0:
                continue

            results.append({
                "id": key.replace("goneby:", ""),
                "score": dot / (norm_a * norm_b),
                "metadata": metadata,
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def delete(self, id: str) -> None:
        """Delete a vector by ID."""
        if not self._client:
            raise RuntimeError("Not connected to Valkey")
        await self._client.delete(f"goneby:{id}")

    async def clear(self) -> None:
        """Clear all GoneBy vectors."""
        if not self._client:
            raise RuntimeError("Not connected to Valkey")

        async for key in self._client.scan_iter(match="goneby:*"):
            await self._client.delete(key)

    async def count(self) -> int:
        """Return number of vectors."""
        if not self._client:
            raise RuntimeError("Not connected to Valkey")

        count = 0
        async for _ in self._client.scan_iter(match="goneby:*"):
            count += 1
        return count
