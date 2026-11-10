"""MemoryStore — index walked entries into the vector backend and retrieve them.

Owns the embedder lifecycle and the backend connection so the MCP tools stay
thin. Stable IDs are derived from a content hash, which makes re-storing the
same entry idempotent instead of duplicating it.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from goneby.memory.embedder import Embedder, EmbeddingError
from goneby.vector.base import VectorBackend

logger = logging.getLogger(__name__)


def content_id(text: str) -> str:
    """Deterministic ID from content, so repeated stores upsert instead of duplicate."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class MemoryStore:
    """Embedding + vector storage for filtered project memories."""

    def __init__(self, embedder: Embedder, backend: VectorBackend, project: str = "default"):
        self._embedder = embedder
        self._backend = backend
        self._project = project
        self._connected = False

    async def connect(self) -> None:
        if not self._connected:
            await self._backend.connect()
            self._connected = True

    async def disconnect(self) -> None:
        if self._connected:
            await self._backend.disconnect()
            self._connected = False

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__

    async def index_entries(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Embed and store walked entries. Returns a result summary.

        Entries without usable text are counted as skipped rather than failing
        the batch — a partially noisy WAL should still index what it can.
        """
        await self.connect()

        indexable: list[tuple[str, str, dict[str, Any]]] = []
        skipped = 0
        for entry in entries:
            text = (entry.get("content") or "").strip()
            if not text:
                skipped += 1
                continue
            indexable.append((content_id(text), text, entry.get("metadata") or {}))

        if not indexable:
            return {
                "status": "empty",
                "stored": 0,
                "skipped": skipped,
                "message": "no indexable entries",
            }

        texts = [text for _, text, _ in indexable]
        vectors = await self._embedder.embed_batched(texts)

        for (entry_id, text, metadata), vector in zip(indexable, vectors):
            await self._backend.upsert(
                id=entry_id,
                vector=vector,
                metadata={
                    "text": text,
                    "timestamp": metadata.get("timestamp"),
                    "type": metadata.get("type", "note"),
                    "project": self._project,
                },
            )

        return {
            "status": "ok",
            "stored": len(indexable),
            "skipped": skipped,
            "dimensions": len(vectors[0]) if vectors else 0,
            "backend": self.backend_name,
        }

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        scope: str = "project",
    ) -> dict[str, Any]:
        """Semantic search over stored memories.

        Args:
            query: Natural-language question.
            top_k: Maximum results to return.
            min_score: Minimum cosine similarity to include a hit.
            scope: ``"project"`` restricts to this store's project;
                ``"all"`` searches every project's memories.
        """
        await self.connect()

        if not query.strip():
            return {"status": "error", "error": "query must not be empty", "results": []}
        if scope not in ("project", "all"):
            return {
                "status": "error",
                "error": f"scope must be 'project' or 'all', got {scope!r}",
                "results": [],
            }

        vector = await self._embedder.embed_one(query)
        metadata_filter = {"project": self._project} if scope == "project" else None
        hits = await self._backend.search(
            query_vector=vector, top_k=top_k, filter=metadata_filter
        )

        results = []
        for hit in hits:
            if hit.get("score", 0.0) < min_score:
                continue
            metadata = hit.get("metadata") or {}
            results.append(
                {
                    "id": hit.get("id"),
                    "score": round(float(hit.get("score", 0.0)), 4),
                    "text": metadata.get("text", ""),
                    "timestamp": metadata.get("timestamp"),
                    "type": metadata.get("type"),
                    "project": metadata.get("project"),
                }
            )

        return {
            "status": "ok",
            "scope": scope,
            "project": self._project,
            "count": len(results),
            "results": results,
        }

    async def health(self) -> dict[str, Any]:
        """Report embedder and backend reachability without raising."""
        report: dict[str, Any] = {"backend": self.backend_name}
        try:
            probe = await self._embedder.health()
            report["embedder"] = probe
        except Exception as exc:  # noqa: BLE001 - health must not raise
            report["embedder"] = {"ok": False, "detail": str(exc)}
        try:
            await self.connect()
            report["stored_vectors"] = await self._backend.count()
        except Exception as exc:  # noqa: BLE001
            report["stored_vectors"] = None
            report["backend_error"] = str(exc)
        return report


__all__ = ["EmbeddingError", "MemoryStore", "content_id"]
