"""Abstract base class for vector backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorBackend(ABC):
    """Abstract vector backend interface."""

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to the vector store."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        ...

    @abstractmethod
    async def upsert(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """Insert or update a vector with metadata."""
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors. Returns list of {id, score, metadata}.

        `filter` is a metadata equality filter (e.g. {"project": "goneby"}).
        Backends that cannot filter natively must filter results themselves.
        """
        ...

    @abstractmethod
    async def delete(self, id: str) -> None:
        """Delete a vector by ID."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all vectors in the collection."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return number of vectors in the store."""
        ...
