"""Qdrant vector backend — for larger installs."""

from __future__ import annotations

from typing import Any

from goneby.vector.base import VectorBackend
from goneby.vector.filtering import matches_filter


class QdrantBackend(VectorBackend):
    """Qdrant vector backend using qdrant-client."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "goneby",
        dimensions: int = 512,
    ):
        self._host = host
        self._port = port
        self._collection = collection
        self._dimensions = dimensions
        self._client = None

    async def connect(self) -> None:
        """Initialize Qdrant connection."""
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = AsyncQdrantClient(host=self._host, port=self._port)

        # Create collection if it doesn't exist
        collections = await self._client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self._collection not in collection_names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimensions, distance=Distance.COSINE
                ),
            )

    async def disconnect(self) -> None:
        """Close Qdrant connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def upsert(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """Insert or update a vector with metadata."""
        if not self._client:
            raise RuntimeError("Not connected to Qdrant")

        from qdrant_client.models import PointStruct

        await self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=id,
                    vector=vector,
                    payload=metadata,
                )
            ],
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors, optionally constrained by metadata.

        Uses Qdrant's native payload index when a filter is supplied, then
        re-checks with the shared predicate — Qdrant matches on exact field
        values, and the shared check keeps list-valued constraints consistent
        with the other backends.
        """
        if not self._client:
            raise RuntimeError("Not connected to Qdrant")

        query_filter = None
        if filter:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            must = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter.items()
                if not isinstance(value, (list, tuple, set))
            ]
            if must:
                query_filter = Filter(must=must)

        results = await self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

        out = []
        for r in results:
            metadata = r.payload or {}
            if filter and not matches_filter(metadata, filter):
                continue
            out.append({"id": str(r.id), "score": r.score, "metadata": metadata})
        return out

    async def delete(self, id: str) -> None:
        """Delete a vector by ID."""
        if not self._client:
            raise RuntimeError("Not connected to Qdrant")

        from qdrant_client.models import PointIdsList

        await self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[id]),
        )

    async def clear(self) -> None:
        """Clear all vectors in the collection."""
        if not self._client:
            raise RuntimeError("Not connected to Qdrant")

        from qdrant_client.models import FilterSelector

        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(filter={}),
        )

    async def count(self) -> int:
        """Return number of vectors."""
        if not self._client:
            raise RuntimeError("Not connected to Qdrant")

        result = await self._client.count(collection_name=self._collection)
        return result.count
