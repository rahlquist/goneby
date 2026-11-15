"""Qdrant backend tests.

Unit level: a fake client mirrors the *modern* qdrant-client API that the
project pins (>=1.7,<2): ``query_points`` exists, legacy ``search`` was
removed. The old code called ``.search()`` and died on every query with the
pinned client (ctx-audit: 'Qdrant for larger stocks').

Integration level: ``@pytest.mark.integration`` runs against a real Qdrant;
it skips unless one is reachable. Start one with:
    docker run -d --rm --name goneby-qdrant-test -p 6333:6333 qdrant/qdrant:latest
"""

from __future__ import annotations

import asyncio
import os
import socket
from types import SimpleNamespace

import pytest

from goneby.vector.qdrant_backend import QdrantBackend


def run(coro):
    return asyncio.run(coro)


class ModernFakeClient:
    """Only the API surface that qdrant-client >=1.7 actually has."""

    def __init__(self, points):
        self.points = points
        self.query_calls: list[dict] = []
        self.upserted: list[dict] = []

    # Deliberately NO .search attribute: code calling the removed method
    # must hit AttributeError, which is exactly the audited defect.

    async def query_points(self, *, collection_name, query, limit, query_filter=None, with_payload=True):
        self.query_calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "limit": limit,
                "query_filter": query_filter,
                "with_payload": with_payload,
            }
        )
        return SimpleNamespace(points=self.points[:limit])

    async def upsert(self, *, collection_name, points):
        for p in points:
            self.upserted.append({"id": p.id, "vector": p.vector, "payload": dict(p.payload)})


def _hit(id, score, payload):
    return SimpleNamespace(id=id, score=score, payload=payload)


class TestSearchUsesModernApi:
    def test_search_calls_query_points(self):
        backend = QdrantBackend(collection="memories")
        backend._client = ModernFakeClient(
            [_hit("uuid-ish", 0.9, {"text": "kept", "project": "alpha", "_goneby_id": "a1"})]
        )

        out = run(backend.search([0.1, 0.2, 0.3], top_k=3))

        assert len(backend._client.query_calls) == 1
        call = backend._client.query_calls[0]
        assert call["collection_name"] == "memories"
        assert call["query"] == [0.1, 0.2, 0.3]
        assert call["with_payload"] is True
        assert out == [{"id": "a1", "score": 0.9, "metadata": {"text": "kept", "project": "alpha"}}]

    def test_metadata_filter_rechecks_project(self):
        backend = QdrantBackend()
        backend._client = ModernFakeClient(
            [
                _hit("u1", 0.9, {"text": "mine", "project": "alpha", "_goneby_id": "a1"}),
                _hit("u2", 0.8, {"text": "foreign", "project": "beta", "_goneby_id": "b1"}),
            ]
        )

        out = run(backend.search([0.1], top_k=5, filter={"project": "alpha"}))

        assert [r["id"] for r in out] == ["a1"]

    def test_point_id_policy_matches_live_server(self):
        """Live-server verified: hex32 IDs are accepted (simple-form UUIDs),
        non-UUID strings like 'north' are rejected, huge digit strings would
        overflow u64 — usable IDs pass through, the rest map to UUIDv5."""
        import re

        backend = QdrantBackend()
        fake = ModernFakeClient([])
        backend._client = fake
        uuid5 = re.compile(
            r"[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
        )

        hex32 = "d41d8cd98f00b204e9800998ecf8427e"  # what content_id() produces
        run(backend.upsert(id=hex32, vector=[1.0, 0.0], metadata={"text": "t", "project": "p"}))
        assert fake.upserted[0]["id"] == hex32, "hex32 content IDs must pass through verbatim"
        assert fake.upserted[0]["payload"]["_goneby_id"] == hex32

        run(backend.upsert(id="north", vector=[1.0, 0.0], metadata={}))
        assert uuid5.fullmatch(fake.upserted[1]["id"]), "non-UUID string must map to UUIDv5"
        assert fake.upserted[1]["payload"]["_goneby_id"] == "north"
        run(backend.upsert(id="north", vector=[1.0, 0.0], metadata={}))
        assert fake.upserted[2]["id"] == fake.upserted[1]["id"], "mapping must be deterministic"

        run(backend.upsert(id="42", vector=[1.0, 0.0], metadata={}))
        assert fake.upserted[3]["id"] == "42", "in-range ints pass through"

        huge = "1" * 21  # digit string beyond u64 and not 32-hex-shaped → must remap
        run(backend.upsert(id=huge, vector=[1.0, 0.0], metadata={}))
        assert fake.upserted[4]["id"] != huge
        assert uuid5.fullmatch(fake.upserted[4]["id"])


@pytest.mark.integration
class TestLiveQdrant:
    """Real server, real wire format — the class of defect mocks never catch."""

    @staticmethod
    def _server_reachable(host="127.0.0.1", port=6333) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    def test_roundtrip_against_real_server(self):
        if not self._server_reachable():
            pytest.skip("no live Qdrant on :6333 — start the docker container from the module docstring")

        collection = f"goneby_it_{os.getpid()}"

        async def go():
            backend = QdrantBackend(
                host="127.0.0.1", port=6333, collection=collection, dimensions=4
            )
            await backend.connect()
            try:
                # hex32 IDs like content_id() produces — the format that
                # Qdrant's wire API rejects as a point ID.
                north = "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a"
                also = "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b"
                east = "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c"
                data = {
                    north: ([1.0, 0.0, 0.0, 0.0], "north", "alpha"),
                    also: ([0.9, 0.1, 0.0, 0.0], "also-north", "alpha"),
                    east: ([0.0, 1.0, 0.0, 0.0], "east", "beta"),
                }
                for mem_id, (vector, text, project) in data.items():
                    await backend.upsert(
                        id=mem_id, vector=vector, metadata={"text": text, "project": project}
                    )
                assert await backend.count() == 3

                hits = await backend.search([1.0, 0.0, 0.0, 0.0], top_k=3)
                assert hits[0]["id"] == north
                assert hits[0]["metadata"]["text"] == "north"

                scoped = await backend.search(
                    [1.0, 0.0, 0.0, 0.0], top_k=3, filter={"project": "alpha"}
                )
                assert {h["id"] for h in scoped} == {north, also}

                await backend.delete(id=east)
                assert await backend.count() == 2
            finally:
                client = backend._client
                await client.delete_collection(collection_name=collection)
                await backend.disconnect()

        run(go())
