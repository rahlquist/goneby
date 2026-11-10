"""ValkeyBackend behavior tests against a stubbed async client.

These exist because the original backend mixed the *sync* `valkey.Valkey`
client into `await`ed calls — a TypeError that only fires on a live
connect and was invisible to the in-memory-backend suite. The stub below
mimics the `valkey.asyncio` contract (coroutine methods, async-iterable
`scan_iter`); a sync client would return raw values where this expects
coroutines and the awaits would fail.

Async bodies run through asyncio.run() to match the suite's convention
(no pytest-asyncio in this project).
"""

from __future__ import annotations

import asyncio

import pytest

from goneby.vector.valkey_backend import ValkeyBackend

valkey_asyncio = pytest.importorskip("valkey.asyncio")


class FakeAsyncValkey:
    """Stand-in for valkey.asyncio.Valkey: every op is a coroutine."""

    def __init__(self, *_args, **_kwargs):
        self.hashes: dict[str, dict[str, str]] = {}
        self.closed = False

    async def ping(self):
        return True

    async def hset(self, key, mapping=None):
        self.hashes.setdefault(key, {}).update(mapping or {})

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def delete(self, key):
        return 1 if self.hashes.pop(key, None) else 0

    def scan_iter(self, match=None):
        keys = list(self.hashes)
        if match and "*" in match:
            prefix = match.split("*")[0]
            keys = [k for k in keys if k.startswith(prefix)]

        async def _gen():
            for k in keys:
                yield k

        return _gen()

    async def close(self):
        self.closed = True


@pytest.fixture()
def stubbed(monkeypatch):
    created: list[FakeAsyncValkey] = []

    def factory(*args, **kwargs):
        client = FakeAsyncValkey(*args, **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(valkey_asyncio, "Valkey", factory)
    return created


async def _connect_uses_async_client(stubbed):
    backend = ValkeyBackend(host="localhost", port=6379)
    await backend.connect()
    assert len(stubbed) == 1
    assert isinstance(stubbed[0], FakeAsyncValkey)


def test_connect_uses_async_client_and_pings(stubbed):
    asyncio.run(_connect_uses_async_client(stubbed))


async def _upsert_search_and_filter():
    backend = ValkeyBackend()
    await backend.connect()
    await backend.upsert("id-a", [1.0, 0.0], {"text": "a", "project": "alpha", "type": "note"})
    await backend.upsert("id-b", [0.0, 1.0], {"text": "b", "project": "beta", "type": "note"})

    hits = await backend.search([1.0, 0.0], top_k=5, filter={"project": "alpha"})
    assert [h["id"] for h in hits] == ["id-a"]
    assert hits[0]["score"] == pytest.approx(1.0)

    all_hits = await backend.search([0.0, 1.0], top_k=5)
    assert {h["id"] for h in all_hits} == {"id-a", "id-b"}
    assert all_hits[0]["id"] == "id-b"


def test_upsert_search_and_filter(stubbed):
    asyncio.run(_upsert_search_and_filter())


async def _delete_count_clear_disconnect():
    backend = ValkeyBackend()
    await backend.connect()
    await backend.upsert("x", [0.5, 0.5], {"project": "p"})
    assert await backend.count() == 1
    await backend.delete("x")
    assert await backend.count() == 0

    await backend.upsert("y", [0.5, 0.5], {"project": "p"})
    await backend.clear()
    assert await backend.count() == 0

    await backend.disconnect()


def test_delete_count_clear_disconnect(stubbed):
    asyncio.run(_delete_count_clear_disconnect())
    assert stubbed[0].closed is True


async def _requires_connection():
    backend = ValkeyBackend()
    with pytest.raises(RuntimeError, match="Not connected"):
        await backend.upsert("id", [1.0], {})


def test_operations_require_connection():
    asyncio.run(_requires_connection())
