"""Tests for the embedding client and memory store.

No live embedding server is required: a deterministic fake
OpenAI-compatible /v1/embeddings endpoint stands in for llama.cpp.
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from goneby.memory.embedder import (
    Embedder,
    EmbeddingConfig,
    EmbeddingError,
    l2_normalize,
    mrl_truncate,
)
from goneby.memory.store import MemoryStore, content_id
from goneby.vector.base import VectorBackend
from goneby.vector.filtering import matches_filter

# --------------------------------------------------------------------------
# Fake embedding server
# --------------------------------------------------------------------------

DIM = 1024


def fake_vector(text: str, dim: int = DIM) -> list[float]:
    """Deterministic pseudo-embedding derived from the text.

    NOT a semantic embedding. Each token is hashed to an arbitrary direction,
    so similar texts do NOT reliably land near each other and ranking order is
    meaningless. It exists only to exercise the plumbing — batching, ordering,
    truncation, normalization, and round-tripping — and to keep the suite free
    of a live model dependency.

    Retrieval QUALITY must be verified against a real model, not this fake.
    """
    vec = [0.0] * dim
    tokens = [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if t]
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        for i in range(dim):
            vec[i] += digest[i % 32] / 255.0 - 0.5
    return vec or [0.0] * dim


def make_fake_app(*, shuffle: bool = False, fail: bool = False) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def models() -> dict:
        return {"data": [{"id": "qwen3-embedding-0.6b"}]}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        if fail:
            # Must raise a real HTTP error — returning an httpx.Response here
            # would be serialized as a 200 JSON body by FastAPI.
            raise HTTPException(status_code=500, detail="simulated embedder failure")
        body = await request.json()
        items = [
            {"object": "embedding", "index": i, "embedding": fake_vector(t)}
            for i, t in enumerate(body["input"])
        ]
        if shuffle:
            items.reverse()
        return {"object": "list", "data": items, "model": body.get("model", "test")}

    return app


class ServerFixture:
    """Run the fake app in a background uvicorn server."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._server: uvicorn.Server | None = None
        self._task = None
        self.base_url = ""

    async def __aenter__(self):
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        self.base_url = f"http://127.0.0.1:{port}/v1"

        config = uvicorn.Config(
            make_fake_app(**self._kwargs),
            host="127.0.0.1",
            port=port,
            log_level="error",
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())

        for _ in range(100):
            if self._server.started:
                break
            await asyncio.sleep(0.05)
        return self

    async def __aexit__(self, *exc):
        if self._server:
            self._server.should_exit = True
        if self._task:
            await self._task


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# In-memory backend for store tests
# --------------------------------------------------------------------------


class MemBackend(VectorBackend):
    def __init__(self):
        self.data: dict[str, tuple[list[float], dict]] = {}

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def upsert(self, id, vector, metadata) -> None:
        self.data[id] = (vector, metadata)

    async def search(self, query_vector, top_k=5, filter=None):
        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(y * y for y in b) ** 0.5
            return dot / (na * nb) if na and nb else 0.0

        scored = [
            {"id": k, "score": cos(query_vector, v), "metadata": m}
            for k, (v, m) in self.data.items()
            if not filter or matches_filter(m, filter)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    async def delete(self, id) -> None:
        self.data.pop(id, None)

    async def clear(self) -> None:
        self.data.clear()

    async def count(self) -> int:
        return len(self.data)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


class TestMRLHelpers:
    def test_truncate_shortens(self):
        assert len(mrl_truncate([1.0] * 1024, 512)) == 512

    def test_truncate_noop_when_target_larger(self):
        vec = [1.0] * 256
        assert mrl_truncate(vec, 512) == vec

    def test_normalize_unit_length(self):
        out = l2_normalize([3.0, 4.0])
        assert out == [0.6, 0.8]

    def test_normalize_zero_vector_is_safe(self):
        assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


class TestConfigValidation:
    def test_rejects_empty_base_url(self):
        cfg = EmbeddingConfig(base_url="", model="m")
        assert cfg.validate()

    def test_accepts_good_config(self):
        assert EmbeddingConfig().validate() == []


# --------------------------------------------------------------------------
# Embedder against the fake server
# --------------------------------------------------------------------------


class TestEmbedder:
    def test_embed_returns_truncated_normalized(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url, dimensions=512))
                vecs = await emb.embed(["vector backend decision"])
                await emb.aclose()
                assert len(vecs) == 1
                assert len(vecs[0]) == 512
                norm = sum(v * v for v in vecs[0]) ** 0.5
                assert abs(norm - 1.0) < 1e-6

        run(go())

    def test_embed_preserves_order_when_server_shuffles(self):
        async def go():
            async with ServerFixture(shuffle=True) as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url, dimensions=64))
                inputs = ["alpha token", "beta token", "gamma token"]
                vecs = await emb.embed(inputs)
                await emb.aclose()
                assert len(vecs) == 3
                # MRL order matters: truncate first, then normalize.
                for got, want in zip(vecs, inputs):
                    expected = l2_normalize(mrl_truncate(fake_vector(want), 64))
                    assert got == pytest.approx(expected)

        run(go())

    def test_empty_input_returns_empty(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url))
                assert await emb.embed([]) == []
                await emb.aclose()

        run(go())

    def test_rejects_blank_text(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url))
                with pytest.raises(EmbeddingError):
                    await emb.embed(["ok", "   "])
                await emb.aclose()

        run(go())

    def test_server_error_raises(self):
        async def go():
            async with ServerFixture(fail=True) as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url))
                with pytest.raises(EmbeddingError, match="500"):
                    await emb.embed(["x"])
                await emb.aclose()

        run(go())

    def test_unreachable_server_raises_clear_error(self):
        async def go():
            emb = Embedder(EmbeddingConfig(base_url="http://127.0.0.1:1/v1", timeout=1.0))
            with pytest.raises(EmbeddingError, match="unreachable"):
                await emb.embed(["x"])
            await emb.aclose()

        run(go())

    def test_health_ok(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url))
                report = await emb.health()
                await emb.aclose()
                assert report["ok"] is True
                assert "qwen3-embedding-0.6b" in report["models"]

        run(go())

    def test_health_reports_failure_without_raising(self):
        async def go():
            emb = Embedder(EmbeddingConfig(base_url="http://127.0.0.1:1/v1", timeout=1.0))
            report = await emb.health()
            await emb.aclose()
            assert report["ok"] is False

        run(go())

    def test_batch_size_splits_requests(self):
        def go():
            async def go2():
                async with ServerFixture() as srv:
                    emb = Embedder(
                        EmbeddingConfig(base_url=srv.base_url, dimensions=32, batch_size=2)
                    )
                    vecs = await emb.embed_batched([f"text {i}" for i in range(5)])
                    await emb.aclose()
                    assert len(vecs) == 5

            asyncio.run(go2())

        go()

    def test_batch_progress_callback_fires(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(
                    EmbeddingConfig(base_url=srv.base_url, dimensions=32, batch_size=2)
                )
                marks: list[tuple[int, int]] = []
                await emb.embed_batched(
                    [f"t{i}" for i in range(5)], progress=lambda d, t: marks.append((d, t))
                )
                await emb.aclose()
                assert marks == [(2, 5), (4, 5), (5, 5)]

        run(go())


# --------------------------------------------------------------------------
# MemoryStore
# --------------------------------------------------------------------------


class TestMemoryStore:
    def test_index_and_search_roundtrip(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url, dimensions=256))
                backend = MemBackend()
                store = MemoryStore(emb, backend, project="goneby-test")

                result = await store.index_entries(
                    [
                        {"content": "chose qdrant for the vector backend", "metadata": {"type": "decision"}},
                        {"content": "wal uses chacha20 poly1305 encryption", "metadata": {"type": "finding"}},
                    ]
                )
                assert result["status"] == "ok"
                assert result["stored"] == 2
                assert result["dimensions"] == 256

                hits = await store.search("qdrant vector backend")
                assert hits["status"] == "ok"
                assert hits["count"] >= 1
                top = hits["results"][0]
                assert "qdrant" in top["text"]
                assert top["project"] == "goneby-test"
                await emb.aclose()

        run(go())

    def test_index_is_idempotent(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url, dimensions=64))
                backend = MemBackend()
                store = MemoryStore(emb, backend)
                entry = [{"content": "same text", "metadata": {}}]
                await store.index_entries(entry)
                await store.index_entries(entry)
                assert len(backend.data) == 1
                await emb.aclose()

        run(go())

    def test_blank_entries_skipped_not_fatal(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url, dimensions=64))
                store = MemoryStore(emb, MemBackend())
                result = await store.index_entries(
                    [{"content": "   ", "metadata": {}}, {"content": "real", "metadata": {}}]
                )
                assert result["stored"] == 1
                assert result["skipped"] == 1
                await emb.aclose()

        run(go())

    def test_index_nothing_returns_empty(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url))
                store = MemoryStore(emb, MemBackend())
                result = await store.index_entries([])
                assert result["status"] == "empty"
                await emb.aclose()

        run(go())

    def test_search_rejects_empty_query(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url, dimensions=32))
                store = MemoryStore(emb, MemBackend())
                result = await store.search("  ")
                assert result["status"] == "error"
                await emb.aclose()

        run(go())

    def test_min_score_filters_weak_hits(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url, dimensions=128))
                store = MemoryStore(emb, MemBackend())
                await store.index_entries([{"content": "alpha", "metadata": {}}])
                strict = await store.search("unrelated zulu content", min_score=0.99)
                assert strict["count"] == 0
                await emb.aclose()

        run(go())

    def test_health_does_not_raise_when_embedder_down(self):
        async def go():
            emb = Embedder(EmbeddingConfig(base_url="http://127.0.0.1:1/v1", timeout=1.0))
            store = MemoryStore(emb, MemBackend())
            report = await store.health()
            assert report["embedder"]["ok"] is False
            assert report["backend"] == "MemBackend"
            await emb.aclose()

        run(go())

    def test_content_id_is_deterministic_and_distinct(self):
        assert content_id("abc") == content_id("abc")
        assert content_id("abc") != content_id("abd")
        assert len(content_id("abc")) == 32
