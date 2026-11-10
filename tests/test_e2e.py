"""End-to-end pipeline test: capture -> walk -> embed -> store -> query.

Runs the real WAL, the real backward walk, the real embedder, and the real
Valkey-shaped backend against a live HTTP embedding server. The vector backend
is swapped for an in-process implementation because Valkey/Qdrant are external
services; everything else is production code.
"""

from __future__ import annotations

import asyncio

import pytest

from goneby.config.settings import GoneByConfig
from goneby.crypto.cipher import Cipher
from goneby.memory.embedder import Embedder, EmbeddingConfig
from goneby.memory.store import MemoryStore
from goneby.vector.factory import create_backend
from goneby.wal.manager import WALManager
from goneby.walk.backward import BackwardWalker
from tests.test_memory import MemBackend, ServerFixture


def run(coro):
    return asyncio.run(coro)


MVP = """
Shipped MVP: GoneBy, an MCP server for retroactive project memory.
- Write-Ahead Log encrypted with ChaCha20-Poly1305, one entry per write.
- Backward walk filters the WAL using the shipped product as ground truth.
- Qdrant was selected as the vector backend for large installs because it
  scales past one million vectors with on-disk indexing.
- The WAL is append-only; nothing is rewritten during capture.
"""


ENTRIES = [
    ("We considered Redis as the vector store but it is RAM-bound past 1M vectors.", "finding"),
    ("Chose Qdrant for the vector backend; it tiers to disk and scales past a million vectors.", "decision"),
    ("Thinking about making a todo app with a subscription model.", "note"),
    ("WAL entries are encrypted per-entry with ChaCha20-Poly1305 and a random nonce.", "decision"),
    ("Ran a benchmark: ember data in a toaster is an absurd benchmark idea.", "note"),
    ("The WAL is append-only; existing entries are never rewritten during capture.", "decision"),
]


class TestFullPipeline:
    def test_end_to_end(self, tmp_path):
        async def go():
            wal_path = tmp_path / "e2e.wal"
            cipher = Cipher(Cipher.generate_key())
            wal = WALManager(wal_path, cipher)

            # --- Phase 1: capture (zero token cost, no filtering) ---
            for text, kind in ENTRIES:
                wal.append(text, {"type": kind})
            assert wal.entry_count == len(ENTRIES)

            # Raw file must contain no plaintext
            raw = wal_path.read_text()
            assert "Qdrant" not in raw
            assert "ChaCha20" not in raw

            # --- Phase 2: backward walk against shipped MVP ---
            # A stand-in judge stands in for the LLM: keep entries sharing a
            # content term with the MVP. Direction of walk is newest->oldest.
            mvp_terms = {
                "qdrant", "vector", "backend", "wal", "append",
                "chacha20-poly1305", "encrypt",
            }

            async def judge(_mvp, entries):
                flags = []
                for entry in entries:
                    tokens = set(
                        entry["content"].lower().replace(",", " ").replace(".", " ").split()
                    )
                    flags.append(bool(tokens & mvp_terms))
                return flags

            entries = wal.read_all()
            result = await BackwardWalker(judge).walk(MVP, entries)

            assert result.total_count == len(ENTRIES)
            assert result.discarded_count == 2  # the two junk notes
            assert result.kept_count == 4
            # Kept entries must come back in chronological order
            assert [e["content"] for e in result.kept_entries] == [
                t for t, _ in ENTRIES if t not in {
                    "Thinking about making a todo app with a subscription model.",
                    "Ran a benchmark: ember data in a toaster is an absurd benchmark idea.",
                }
            ]

            # --- Phase 3: embed and store ---
            async with ServerFixture() as srv:
                embedder = Embedder(
                    EmbeddingConfig(base_url=srv.base_url, dimensions=256)
                )
                store = MemoryStore(embedder, MemBackend(), project="e2e")

                indexed = await store.index_entries(result.kept_entries)
                assert indexed["status"] == "ok"
                assert indexed["stored"] == 4
                assert indexed["dimensions"] == 256

                # --- Phase 4: retrieve ---
                # The fake embedder has no semantic structure, so this asserts
                # the round trip and result shape, NOT ranking quality. Only
                # real-model checks can judge whether Qdrant outranks junk.
                hits = await store.search("which vector backend did we choose", top_k=3)
                assert hits["status"] == "ok"
                assert hits["count"] == 3  # top_k honored
                assert 0.0 < hits["results"][0]["score"] <= 1.0
                assert hits["results"][0]["project"] == "e2e"
                # Junk never entered the store in the first place — the walk
                # removed it before embedding, so it cannot surface at all.
                indexed_text = " ".join(r["text"] for r in hits["results"])
                assert "toaster" not in indexed_text
                assert "subscription" not in indexed_text

                health = await store.health()
                assert health["embedder"]["ok"] is True
                assert health["stored_vectors"] == 4
                await embedder.aclose()

        run(go())

    def test_wal_survives_roundtrip_across_cipher_instances(self, tmp_path):
        """Reopening the WAL with the same key must reproduce entries exactly."""
        key = Cipher.generate_key()
        wal_path = tmp_path / "persist.wal"

        WALManager(wal_path, Cipher(key)).append("durable entry", {"type": "note"})
        reopened = WALManager(wal_path, Cipher(key))
        assert reopened.read_all()[0]["content"] == "durable entry"

    def test_wrong_key_cannot_read(self, tmp_path):
        wal_path = tmp_path / "locked.wal"
        WALManager(wal_path, Cipher(Cipher.generate_key())).append("secret", {"type": "note"})

        reader = WALManager(wal_path, Cipher(Cipher.generate_key()))
        entries = reader.read_all()
        assert "CORRUPTED" in entries[0]["content"]  # fails closed, leaks nothing

    def test_store_memory_is_idempotent_across_runs(self, tmp_path):
        """Re-indexing the same walked set must not duplicate vectors."""
        async def go():
            async with ServerFixture() as srv:
                embedder = Embedder(
                    EmbeddingConfig(base_url=srv.base_url, dimensions=128)
                )
                backend = MemBackend()
                store = MemoryStore(embedder, backend)
                entries = [{"content": "qdrant scales to disk", "metadata": {}}]

                await store.index_entries(entries)
                await store.index_entries(entries)
                await store.index_entries(entries)
                assert await backend.count() == 1
                await embedder.aclose()

        run(go())

    def test_factory_builds_configured_backend(self):
        cfg = GoneByConfig(vector_backend="qdrant", embedding_dimensions=384)
        backend = create_backend(cfg)
        assert type(backend).__name__ == "QdrantBackend"
        assert backend._dimensions == 384

        cfg2 = GoneByConfig(vector_backend="valkey")
        assert type(create_backend(cfg2)).__name__ == "ValkeyBackend"

    def test_factory_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown vector backend"):
            create_backend(GoneByConfig(vector_backend="mysql"))
