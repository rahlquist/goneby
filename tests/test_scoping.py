"""Tests for project isolation: per-project WALs and scoped search.

The backward walk must never see another project's evidence, and retrieval
must be isolated by default while still allowing deliberate cross-project
lookups.
"""

from __future__ import annotations

import asyncio

from goneby.config.settings import GoneByConfig, project_slug, project_wal_path
from goneby.crypto.cipher import Cipher
from goneby.memory.embedder import Embedder, EmbeddingConfig
from goneby.memory.store import MemoryStore
from goneby.wal.manager import WALManager
from tests.test_memory import MemBackend, ServerFixture


def run(coro):
    return asyncio.run(coro)


class TestProjectSlug:
    def test_simple_name_unchanged(self):
        assert project_slug("goneby") == "goneby"

    def test_path_flattened(self):
        slug = project_slug("/home/rahlquist/goneby")
        assert "/" not in slug
        assert slug == "home_rahlquist_goneby"

    def test_dots_and_dashes_kept(self):
        assert project_slug("my-project.v2") == "my-project.v2"

    def test_empty_falls_back_to_default(self):
        assert project_slug("") == "default"
        assert project_slug("   ") == "default"
        assert project_slug("///") == "default"

    def test_no_directory_traversal(self):
        assert ".." not in project_slug("../../etc/passwd")


class TestProjectWalPath:
    def test_per_project_files(self, tmp_path):
        a = project_wal_path(tmp_path, "alpha")
        b = project_wal_path(tmp_path, "beta")
        assert a != b
        assert a.name == "alpha.wal"
        assert b.name == "beta.wal"
        assert a.parent == b.parent == tmp_path

    def test_default_project_is_namespaced(self, tmp_path):
        assert project_wal_path(tmp_path, "default").name == "default.wal"

    def test_different_projects_do_not_share_a_wal(self, tmp_path):
        """The core isolation guarantee: a walk for A cannot see B."""
        key = Cipher.generate_key()
        a = WALManager(project_wal_path(tmp_path, "alpha"), Cipher(key))
        b = WALManager(project_wal_path(tmp_path, "beta"), Cipher(key))

        a.append("alpha decision: chose qdrant", {"type": "decision"})
        b.append("beta decision: chose valkey", {"type": "decision"})

        assert [e["content"] for e in a.read_all()] == ["alpha decision: chose qdrant"]
        assert [e["content"] for e in b.read_all()] == ["beta decision: chose valkey"]
        assert a.entry_count == 1 and b.entry_count == 1


class TestScopedSearch:
    def test_default_scope_excludes_other_projects(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url, dimensions=128))
                backend = MemBackend()

                alpha = MemoryStore(emb, backend, project="alpha")
                beta = MemoryStore(emb, backend, project="beta")

                await alpha.index_entries(
                    [{"content": "alpha only: qdrant backend choice", "metadata": {}}]
                )
                await beta.index_entries(
                    [{"content": "beta only: valkey backend choice", "metadata": {}}]
                )

                hits = await alpha.search("backend choice")
                assert hits["status"] == "ok"
                assert hits["scope"] == "project"
                assert hits["project"] == "alpha"
                texts = " ".join(r["text"] for r in hits["results"])
                assert "qdrant" in texts
                assert "valkey" not in texts
                await emb.aclose()

        run(go())

    def test_scope_all_spans_projects(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url, dimensions=128))
                backend = MemBackend()

                alpha = MemoryStore(emb, backend, project="alpha")
                beta = MemoryStore(emb, backend, project="beta")
                await alpha.index_entries([{"content": "alpha qdrant choice", "metadata": {}}])
                await beta.index_entries([{"content": "beta valkey choice", "metadata": {}}])

                # min_score=-1 admits negative similarities so this asserts on
                # project membership, not on the fake embedder's arbitrary scores.
                hits = await alpha.search(
                    "backend", scope="all", top_k=10, min_score=-1.0
                )
                assert hits["scope"] == "all"
                assert len(hits["results"]) == 2
                assert {r["project"] for r in hits["results"]} == {"alpha", "beta"}
                await emb.aclose()

        run(go())

    def test_negative_similarity_dropped_by_default(self):
        """min_score=0.0 excludes entries the embedder rates as dissimilar.

        The fake embedder produces arbitrary directions, so a stored entry can
        score negatively against a query — the default floor must drop it.
        """
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url, dimensions=128))
                backend = MemBackend()
                store = MemoryStore(emb, backend, project="alpha")
                await store.index_entries([{"content": "alpha qdrant choice", "metadata": {}}])

                default = await store.search("backend")
                permissive = await store.search("backend", min_score=-1.0)
                assert permissive["count"] >= default["count"]
                assert all(r["score"] >= 0.0 for r in default["results"])
                await emb.aclose()

        run(go())

    def test_invalid_scope_is_rejected(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url, dimensions=128))
                store = MemoryStore(emb, MemBackend(), project="alpha")
                out = await store.search("anything", scope="everywhere")
                assert out["status"] == "error"
                assert "scope" in out["error"]
                await emb.aclose()

        run(go())

    def test_index_tags_entries_with_project(self):
        async def go():
            async with ServerFixture() as srv:
                emb = Embedder(EmbeddingConfig(base_url=srv.base_url, dimensions=128))
                backend = MemBackend()
                store = MemoryStore(emb, backend, project="tagged-project")
                await store.index_entries([{"content": "some memory", "metadata": {}}])

                _, (_, metadata) = next(iter(backend.data.items()))
                assert metadata["project"] == "tagged-project"
                await emb.aclose()

        run(go())


class TestBackendFilter:
    def test_matches_filter_equality(self):
        from goneby.vector.filtering import matches_filter

        assert matches_filter({"project": "a"}, {"project": "a"})
        assert not matches_filter({"project": "a"}, {"project": "b"})
        assert matches_filter({}, {})
        assert not matches_filter({}, {"project": "a"})
        assert not matches_filter({"project": "a"}, {"missing": "x"})

    def test_matches_filter_list_constraint(self):
        from goneby.vector.filtering import matches_filter

        assert matches_filter({"project": "a"}, {"project": ["a", "b"]})
        assert not matches_filter({"project": "c"}, {"project": ["a", "b"]})

    def test_mem_backend_honours_filter(self):
        async def go():
            backend = MemBackend()
            await backend.connect()
            await backend.upsert(
                id="1", vector=[1.0, 0.0], metadata={"project": "a", "text": "alpha"}
            )
            await backend.upsert(
                id="2", vector=[1.0, 0.0], metadata={"project": "b", "text": "beta"}
            )

            all_hits = await backend.search([1.0, 0.0], top_k=5)
            assert len(all_hits) == 2

            filtered = await backend.search(
                [1.0, 0.0], top_k=5, filter={"project": "a"}
            )
            assert len(filtered) == 1
            assert filtered[0]["id"] == "1"
            await backend.disconnect()

        run(go())


class TestConfigWiring:
    def test_project_comes_from_env(self, monkeypatch):
        monkeypatch.setenv("GONEBY_PROJECT", "from-env")
        assert GoneByConfig.from_env().project == "from-env"

    def test_project_defaults_to_default(self, monkeypatch):
        monkeypatch.delenv("GONEBY_PROJECT", raising=False)
        assert GoneByConfig.from_env().project == "default"
