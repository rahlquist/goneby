"""End-to-end smoke test: exercises the WHOLE configured stack for real.

Unlike the unit suite (fake embedder, in-memory backend), this script hits
the actual services your GONEBY_* config points at and proves the full
chain works: embedder health -> real semantic ranking -> store round trip
through the vector backend -> project scoping -> encrypted WAL round trip.

Usage:
    export GONEBY_ENCRYPTION_KEY=...            # or source your env file
    python scripts/smoke_test.py                # uses defaults from env
    python scripts/smoke_test.py --project smoke-$(date +%s)

Exit code 0 = every stage passed. Any failure prints the stage and detail.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from goneby.config.settings import GoneByConfig, project_wal_path
from goneby.crypto.cipher import Cipher
from goneby.memory.embedder import Embedder, EmbeddingConfig
from goneby.memory.store import MemoryStore
from goneby.vector.factory import create_backend
from goneby.wal.manager import WALManager

# Semantically distinct pairs; a real embedding model MUST rank these apart.
RELATED_A = "qdrant payload filtering versus brute-force cosine for the vector backend choice"
RELATED_B = "how should the project pick its similarity search database backend"
UNRELATED = "chocolate chip cookie recipe with brown butter and sea salt"

PASS = "PASS"
failures: list[str] = []


def report(stage: str, ok: bool, detail: str = "") -> None:
    print(f"[{PASS if ok else 'FAIL'}] {stage}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(stage)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="smoke-test", help="temp project namespace")
    args = parser.parse_args()

    cfg = GoneByConfig.from_env()
    if not cfg.encryption_key:
        print("GONEBY_ENCRYPTION_KEY is not set — export it first.")
        return 2

    # Stage 1: embedder health + native dimensions
    emb_cfg = EmbeddingConfig(
        base_url=cfg.embedding_base_url,
        model=cfg.embedding_model,
        api_key=cfg.embedding_api_key,
        dimensions=cfg.embedding_dimensions,
        batch_size=cfg.embedding_batch_size,
    )
    emb = Embedder(emb_cfg)
    health = await emb.health()
    report("embedder reachable", bool(health.get("ok")), str(health.get("detail", health.get("models", ""))))
    if not health.get("ok"):
        await emb.aclose()
        return 1

    # Stage 2: real semantic ranking (the quality check the unit tests fake)
    va = await emb.embed_one(RELATED_A)
    vb = await emb.embed_one(RELATED_B)
    vc = await emb.embed_one(UNRELATED)
    report("client dimensions", len(va) == cfg.embedding_dimensions, f"got {len(va)}")
    cos_rel = sum(x * y for x, y in zip(va, vb))
    cos_unrel = sum(x * y for x, y in zip(va, vc))
    report(
        "semantic ranking sane",
        cos_rel - cos_unrel > 0.15,
        f"related={cos_rel:.3f} unrelated={cos_unrel:.3f} (margin must exceed 0.15)",
    )

    # Stage 3: vector store round trip with project scoping
    other = f"{args.project}-other"
    backend = create_backend(cfg)
    store = MemoryStore(emb, backend, project=args.project)
    indexed_ids: list[str] = []
    try:
        from goneby.memory.store import content_id

        indexed_ids = [content_id(t) for t in (RELATED_A, UNRELATED, RELATED_B)]
        await store.index_entries(
            [
                {"content": RELATED_A, "metadata": {"type": "decision"}},
                {"content": UNRELATED, "metadata": {"type": "note"}},
            ]
        )
        await MemoryStore(emb, backend, project=other).index_entries(
            [{"content": RELATED_B, "metadata": {"type": "note"}}]
        )
        hits = await store.search("choosing the vector database backend", top_k=5)
        results = hits.get("results", [])
        report("store round trip", hits.get("status") == "ok" and bool(results))
        report(
            "correct top hit",
            bool(results) and results[0]["text"] == RELATED_A,
            f"top score={results[0]['score'] if results else '-'}",
        )
        report(
            "project scoping holds",
            all(r["project"] == args.project for r in results),
        )
        scoped_all = await store.search("choosing the vector database backend", top_k=5, scope="all")
        report(
            "cross-project scope works",
            any(r["project"] == other for r in scoped_all.get("results", [])),
        )
    except Exception as exc:  # noqa: BLE001
        report("vector backend reachable", False, f"{type(exc).__name__}: {exc}")
    finally:
        try:
            for entry_id in indexed_ids:  # leave the store as we found it
                await backend.delete(entry_id)
            await store.disconnect()
        except Exception:  # noqa: BLE001
            pass

    # Stage 4: encrypted WAL round trip (throwaway project slug, cleaned up)
    try:
        secret = f"SMOKE-MARKER-{os.getpid()}"
        wal_path = project_wal_path(cfg.wal_path, args.project)
        wal = WALManager(wal_path, Cipher(cfg.encryption_key))
        wal.append(secret, {"type": "verification"})
        raw = wal_path.read_bytes()
        back = [e["content"] for e in wal.read_all()]
        report("WAL round trip", back[-1] == secret)
        report("WAL encrypted at rest", secret.encode() not in raw)
        wal.clear()
    except Exception as exc:  # noqa: BLE001
        report("WAL usable", False, f"{type(exc).__name__}: {exc}")

    await emb.aclose()

    print()
    if failures:
        print(f"SMOKE TEST FAILED at: {', '.join(failures)}")
        return 1
    print("SMOKE TEST PASSED — embedder, ranking, store, scoping, and WAL are all live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
