"""Throwaway live check: real llama-server-style endpoint on :8081."""
import asyncio
import sys

import uvicorn
from fastapi import FastAPI, HTTPException, Request

sys.path.insert(0, "/home/rahlquist/goneby/tests")
from test_memory import fake_vector

from goneby.memory.embedder import Embedder, EmbeddingConfig
from goneby.memory.store import MemoryStore

DIM = 1024

app = FastAPI()


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "qwen3-embedding-0.6b"}]}


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    if not body.get("input"):
        raise HTTPException(status_code=400, detail="empty input")
    return {
        "object": "list",
        "model": "qwen3-embedding-0.6b",
        "data": [
            {"object": "embedding", "index": i, "embedding": fake_vector(t, DIM)}
            for i, t in enumerate(body["input"])
        ],
    }


async def main():
    cfg = uvicorn.Config(app, host="127.0.0.1", port=8081, log_level="error")
    server = uvicorn.Server(cfg)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    print("=== live embedder on :8081 ===\n")

    emb = Embedder(EmbeddingConfig(base_url="http://127.0.0.1:8081/v1", dimensions=512))
    health = await emb.health()
    print("health:", health)

    v = await emb.embed_one("qdrant vector backend choice")
    print(f"embed_one -> {len(v)} dims, norm={sum(x*x for x in v)**0.5:.6f}")

    batch = await emb.embed_batched([f"entry number {i}" for i in range(7)])
    print(f"embed_batched(7) -> {len(batch)} vectors, each {len(batch[0])} dims")

    # Full store round trip against the in-memory backend
    from test_memory import MemBackend
    store = MemoryStore(emb, MemBackend(), project="live-check")
    res = await store.index_entries([
        {"content": "chose qdrant for the vector backend because it scales past 1M vectors",
         "metadata": {"type": "decision"}},
        {"content": "wal is append only and never rewritten", "metadata": {"type": "finding"}},
        {"content": "thought about making a subscription todo app", "metadata": {"type": "note"}},
    ])
    print("index:", res)

    hits = await store.search("why did we pick the vector database", top_k=2)
    for h in hits["results"]:
        print(f"  hit {h['score']}: {h['text'][:70]}")

    await emb.aclose()
    server.should_exit = True
    await task


asyncio.run(main())
