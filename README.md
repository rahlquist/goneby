# GoneBy — Retroactive Project Memory for LLM Agents

GoneBy is an MCP server that captures LLM project output between context compactions, then retroactively filters and stores relevant memories using the shipped product as ground truth.

## How It Works

1. **Capture (WAL):** During a project, append output to an encrypted Write-Ahead Log at each context compaction boundary. No filtering, no LLM calls — just append.
2. **Filter (Backward Walk):** At project close, walk backward through the WAL using the shipped product as the oracle to determine which entries were actually relevant.
3. **Store (Embed + Vector DB):** Embed the surviving entries and upsert them into Valkey (default) or Qdrant.
4. **Retrieve:** Future sessions query the store semantically.

The capture phase costs zero tokens. All inference is deferred to the single backward-walk pass and the embedding call at store time.

## Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| Python | 3.11+ | |
| Encryption key | 32 bytes | `goneby generate-key` |
| Embedding server | Any OpenAI-compatible `/v1/embeddings` | llama.cpp recommended |
| Vector store | Valkey (default) or Qdrant | |

### Embedding server

GoneBy talks to any server exposing `POST /v1/embeddings`. Recommended: **Qwen3-Embedding-0.6B** via llama.cpp, running locally.

```bash
# Start the embedding server (CPU is fine for this model size)
llama-server \
  -m qwen3-embedding-0.6b.gguf \
  --embeddings \
  --pooling cls \
  --port 8081 \
  --ctx-size 8192
```

Verify:

```bash
curl http://localhost:8081/v1/models
goneby status
```

**MRL (Matryoshka truncation).** The model emits 1024 dimensions. GoneBy truncates to
`GONEBY_EMBEDDING_DIMENSIONS` (default 512) and renormalizes, halving storage while
retaining ~97% of retrieval quality. Raise it to 1024 for maximum fidelity; drop to 256
for large stores. Truncation happens before normalization so the stored vectors are
unit-length and cosine similarity is a plain dot product.

## Quick Start

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[mcp,valkey,qdrant]"

export GONEBY_ENCRYPTION_KEY=$(.venv/bin/goneby generate-key | head -1 | awk '{print $NF}')
goneby status
goneby serve
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GONEBY_ENCRYPTION_KEY` | (required) | 32-byte hex key for ChaCha20-Poly1305 |
| `GONEBY_WAL_PATH` | `~/.goneby/wal` | Path to the encrypted WAL |
| `GONEBY_PROJECT` | `default` | Project namespace for stored memories |
| `GONEBY_VECTOR_BACKEND` | `valkey` | `valkey` or `qdrant` |
| `GONEBY_VALKEY_HOST` / `_PORT` / `_DB` | `localhost` / `6379` / `0` | Valkey connection |
| `GONEBY_QDRANT_HOST` / `_PORT` | `localhost` / `6333` | Qdrant connection |
| `GONEBY_QDRANT_COLLECTION` | `goneby` | Qdrant collection |
| `GONEBY_EMBEDDING_BASE_URL` | `http://localhost:8081/v1` | Embedding server |
| `GONEBY_EMBEDDING_MODEL` | `qwen3-embedding-0.6b` | Model name sent to the server |
| `GONEBY_EMBEDDING_API_KEY` | (none) | Bearer token, for hosted embedders |
| `GONEBY_EMBEDDING_DIMENSIONS` | `512` | MRL truncation target |
| `GONEBY_EMBEDDING_BATCH_SIZE` | `16` | Texts per request |

## MCP Tools

| Tool | Purpose |
|------|---------|
| `wall_append(content, entry_type)` | Append to the encrypted WAL — call at each compaction. Zero token cost. |
| `wall_read()` | Review all captured entries |
| `walk_backward(mvp_text)` | Filter the WAL against the shipped product. Returns kept entries + stats. |
| `store_memory(entries)` | Embed and upsert walked entries. Idempotent — IDs are content-derived. |
| `query_memory(query, top_k, min_score)` | Semantic search over stored memories |
| `generate_key()` | Generate a new encryption key |
| `status()` | Config, WAL stats, and a live health probe of both backends |

### Typical session

```
wall_append(<output since last compaction>, entry_type="note")
... work continues ...
walk_backward(<shipped MVP description>)   # at project close
store_memory(<kept_entries>)
query_memory(<question in a later session>)
```

## Hermes plugin

GoneBy ships a Hermes plugin that makes capture and recall **automatic** — the agent does not have to remember to call anything.

```
~/.hermes/plugins/goneby/
├── plugin.yaml      # manifest: declares tools + hooks
├── __init__.py      # register(ctx)
├── adapter.py       # tools, hooks, capture/recall logic
└── pyproject.toml   # declares the goneby dependency for PM
```

Install by copying the `plugin/` directory to `~/.hermes/plugins/goneby/`, then:

```bash
hermes plugins validate ~/.hermes/plugins/goneby   # expect overall ok: True
hermes plugins enable goneby
```

### What the hooks do

| Hook | Behavior |
|------|----------|
| `pre_llm_call` | Captures messages not yet seen into this project's WAL (skips `system`/`tool` roles). Also performs recall when `GONEBY_AUTO_RECALL=true`. |
| `post_compaction` | Records a compaction-boundary marker so the WAL reflects the project's shape. |

Capture is per-turn rather than per-compaction: `post_compaction` fires *after* the transcript is rewritten, so the dropped messages are already gone. `pre_llm_call` hands us the full `conversation_history` before that happens, which is a superset and loses nothing.

Capture is idempotent — messages are keyed on `message_uid` when present, else a role+content hash — so a turn that appears in several `pre_llm_call` payloads is stored once.

### Configuration (plugin)

| Variable | Default | Description |
|----------|---------|-------------|
| `GONEBY_AUTO_CAPTURE` | `true` | Capture turns into the WAL automatically |
| `GONEBY_AUTO_RECALL` | `false` | Inject relevant memories into every turn |
| `GONEBY_RECALL_MIN_CHARS` | `12` | Skip recall for turns shorter than this |
| `GONEBY_RECALL_TOP_K` | `3` | Memories injected per turn when recall is on |

`GONEBY_AUTO_RECALL` defaults **off** because a semantic search on every turn costs latency; the agent already has `query_memory`.

### Project isolation

Each project writes its own WAL (`~/.goneby/wal/<project>.wal`), so a backward walk for one project never sees another's evidence. Retrieval is scoped by default; pass `scope="all"` for deliberate cross-project lookup.

The project name comes from `GONEBY_PROJECT`. To key on the working directory instead, point it at a stable slug — the plugin flattens paths safely.

### Tools

The same six tools the MCP server exposes, registered as first-class Hermes tools: `wall_append`, `wall_read`, `walk_backward`, `store_memory`, `query_memory`, `goneby_status`.

`walk_backward` judges relevance through `ctx.llm.complete_structured` — it borrows the user's active model and auth, so there is no separate API key and no hardcoded provider.

## Security

- WAL entries are encrypted independently with ChaCha20-Poly1305 and a random 96-bit nonce, then base64-encoded one per line. Append-only — existing entries are never rewritten.
- A wrong key or damaged line yields a `CORRUPTED` marker rather than aborting the read or leaking plaintext.
- **Losing the key means losing the WAL.** Store it in Bitwarden or the Hermes vault, not in the repo.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    LLM Agent (Hermes)                         │
│   context full → wall_append() → context compacted            │
└───────────────────────────┬──────────────────────────────────┘
                            │ zero token cost
                            ▼
              ┌──────────────────────────────┐
              │  Encrypted WAL (ChaCha20)    │  append-only
              └──────────────┬───────────────┘
                             │ at project close
                             ▼
   Shipped MVP ──► Backward Walk ──► LLM Judge ──► Deduplicate
   (oracle)         newest → oldest
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Qwen3-Embedding-0.6B         │  MRL → 512 dims
              │ llama.cpp :8081              │  local, no API cost
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │ Valkey (default) | Qdrant    │  cosine search
              └──────────────────────────────┘
```

Full diagram: `docs/architecture.html` (open in any browser).

## Development

```bash
uv pip install --python .venv/bin/python -e ".[mcp,valkey,qdrant,dev]"
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m ruff check .
```

The test suite runs a fake OpenAI-compatible embedding server, so no model or vector
database is required. The fake is **not** semantic — it validates plumbing (batch
ordering, MRL truncation, normalization, round trips, idempotency), not retrieval
quality. Verify ranking against a real model before trusting it.

`scripts/live_check.py` exercises the embedder plumbing (batching, MRL,
normalization, round trips) against a **fake** vector server on :8081 —
it validates transport, not ranking quality.

## Live Deployment (this host, 2026-11-10)

| Piece | Where |
|---|---|
| Embedder | `llama-server` b12261 (CPU) at `~/.local/opt/llama.cpp/`, serving `~/models/Qwen3-Embedding-0.6B-Q8_0.gguf` on `127.0.0.1:8091` |
| Services | systemd **user** units `goneby-embedder` and `goneby-valkey` (valkey on `127.0.0.1:6379`, data in `~/.goneby/valkey`) |
| Config | `GONEBY_ENCRYPTION_KEY` and `GONEBY_EMBEDDING_BASE_URL` live in `~/.hermes/.env` (mode 600); the key is recorded nowhere else |
| Verified | real-model ranking: related/unrelated cosine 0.67/0.25; top hit 0.79; project-scoped search isolation proven; WAL ciphertext-at-rest proven; `goneby status` reports embedder UP |

Manage with `systemctl --user {status,restart} goneby-embedder goneby-valkey`.
Logs: `journalctl --user -u goneby-embedder`.

## License

Apache 2.0
