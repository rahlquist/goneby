# GoneBy — Full Stack Setup from Zero

This is the complete path from nothing-installed to a working GoneBy.
The library alone does almost nothing — GoneBy needs three services wired
together: an **embedding server**, a **vector store**, and (for the WAL) just
a directory and an **encryption key**. Every command below is generic —
adjust only the paths in angle brackets.

```
LLM agent ──capture──▶ encrypted WAL ──backward walk──▶ embed ──▶ vector store
                                                            ▲
                                     future sessions ──query┘
```

| Component | Required | What it is |
|---|---|---|
| Python 3.11+ | yes | the library, CLI, MCP server, plugin |
| Embedding server | yes | any OpenAI-compatible `/v1/embeddings` |
| Vector store | yes | Valkey (default) or Qdrant |
| Encryption key | yes | 32 random bytes — no recovery path if lost |
| Hermes | no | only for the plugin surface; MCP/CLI work standalone |

---

## 1 · Python library

```bash
git clone https://github.com/rahlquist/goneby.git <your-checkout>
cd <your-checkout>
uv venv .venv
uv pip install --python .venv/bin/python -e ".[mcp,valkey]"    # or [qdrant]
.venv/bin/goneby --help
```

Plain `pip install -e ".[mcp,valkey]"` works the same without uv.

## 2 · Encryption key

```bash
.venv/bin/goneby generate-key
```

Pick ONE home for the printed 64-hex value and treat it as a password:

- CLI/MCP use: `export GONEBY_ENCRYPTION_KEY=...` in your shell or env file.
- Hermes gateway use: add it to `~/.hermes/.env` (Hermes' env file, mode 600).
- Vault: store the same value in Bitwarden/1Password as a fallback record.

The WAL is only decryptable with the key that wrote it. No reset, no
recovery. The key never belongs in the repo or any other plaintext file.

## 3 · Embedding server (llama.cpp + Qwen3-Embedding)

GoneBy talks to any OpenAI-compatible embeddings endpoint. The reference
stack is llama.cpp serving Qwen3-Embedding-0.6B — small, fast on CPU, and
Matryoshka-trained so GoneBy can truncate 1024→512 dims client-side.

**3a. Get `llama-server`.** Either grab a prebuilt release — go to
<https://github.com/ggml-org/llama.cpp/releases>, open the newest `b####`
release, download the **`llama-b<NNN>-bin-ubuntu-x64.tar.gz`** asset (CPU;
there are CUDA/ROCm/Vulkan variants for GPUs) — or build:

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build -S llama.cpp
cmake --build llama.cpp/build --config Release -j --target llama-server
```

Note: prebuilt tarballs unpack FLAT — `llama-server` sits at the top of the
extracted directory next to the `.so` files, not in a `bin/` subdir. Put it
somewhere stable, e.g. `~/llama.cpp/`. (Brand-new releases sometimes finish
uploading their assets hours late — if a just-published `b####` 404s, take
the previous one.)

**3b. Get the model (~640 MB):**

```bash
mkdir -p ~/models
curl -L --fail -o ~/models/Qwen3-Embedding-0.6B-Q8_0.gguf \
  "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf"
```

**3c. Serve it:**

```bash
~/llama.cpp/llama-server \
  -m ~/models/Qwen3-Embedding-0.6B-Q8_0.gguf \
  --embedding --host 127.0.0.1 --port 8081 -c 4096
```

`--embedding` is the load-time flag; no `--pooling` override is needed —
the pooling type ships in the GGUF metadata and llama-server applies it.
Keep it bound to `127.0.0.1`. CPU-only is genuinely fine at this model size.

**3d. Verify:**

```bash
curl http://127.0.0.1:8081/health          # -> {"status":"ok"}
curl http://127.0.0.1:8081/v1/models       # -> lists your model
curl -s http://127.0.0.1:8081/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":["smoke"],"model":"qwen3-embedding-0.6b"}' | head -c 200
# each vector should carry 1024 floats — GoneBy truncates to 512 itself
```

To run it persistently, use whatever supervisor you already use — a systemd
user unit example is in README §Deployment guidance.

## 4 · Vector store

### Valkey (default)

Install via your distro package (`valkey-server`), or containers:

```bash
docker run -d --name goneby-valkey -p 127.0.0.1:6379:6379 valkey/valkey:8
```

Or run the binary directly (this is the reference invocation):

```bash
mkdir -p ~/.goneby/valkey
valkey-server --port 6379 --bind 127.0.0.1 --protected-mode yes \
  --dir ~/.goneby/valkey --save "900 1 300 10 60 10000"
```

Verify: `valkey-cli -h 127.0.0.1 ping` → `PONG`.
GoneBy stores each vector as a hash under `goneby:<content-hash>` with
cosine scoring and a `project` metadata tag; RDB snapshots (the `--save`
schedule above) are enough for durability at this scale.

### Qdrant (alternative)

```bash
docker run -d -p 127.0.0.1:6333:6333 qdrant/qdrant
```

Then `export GONEBY_VECTOR_BACKEND=qdrant` — host/port/collection defaults to
`localhost:6333` / collection `goneby`. Qdrant gives native payload filtering;
the valkey backend does the same filtering in-process (fine for thousands to
tens of thousands of entries).

## 5 · Point GoneBy at the stack

Everything is environment variables (full table: README §Configuration). A
minimal complete wiring for a non-Hermes install:

```bash
export GONEBY_ENCRYPTION_KEY=<from step 2>
export GONEBY_EMBEDDING_BASE_URL=http://localhost:8081/v1
# vector backend + WAL path defaults are already correct
```

Hermes gateway users: put the same lines in `~/.hermes/.env`.

```bash
.venv/bin/goneby status     # shows config + a live probe of the embedder
```

## 6 · Smoke-test the whole chain

The one command that proves every stage is actually working — with a REAL
embedding model this time, not the unit suite's fake:

```bash
.venv/bin/python scripts/smoke_test.py
```

It checks nine things: embedder reachable → dimensions → **semantic ranking
sane** (real model required to rank related/unrelated texts apart) → store
round trip → correct top hit → project scoping → cross-project scope → WAL
round trip → ciphertext at rest. It cleans up everything it writes. Exit 0
means the stack is genuinely operational. If any stage fails, see
§9 Troubleshooting.

## 7 · Use it

**MCP server (any MCP client, no Hermes required):**

```json
{
  "mcpServers": {
    "goneby": {
      "command": "<your-checkout>/.venv/bin/goneby",
      "args": ["serve"],
      "env": {
        "GONEBY_ENCRYPTION_KEY": "...",
        "GONEBY_EMBEDDING_BASE_URL": "http://localhost:8081/v1"
      }
    }
  }
}
```

Typical flow: `wall_append` as work happens → at project close
`walk_backward` with a description of the shipped product → `store_memory`
the kept entries → `query_memory` from future sessions.

**CLI:** `goneby append "..."` / `goneby read` / `goneby status` cover the
WAL; the full pipeline lives behind the MCP/plugin tools.

**Hermes plugin (automatic capture every turn — nothing to remember):**

The repository root IS the plugin directory: the reviewed tree carries the
manifest, the plugin code and the library together, so any install source —
catalog, git URL, or pinned `--ref` — is self-contained. No path editing.

1. Install from the catalog once it merges (`hermes plugins install goneby`),
   or straight from the repo today:
   `hermes plugins install https://github.com/rahlquist/goneby --ref <FULL_SHA>`.
2. `hermes plugins enable goneby`, then **restart the hosting process** —
   installed dependencies only load into a fresh process. Note: if your chat
   runs on the dashboard (desktop app), restart `hermes-dashboard`, not just
   the gateway — tool dispatch is bound to the hosting process.
3. `/goneby status` in any session to see the effective capture verdict.

Optional tuning without env vars — non-secret knobs live in plugin config:
`plugins.entries.goneby.settings.auto_capture/auto_recall/judge_batch/judge_concurrency/judge_max_chars`
(the `GONEBY_*` env vars remain as a fallback). Under Hermes, runtime data
defaults to `~/.hermes/plugin-data/goneby/` per profile (WAL + policy file);
override with `GONEBY_DATA_DIR`/`GONEBY_WAL_PATH`.
(If a capability probe says `No module named 'goneby'` right after enabling,
that is the documented transitional state — restart and validate again.)

With the plugin, capture is per-turn and per-project automatically (the WAL
slug comes from the session's repo); `walk_backward`/`store_memory`/
`query_memory` are tools the agent can call.

## 8 · Day-2 controls

Everything about *what gets captured* is policy-tunable — master gate,
per-gateway platform kills, per-project `.goneby.json` markers, and a
disable-by-default `opt-in` mode. Full table: README §Capture policy.

## 9 · Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Enable refuses: "goneby was not found in the package registry" | Pre-self-contained-tree install (an old copy without the library at its root). Reinstall from the repo root: `hermes plugins install https://github.com/rahlquist/goneby`. |
| Probe says `No module named 'goneby'` after a successful enable | Transitional: the gateway is still running the old dependency environment. Restart the gateway. |
| `connection refused` / embedder unreachable | The llama-server unit isn't running or is bound elsewhere — `curl http://127.0.0.1:8081/health`. |
| Smoke: `vector backend reachable` fails | Valkey not running (`valkey-cli ping`) or wrong `GONEBY_VALKEY_PORT`/backend. |
| Smoke: `semantic ranking sane` fails | Your embedder is serving a chat model or non-MRL embedding. Use the Qwen3 GGUF above, or another verified embeddings endpoint. |
| WAL reads show `CORRUPTED` entries | Key mismatch — that WAL was written with a different `GONEBY_ENCRYPTION_KEY`, or lines were tampered with. Entries encrypted under other keys can never be decrypted. |
| Plugin enabled but nothing lands in `~/.goneby/wal/` | Check `/goneby status` — it prints the effective verdict and which control (gate / platform kill / marker / opt-in mode) produced it. Also confirm the gateway was restarted. |
| `goneby status` raises `IsADirectoryError` | You're on a pre-fix checkout — pull latest. |

---

**That's the whole stack.** If `smoke_test.py` exits 0, GoneBy is working on
your machine exactly the way it works on the reference deployment.
