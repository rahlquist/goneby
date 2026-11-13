# GoneBy — Overview: How It Works, Why, and What You Can Configure

This is the careful documentation: the mechanism, the benefits, and every
option a user has. README.md covers install and reference tables; this
document explains *why the system is shaped this way* and *what you can
change*.

---

## 1 · How it works

GoneBy gives LLM agents durable, searchable project memory **retroactively**.
The entire design follows one principle:

> **Capture is deliberately dumb and free; intelligence is deferred to a
> single retrospective pass.**

Four phases:

### Phase 1 — Capture (zero tokens, zero judgment)

Every agent turn is appended to the project's **write-ahead log (WAL)** at
each `pre_llm_call` boundary — the full conversation history before the host
compacts it, so nothing is lost to context-rewrite. There is no filtering,
no scoring, no LLM call. Cost: disk I/O.

Details that matter:

- **Encrypted at rest.** Each line is one ChaCha20-Poly1305 sealed entry
  with its own random 96-bit nonce. One bad line yields a `CORRUPTED`
  marker; it never hides or leaks the rest. Losing the key means losing
  the WAL — there is no second copy anywhere.
- **Deduplicated by stable identity** (`message_uid`, falling back to a
  role+content hash): the same messages reappear across several hook
  payloads; without this the log fills with repeats.
- **`system` and `tool` roles are skipped** — host scaffolding, not project
  memory, and tool output is most of transcript volume.
- **One WAL per project** (`~/.goneby/wal/<project>.wal`). This is
  load-bearing: a shared log would force every retrospective walk to judge
  project B's evidence against project A's product.

### Phase 2 — Filter: the backward walk (the core idea)

At project close, GoneBy walks the log **backward from the shipped product**
and asks of each entry: *"did this contribute to what I can already see?"*

This inverts the standard memory-system question. Every other approach asks
*what will matter?* (prediction) while the project is still in flight and
judgment is hardest. GoneBy asks *what did matter?* (verification) after the
artifact exists — verification is cheap, prediction is not. No existing
memory system (Mem0, MemGPT, Cognee, LangGraph, Letta) does this.

- The judge is your **active LLM**, borrowed via the host's structured
  completion API — no extra API key, no provider hardcoded.
- The judge **fails open**: an unparseable verdict means "keep everything".
  A bad judgment day costs storage, never evidence.
- Duplicate/near-duplicate survivors are deduplicated during the walk.

### Phase 3 — Store

Survivors are embedded and upserted into a vector store:

- Embeddings come from any OpenAI-compatible endpoint. The verified default
  deployment: **Qwen3-Embedding-0.6B** served locally by llama.cpp, Matryoshka
  (MRL) truncation from 1024 to **512 dimensions**, re-normalized after
  truncation so stored vectors are unit-length and cosine is a dot product.
- **Entry IDs are content hashes** — re-indexing the same material upserts
  instead of duplicating, which makes the whole pipeline idempotent and safe
  to re-run.
- A `project` metadata tag rides on every record, so one shared store still
  separates projects cleanly.

### Phase 4 — Retrieve

Semantic search over stored memories. **Project-scoped by default**, with an
explicit `scope="all"` to search across every project — isolation is the
sensible default, cross-project recall ("how did I solve this in the other
repo?") is opt-in because it is genuinely useful and rarely the intent.

### Project identity (automatic)

Sessions are resolved to projects through the **public plugin context**
(`ctx.session_cwd()`, bound by the host to the calling session) plus the
plugin's own nearest-`.git`-ancestor walk — no Hermes internals are read.
The WAL slug is the real repo name, and every tool call, capture, and
scoped search agrees without configuration. Unknown or scopeless sessions
fall back to `GONEBY_PROJECT`.

---

## 2 · Benefits

| Benefit | Why it holds |
|---|---|
| **Zero token cost during work** | Capture is file I/O only. No summarization passes, no "memory manager" LLM calls interrupting the project. |
| **Verified, not guessed, relevance** | The backward walk judges against the shipped artifact — a ground truth that literally does not exist mid-project. |
| **Evidence can't be destroyed by a bad model day** | The judge fails open to keeping everything; encryption failures mark corruption instead of dropping data. |
| **Nothing sensitive in plaintext anywhere** | WAL encrypted per-entry; store scoping keeps memories out of unrelated projects. |
| **Coexists with everything** | Implemented as a *general plugin*, not a memory provider — it never claims or displaces a memory-provider slot (Mem0, Honcho, built-in). At project close it hands your memory system additional semantically searchable data: the verified choices that reached completion. |
| **You decide what it remembers of you** | Default-on capture is fully overridable per project, per gateway, or globally (see §3). |
| **Re-runnable and auditable** | Content-hash IDs make every phase idempotent; the WAL remains the full raw record even after selective storage. |
| **Portable** | Same engine behind Hermes plugin tools, `/goneby` chat commands, an MCP server, and a CLI — any MCP client can use it without Hermes at all. |

---

## 3 · Options — everything a user can set

### 3.1 Capture policy (most specific wins)

Evaluated in this order:

| Priority | Knob | Scope | Set via |
|---|---|---|---|
| 1 | **Master gate** | everywhere, persisted | `/goneby gate on\|off`, `goneby gate on\|off` |
| 2 | **Platform kill** | one gateway (`telegram`, `discord`, `cli`, `desktop`…) | `/goneby platform telegram off`, `goneby platform disable telegram`, `GONEBY_DISABLED_PLATFORMS=telegram,discord` (env: additive per-process, not persisted) |
| 3 | **Project marker** | one repo — `.goneby.json` at the project root; team-visible, travels with the repo | `/goneby enable [path]`, `/goneby disable [path]`, `goneby project enable PATH`, the `goneby_project` tool |
| 4 | **User override** | one project, personal — for repos you don't own or can't write | `goneby project enable --project SLUG`, tool fallback |
| 5 | **Mode** | global posture | `/goneby mode all\|opt-in`, `goneby mode …`, `GONEBY_CAPTURE_MODE` |

Two postures:

- **`all` (default)** — capture everything except what the switches above
  exclude. Zero setup; opt-out style.
- **`opt-in`** — capture nothing except projects explicitly enabled. This is
  the "disabled by default, enable per project" posture, one command away.

Policy persists in `~/.goneby/config.json` (`GONEBY_CONFIG_PATH` relocates
it). `goneby_status` and `/goneby status` always show the *effective* verdict
and the inputs that produced it.

### 3.2 Storage backends

`GONEBY_VECTOR_BACKEND`:

- **`valkey`** (default) — brute-force cosine with metadata filter; runs
  locally on 127.0.0.1, no external service needed beyond valkey itself.
- **`qdrant`** — native payload filtering plus the shared re-check; for
  larger corpora where ANN search matters.
- **`memory`** — in-process; tests and throwaway sessions. No persistence.

### 3.3 Embedder

Any OpenAI-compatible `/v1/embeddings` endpoint:

| Var | Default | Notes |
|---|---|---|
| `GONEBY_EMBEDDING_BASE_URL` | `http://localhost:8081/v1` | e.g. llama.cpp on another port, vLLM, TEI, OpenAI itself |
| `GONEBY_EMBEDDING_MODEL` | `qwen3-embedding-0.6b` | |
| `GONEBY_EMBEDDING_DIMENSIONS` | `512` | MRL truncation target (1024-dim model) |
| `GONEBY_EMBEDDING_API_KEY` | *(none)* | only needed for hosted providers |
| `GONEBY_EMBEDDING_BATCH_SIZE` | `16` | bounds request size |

### 3.4 Plugin behavior

| Var | Default | Effect |
|---|---|---|
| `GONEBY_AUTO_CAPTURE` | `true` | legacy per-process kill switch for automatic capture (the persistent master gate is `/goneby gate`) |
| `GONEBY_AUTO_RECALL` | `false` | inject semantic-search context every turn. Off by default: per-turn search costs latency and the agent can always call `query_memory`. |
| `GONEBY_RECALL_MIN_CHARS` | `12` | ignore tiny turns ("ok", "continue") as recall queries |
| `GONEBY_RECALL_TOP_K` | `3` | injected memories per turn |

### 3.5 Core configuration

| Var | Default | Notes |
|---|---|---|
| `GONEBY_ENCRYPTION_KEY` | **required** | 32-byte hex (`goneby generate-key`). Store in a vault. No copy in this repo, ever. |
| `GONEBY_WAL_PATH` | `~/.goneby/wal` | directory base; files are `<project>.wal` |
| `GONEBY_PROJECT` | `default` | fallback slug when no session context resolves |

### 3.6 Interfaces (same engine, four fronts)

| Front | Who uses it | Surface |
|---|---|---|
| Hermes plugin tools | the agent | `wall_append`, `wall_read`, `walk_backward`, `store_memory`, `query_memory`, `goneby_status`, `goneby_project` |
| `/goneby` chat command | the human, in-session | `status`, `enable/disable [path]`, `gate`, `mode`, `platform` |
| MCP server (`goneby serve`) | any MCP client, no Hermes | same seven tools over MCP |
| CLI | terminal / scripting | `generate-key`, `append`, `read`, `status`, `serve`, `project`, `mode`, `platform`, `gate` |

---

## 4 · Where the claims are tested

- 158 tests in `tests/`: WAL crypto (roundtrip, tamper, corruption markers),
  store/search behavior, scoping isolation, backward-walk judging (including
  fail-open), the full e2e pipeline (junk entries provably never stored), the
  plugin surface against a fake Hermes ctx, session-cwd project resolution, and
  the policy decision table.
- Live verification against the real Qwen3-Embedding-0.6B on llama.cpp
  (2026-11-10): related/unrelated cosine margin 0.67/0.25, correct top hit,
  project isolation proven against real valkey, ciphertext-at-rest proven
  with a live key. The unit suite uses a fake embedder — it validates
  plumbing, never ranking quality; ranking is verified separately, and the
  fake is labeled as such in the code and README.

## 5 · Known limits (honest ones)

- The **key is the vault**: lost key ⇒ unrecoverable WAL. No reset path.
- **MRL 512** is a quality/storage compromise, not a benchmark floor; 1024
  keeps a little more signal if you change `GONEBY_EMBEDDING_DIMENSIONS`
  (store and query must agree).
- The backward walk's quality is your active model's judgment quality —
  mitigated by fail-open, never eliminated.
- Capture at `pre_llm_call` sees what the host passed to the hook; messages
  the host itself dropped before that point are not GoneBy's to keep.
