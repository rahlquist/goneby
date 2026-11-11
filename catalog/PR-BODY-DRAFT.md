# PR body draft — plugin-catalog submission for goneby
# (fill <SHA>/pins after PyPI publish + path-source removal; re-run the duplicate search immediately before gh pr create)

## What this plugin does

GoneBy — retroactive project memory for LLM agents. Capture is deliberately
dumb and free: every turn appends to an encrypted (ChaCha20-Poly1305)
per-project write-ahead log, zero tokens, zero judgment. At project close it
walks the log **backward from the shipped product**, using the product as the
oracle to verify which entries actually contributed; survivors are embedded
(Qwen3-Embedding-0.6B via any OpenAI-compatible endpoint) and upserted into
Valkey (default) or Qdrant for semantic search in later sessions. The
backward-walk framing (verification, not prediction) is not present in
Mem0/MemGPT/Cognee/Letta.

## Hermes surfaces used

- **Hooks:** `pre_llm_call` (per-turn capture with `message_uid` dedup,
  `system`/`tool` roles skipped; optional recall injection), `post_compaction`
  (boundary marker only).
- **Tools (7):** `wall_append`, `wall_read`, `walk_backward`, `store_memory`,
  `query_memory`, `goneby_status`, `goneby_project`.
- **Slash command:** `/goneby` (status / enable / disable / gate / mode /
  platform — capture-policy control).
- **Host LLM:** `ctx.llm.complete_structured` for the walk judge — the user's
  active model and auth, no provider hardcoded.
- General plugin only. No memory-provider slot claimed (coexists with
  Mem0/Honcho/built-in), no core overrides, no middleware, no desktop surface
  today.

## Rule-13 disclosure

- **Network:** HTTP calls **only to user-configured endpoints** — an
  OpenAI-compatible `/v1/embeddings` server (`GONEBY_EMBEDDING_BASE_URL`,
  default `http://localhost:8091/v1`… default is localhost) and the
  user's Valkey/Qdrant host. No third-party calls, no telemetry, no update
  checks.
- **Reads outside plugin data:** one **read-only** SQLite query of Hermes'
  `state.db` (`SELECT cwd, git_repo_root FROM sessions WHERE id=?`) to
  namespace the WAL per project; opened `mode=ro`, never writes, failure
  falls back to `GONEBY_PROJECT`. No other Hermes internals read.
- **Disk writes:** `~/.goneby/wal/<project>.wal` (ciphertext per entry) and
  `~/.goneby/config.json` (policy state). Keys are never written by the
  plugin — the operator supplies `GONEBY_ENCRYPTION_KEY`.
- **Background process:** one daemon-thread event loop for async embedder/
  store work; bounded hook bodies (never blocks a turn; default recall off).
- **Secrets:** `GONEBY_ENCRYPTION_KEY` read via `agent.secret_scope.get_secret`
(fail-closed under multiplexing).
- **Dependency graph:** the plugin depends on the `goneby` library via a
relative `tool.uv.sources` path inside this same repo — no registry, no
network fetch at lock time; the review pin covers library + plugin as one
tree. Clean-room verified (fresh container, PM-style rewrite): uv locks
15 packages, builds, imports.
- **Self-update:** none. Updates via SHA-bump PR like everyone else.

## Validation

`hermes plugins validate <clean clone>/plugin --install-deps` → all checks
pass, security scan `safe`, capability probe green, declared tools/hooks/
env match registrations. Repo: 158 tests, ruff clean, MIT/Apache-2.0 per
LICENSE (Apache-2.0). Docs page:
https://github.com/rahlquist/goneby/blob/main/docs/OVERVIEW.md
Video (100 s narrated explainer): release asset on v1.0.0.

## Maintainer

rahlquist — repository owner (rule 5).
