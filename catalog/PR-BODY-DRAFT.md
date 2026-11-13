# PR body — catalog submission for goneby (Hermes plugin catalog)
# Synced to the revision addressing teknium1's review of PR #153868 (pin: <SHA>).

## What this plugin does

**GoneBy — retroactive project memory for LLM agents.** Capture is deliberately dumb and free: every turn appends to an encrypted (ChaCha20-Poly1305) per-project write-ahead log — zero tokens, zero judgment, disk I/O only. At project close, GoneBy walks that log **backward from the shipped product**, using the product as the oracle to *verify* which entries actually contributed (rather than predicting mid-project what might matter), embeds the survivors (Qwen3-Embedding-0.6B or any OpenAI-compatible endpoint; MRL truncation 1024→512), and upserts them into Valkey (default) or Qdrant for semantic search in later sessions. The backward-walk framing is not present in Mem0 / MemGPT / Cognee / Letta — that is the thesis. Project-scoped by default, cross-project recall opt-in, full per-project/per-gateway capture policy including a disable-by-default opt-in mode.

## Hermes surfaces used

- **Hooks:** `pre_llm_call` (per-turn capture, `message_uid` dedup, `system`/`tool` roles skipped; optional recall injection, off by default), `post_compaction` (boundary marker only).
- **Tools (7):** `wall_append`, `wall_read`, `walk_backward`, `store_memory`, `query_memory`, `goneby_status`, `goneby_project` (no path argument on the model surface — opting arbitrary directories into capture is operator-only via `/goneby enable <path>`).
- **Slash command:** `/goneby` (status / enable / disable / gate / mode / platform).
- **Host LLM:** `ctx.llm.complete_structured` for the walk judge — the user's active model and auth, nothing hardcoded. The judge fails open **per batch**: a bad or timed-out verdict keeps that batch, never destroys evidence.
- **Plugin config:** `config_schema` declares the non-secret knobs (`auto_capture`, `auto_recall`, `judge_batch`, `judge_concurrency`, `judge_max_chars`); `GONEBY_*` env vars remain as fallback.
- General plugin only: no memory-provider slot claimed (coexists with Mem0/Honcho/built-in), no middleware, no desktop surface, **no core overrides of any kind**.

## Rule-13 disclosure

- **Network:** HTTP only to **user-configured endpoints** — an OpenAI-compatible `/v1/embeddings` server (`GONEBY_EMBEDDING_BASE_URL`, default `http://localhost:8081/v1`) and the user's Valkey/Qdrant host. No third-party services, no telemetry, no update checks (rule 3: none, by design).
- **Reads outside plugin data:** none since the review revision — project resolution uses the **public** `ctx.session_cwd()` accessor (host-bound to the calling session) plus a pure-filesystem nearest-`.git`-ancestor walk implemented plugin-side; Hermes' `state.db` and its `sessions` table are no longer read at all. Any failure falls back to `GONEBY_PROJECT`.
- **Writes:** `<profile-data>/wal/<project>.wal` (per-entry ciphertext; files created `0600` in a `0700` directory) and `<profile-data>/config.json` (capture-policy state), where `<profile-data>` defaults to `get_hermes_home()/plugin-data/goneby` — per profile, never shared. The plugin never writes keys; the operator supplies `GONEBY_ENCRYPTION_KEY`.
- **Background:** one daemon-thread event loop bridging sync hooks to the async embedder/store; hook bodies are bounded and never raise.
- **Secrets:** `GONEBY_ENCRYPTION_KEY` (declared in `requires_env`) read via `agent.secret_scope.get_secret`, failing closed when multiplexing has no bound scope — and with **no raw `os.environ` fallback** when the scoped lookup returns unset, so one profile can never pick up another profile's key.
- **Dependency graph:** self-contained. The **repository root is the plugin directory** — manifest, plugin code and the `goneby` library (`src/goneby`) ship as one reviewed tree; the root pyproject declares the runtime deps (cryptography, httpx, valkey). No registry dependency, no path sources, nothing to edit after clone; the reviewed pin is simultaneously the library pin and the plugin pin.

## Validation (same command as the CI gate)

At the pinned commit, on Linux:

```
hermes plugins validate <fresh-clone> --install-deps
```

All checks pass, including the **security scan (safe)**, capability probe (register() in isolation), declared-vs-actual tools/hooks/env match, `requires_hermes` floor `>=0.23.0`, and **no core override**. Repo-side: 169 tests, ruff clean.

## Review round 1 — delta from `c7cc997e`

Addresses teknium1's `CHANGES_REQUESTED` review in full: **(1)** self-contained install tree (root-as-plugin-dir; the previous `subdir: plugin` + relative path source could not resolve on real catalog installs — acknowledged, my earlier comment asserting it worked was wrong); **(2)** `get_secret()`→None no longer falls back to `os.environ` (cross-profile leak); **(3)** WAL + policy files default per-profile under `plugin-data` (env overrides win); **(4)** `state.db` resolved from `get_hermes_home()` — no hardcoded `~/.hermes`; **(5)** `goneby_project` lost its model-callable `path`; **(6)** re-pinned. Recommended items also included: non-secret knobs → `config_schema`, WAL files `0600`. The judge-path fixes found by running live traffic (typed input blocks, batched concurrent fail-open judging, newest-first flag alignment) are in the pinned tree.

## Links

- Repo (public, Apache-2.0): https://github.com/rahlquist/goneby
- Mechanism doc: https://github.com/rahlquist/goneby/blob/main/docs/OVERVIEW.md
- Idea/lineage doc: https://github.com/rahlquist/goneby/blob/main/docs/IDEATION.md
- 100 s narrated explainer: https://github.com/rahlquist/goneby/releases/download/v1.0.0/goneby-explainer.mp4

## Maintainer

rahlquist — repository owner (rule 5). Banner and screenshots are frames from our own explainer video, committed at the pinned SHA.
