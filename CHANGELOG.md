# Changelog

## 1.1.1 — 2026-11-17 — profile isolation restored

The Hermes catalog review (NousResearch/hermes-agent#155972) found a
regression against the 1.0.0 listing: v1.1.0 moved the Hermes plugin's data
dir to a shared `~/.goneby`, so two profiles working the same project append
to — and walk — the same log, and one profile's transcript ends up in
another's recall. Hermes profiles are kept apart on purpose.

### Fixed
- **Profile isolation is the plugin default again.** Under Hermes the store is
  `<hermes-home>/plugin-data/goneby` (WAL and policy file), one per profile.
  `GONEBY_DATA_DIR` is the explicit opt-in to a single shared store.
- **The CLI and MCP server resolve the store the same way.** `~/.goneby`
  remains their standalone default; `HERMES_HOME` selects the profile store
  and `GONEBY_DATA_DIR` overrides both — so a CLI run inside a profile shares
  that profile's log instead of splitting it.
- **`scripts/migrate_wal.sh` now carries a v1.1.0 store to the profile store**
  (`~/.goneby` → `${HERMES_HOME:-$HOME/.hermes}/plugin-data/goneby`), still
  dry-run by default, sha256-verified, non-destructive. v1.0.0 hosts need no
  migration: their store is already in the restored location.

Tests: `TestWalPathProfileIsolation` (5) and `TestDefaultDataDir` (4) replace
the unification tests that asserted the shared default.

## 1.1.0 — 2026-11-14 — the ctx-audit release

The comparison page "ctx und GoneBy" (gottz.de/ctx-vs-goneby.html, by the
ctx project) catalogued nine ways GoneBy 1.0.0's documentation outpaced its
code. Every claim was independently re-verified against `a4e326e`, found
true, and closed here — with one test per finding, plus six adjacent
defects found along the way. Claim-by-claim ledger: `docs/AUDIT.md`.

### Fixed — correctness
- **Project-scoped memory IDs.** Identical text stored in two projects no
  longer overwrites each other's entries (`content_id(project, text)`).
- **The judge really fails open.** Short or empty answer lists keep the
  unjudged entries (previously: silent drops), warn, and surface as
  `judge_fallback_batches` in walk results — in the plugin AND the
  standalone judge.
- **Master capture switch applies everywhere.** `wall_append` now honors
  the capture verdict it already computed (previously: discarded it and
  wrote anyway).
- **One WAL for all frontends.** CLI, MCP server, and Hermes plugin now
  share `~/.goneby` by default; `GONEBY_DATA_DIR` makes a store private
  per profile. `scripts/migrate_wal.sh` (dry-run by default, sha256
  verified, non-destructive) moves existing plugin data.
- **Restart-safe recording.** Message identities are stored in WAL
  metadata and `_seen` is seeded per WAL — a restart no longer re-captures
  the same transcript.
- **Deterministic dedup.** SHA-256 replaces the per-process-salted builtin
  `hash()`; dedup also merges entries differing only by trailing sentence
  punctuation (Latin and CJK).
- **Qdrant works.** `search()` → `query_points()` (the method was removed
  in the pinned client — every query died). Live-server testing further
  showed non-UUID string point IDs are rejected outright; IDs now map
  deterministically (UUIDv5) with the original recoverable from payload.
- **No provider hardcoded anywhere.** The MCP server's judge is
  configured (`GONEBY_LLM_PROVIDER/BASE_URL/MODEL/API_KEY`), OpenAI-
  compatible via the already-declared httpx; Anthropic is an opt-in
  `goneby[anthropic]` extra. Unconfigured walks return an actionable
  `stage=judge` error instead of crashing.

### Added — durability & measurement
- **Position-bound WAL (v2 framing).** Lines are `v2:<seq>:<base64>` with
  the file name, a hashed directory token, and the sequence authenticated
  into the AEAD tag. Deletions report `[GAP]`, reorders flag
  `out_of_order` and are excluded from judging, splices from other logs
  fail their tags. Legacy lines stay readable, flagged `integrity:
  unbound`. Key-id sidecars (`.keyid`) call out wrong-key reads as such
  and lay groundwork for rotation. Trailing deletion is not detectable
  without a head-hash chain — documented, not claimed.
- **Live integration tests** for both vector backends against real
  containers (`@pytest.mark.integration`, skipped when unreachable) — the
  default store now has real-server coverage too.
- **Backward-walk selection harness** (`scripts/eval_walk.py`,
  `tests/fixtures/walk_golden.json`, `docs/BASELINE.md`): labeled fixture,
  precision/recall/F1 and verdict variance, no-network `--self-test`.
  First real baseline row requires a configured judge endpoint.

### Docs
- Security/threat model rewritten to the truth: encrypted + position-bound
  transcript log; the searchable vector store holds **plaintext by design**
  (semantic search needs it) and needs store-side access controls.
- "Keeps ~10-20%" and the "tens of thousands" scale figures are labeled
  design targets until `docs/BASELINE.md` has measured rows.
- pytest hygiene: fixed the `importmode` typo (bare `pytest` now collects
  the suite) and registered the integration marker.

Test suite: 172 → 205 passed (+33), all mocked-service tests still hermetic;
`ruff` clean; suite green under both `pytest` and `python -m pytest`.

## 1.0.0 — 2026-11-13

Initial release: encrypted per-entry WAL, Hermes plugin (host-LLM backward
walk, capture/recall hooks, capture policy gates), stdio MCP server, CLI,
Valkey/Qdrant/in-memory vector backends. Listed in the Hermes plugin
catalog 2026-11-13.
