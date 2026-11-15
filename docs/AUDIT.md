# Audit Ledger — ctx comparison claims → fixes → tests

Every claim from gottz.de's "ctx und GoneBy" comparison (2026-11-15),
verified against commit `a4e326e` and resolved by 2026-11-14 on branch
`fix/ctx-audit-1.1.0`. All nine catalog rows were **true at that commit**;
none were rebutted — all were fixed or, where the promise was wrong, the
docs now say what the code does.

| # | Article claim (Promise) | Resolution | Fix commit | Test(s) |
|---|------------------------|------------|-----------|---------|
| 1 | "Nothing sensitive in plaintext anywhere" | **Both.** Code: project-scoped IDs (#3). Docs: the vector store holds plaintext *by design*; the README/OVERVIEW now state the real threat model (D1). | `2aaca27` | doc-only; scoping covered by #3 |
| 2 | "The judge fails open" | Code fixed: short and empty verdict lists now pad to KEEP with a warning; walks report `judge_fallback_batches`. Same bug in the standalone judge (not in the article) fixed in parity. | `2eb48b2` | `test_short_verdict_keeps_unjudged_entries`, `test_empty_verdict_keeps_whole_batch`, `test_short_json_keeps_unjudged_entries`, `test_empty_json_keeps_all` |
| 3 | "Projects are separate" | Code fixed: `content_id(project, text)`; identical text in two projects are two entries. | `9ce34e1` | `test_same_text_in_two_projects_stays_findable`, `test_content_id_is_deterministic_and_project_scoped` |
| 4 | "Damaged entries are noticed" | Code fixed: v2 WAL lines bind file name + dir token + sequence via AEAD AAD; [GAP]/out-of-order/splice markers, and markers never reach the judge. Wrong key now says "wrong key" (`.keyid` sidecar). Trailing deletion still undetectable — documented honestly. | `1b675c1` | `TestV2Framing`, `TestTamperDetection` (gap, reorder, foreign splice, same-name-different-dir splice, legacy unbound), `TestKeyId` (2) |
| 5 | "No provider hard-coded" | Code fixed (MCP server): `GONEBY_LLM_*` config, OpenAI-compatible default via declared httpx, Anthropic as declared opt-in extra; unconfigured ⇒ actionable `stage=judge` error. Hermes plugin path was already host-LLM. | `cf41525` | `TestLLMProviderConfig` (5) |
| 6a | "The master switch applies everywhere" (tool) | Code fixed: `wall_append` honors the capture verdict it already computed; reads stay ungated. `store_memory` is deliberately NOT gated: it is an explicit post-walk save of already-judged evidence, not a recording hook — the switch is a capture gate (README §Capture policy), and blocking saves would strand walk results when a toggle lands mid-session. | `3c30829` | `TestCaptureGate` (2) |
| 6b | "The master switch applies everywhere" (CLI vs plugin paths) | Code fixed: one shared default `~/.goneby` for CLI/MCP/plugin; `GONEBY_DATA_DIR` for profile-private stores. Live hosts must run `scripts/migrate_wal.sh` (dry-run default) before switching. | `c5389c6` | `TestWalPathUnification` (3), inverted `test_data_paths_default_to_cli_dir_not_profile_private` |
| 7 | "Recording without duplicates" | Code fixed: message identity stored in WAL metadata; `_seen` lazily seeded per WAL file; restart no longer re-captures. | `c5389c6` | `TestRestartDedup` |
| 8 | "Qdrant for larger stocks" | Code fixed: `query_points` (the pinned client has no `.search`), plus a live-server discovery: non-UUID string point IDs are rejected server-side → UUIDv5 mapping with `_goneby_id` payload recovery. Live integration tests for BOTH backends now exist. | `54b852c` | `TestSearchUsesModernApi` (3), `TestLiveQdrant`, `TestLiveValkey` |
| 9 | "Similar entries are merged" | Code: dedup now ignores trailing sentence punctuation (Latin+CJK); docs state exactly what merges and that semantic near-dupes do NOT. | `76afa9f`, `2aaca27` | `test_trailing_punctuation_does_not_split_entries` |
| 10 | "no measurement of the selection step" (Method) | Harness shipped: labeled golden fixture, precision/recall/F1 + variance, `--self-test` without network, `docs/BASELINE.md` recording format. First real row pending a configured judge endpoint. | `a3ee531` | `scripts/eval_walk.py --self-test` |

## Found beyond the article

| Defect | Fix commit | Test |
|--------|-----------|------|
| Standalone `LLMJudge` padded short verdicts with `False` too | `2eb48b2` | `TestLLMJudge` additions |
| Dedup used builtin `hash()` — salted per process by PYTHONHASHSEED | `7e0640f` | `test_content_hash_is_stable_across_processes` |
| Qdrant rejects non-UUID string point IDs (whole backend was dead for arbitrary IDs) | `54b852c` | `test_point_id_policy_matches_live_server` |
| `pyproject` misspelled `importmode` → bare `pytest` could not collect `test_e2e` | `54b852c` | suite now passes under both `pytest` and `python -m pytest` |
| Valkey `close()` deprecation warning | (this commit) | live Valkey test runs warning-free |
| MCP `walk_backward` crashed the server on judge exceptions | `cf41525` | `test_walk_without_llm_config_returns_actionable_error` |

## Article claims that were NOT defects

- README's `CORRUPTED` marker claim — verified true at `a4e326e`, unchanged.
- The encryption table's characterization of GoneBy's protection scope —
  accurate; docs now align to it from our side too.
