# Backward-Walk Selection Baseline

The ctx comparison (gottz.de, 2026-11) hit one point squarely: GoneBy had
**no measurement at all** of how well the judge's keep/discard verdicts
select — the 172-test suite mocked every service, and the single reported
walk (216 → 43 kept) came from the catalog submission, not the repo.
This directory's harness is the start of fixing that.

## What is measured

`scripts/eval_walk.py` runs the real `LLMJudge` + `BackwardWalker` over
`tests/fixtures/walk_golden.json`: a 44-entry synthetic project transcript
("roaster-export") with hand-assigned relevance labels, two superseded
decisions, three raw duplicate pairs, and normal work chatter. Metrics per
run: precision / recall / F1 against the labels, dedup count, and — over
`--repeat N` — verdict variance of the same prompt.

## What it is not

- Labels were authored by an AI agent for a fictional project. They measure
  judge behavior on a controlled corpus, not quality on your real project.
- The plugin's production path batches judgment (25 entries/call, 1200-char
  caps, host model); the harness uses the standalone unbatched path against
  whatever `GONEBY_LLM_*` points at. Numbers are comparable *between runs*,
  not directly transferable to Hermes-side walks.

## Reproduce

```bash
export GONEBY_LLM_BASE_URL=http://your-server/v1     # OpenAI-compatible chat
export GONEBY_LLM_MODEL=your-judge-model
.venv/bin/python scripts/eval_walk.py --repeat 3
```

No-network plumbing check: `.venv/bin/python scripts/eval_walk.py --self-test`
(keep-everything judge ⇒ recall 1.0, dedup 3).

## Baseline runs

| Date | Commit | Judge model | Recall | Recall spread | Precision | Dedup | Notes |
|------|--------|-------------|--------|---------------|-----------|-------|-------|
| (pending first real run) | 1.1.0+ | — | — | — | — | — | Requires a configured judge endpoint |

Record every run here — a number without date, commit, and model is noise.
The documentation's "~10-20% kept" figure is a **design target** until this
table has real rows.
