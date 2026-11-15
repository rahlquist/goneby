#!/usr/bin/env python3
"""Backward-walk selection-quality harness (ctx-audit: 'no measurement existed').

Runs the real LLMJudge over a hand-labeled fixture (tests/fixtures/
walk_golden.json) and reports how the model's keep/discard verdicts compare
to the labels: precision, recall, F1, dedup count, and run-to-run variance.
Judge-only — no embedding server or vector store needed.

Usage:
    export GONEBY_LLM_BASE_URL=http://your-server/v1   # OpenAI-compatible
    export GONEBY_LLM_MODEL=your-judge-model
    export GONEBY_LLM_API_KEY=optional
    python scripts/eval_walk.py [--repeat 3] [--json out.json] [--fixture PATH]

    python scripts/eval_walk.py --self-test   # plumbing check, no network

Caveats this harness cannot hide: labels are synthetic (authored by an
agent, see the fixture's _note); results measure judge behavior on a
controlled corpus, not real-project quality. Record runs in
docs/BASELINE.md with date, model, and commit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goneby.walk.backward import BackwardWalker, LLMJudge

DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "walk_golden.json"


def load_fixture(path: Path) -> tuple[str, list[dict], dict[str, bool], dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    product = data["product"]
    entries = [
        {
            "content": e["content"],
            "timestamp": f"2026-11-{14:02d}T00:{i:02d}:00+00:00",
            "metadata": {"role": e.get("role", "user"), "fid": e["fid"]},
        }
        for i, e in enumerate(data["entries"])
    ]
    label_by_content: dict[str, bool] = {}
    fid_by_content: dict[str, int] = {}
    for e in data["entries"]:  # first occurrence wins; duplicates share content+label
        label_by_content.setdefault(e["content"], bool(e["relevant"]))
        fid_by_content.setdefault(e["content"], e["fid"])
    return product, entries, label_by_content, fid_by_content


def score(
    kept_entries: list[dict],
    label_by_content: dict[str, bool],
    fid_by_content: dict[str, int],
) -> dict:
    kept_contents = {e.get("content", "") for e in kept_entries}
    relevant = {c for c, rel in label_by_content.items() if rel}
    irrelevant = set(label_by_content) - relevant
    tp = len(kept_contents & relevant)
    fp = len(kept_contents & irrelevant)
    fn = len(relevant - kept_contents)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    missed = sorted(fid_by_content[c] for c in relevant - kept_contents)
    noise = sorted(fid_by_content[c] for c in kept_contents & irrelevant)
    return {
        "kept": len(kept_contents),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "missed_relevant_fids": missed,
        "kept_irrelevant_fids": noise,
    }


async def run_once(product: str, entries: list[dict], judge_fn) -> tuple[object, list[dict]]:
    result = await BackwardWalker(judge_fn).walk(product, entries)
    return result, result.kept_entries


async def evaluate(args, judge_fn, label_by_content: dict[str, bool], fid_by_content: dict[str, int]) -> None:
    product, entries, _, _ = load_fixture(Path(args.fixture))
    rows = []
    for i in range(args.repeat):
        result, kept = await run_once(product, entries, judge_fn)
        row = score(kept, label_by_content, fid_by_content)
        row["run"] = i + 1
        row["dedup"] = result.dedup_count
        row["discarded"] = result.discarded_count
        rows.append(row)
        print(
            f"run {row['run']}: kept={row['kept']} dedup={row['dedup']} "
            f"discarded={row['discarded']} precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} f1={row['f1']:.3f}"
        )
        if row["missed_relevant_fids"]:
            print(f"       missed relevant fids: {row['missed_relevant_fids']}")
        if row["kept_irrelevant_fids"]:
            print(f"       kept-irrelevant fids: {row['kept_irrelevant_fids']}")
    recalls = [r["recall"] for r in rows]
    precisions = [r["precision"] for r in rows]
    summary = {
        "fixture": args.fixture,
        "runs": rows,
        "recall_mean": round(statistics.mean(recalls), 4),
        "recall_spread": round(max(recalls) - min(recalls), 4),
        "precision_mean": round(statistics.mean(precisions), 4),
    }
    print(
        f"\nmeans over {args.repeat} runs: recall={summary['recall_mean']:.3f} "
        f"(spread {summary['recall_spread']:.3f}) precision={summary['precision_mean']:.3f}"
    )
    print("Record this run in docs/BASELINE.md: date, model, commit, numbers.")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


async def self_test() -> None:
    """No network: keep-everything judge must yield recall=1.0, precision<1."""
    product, entries, label_by_content, fid_by_content = load_fixture(DEFAULT_FIXTURE)

    async def keep_all(mvp, batch):
        return [True] * len(batch)

    _, kept = await run_once(product, entries, keep_all)
    row = score(kept, label_by_content, fid_by_content)
    assert row["recall"] == 1.0, row
    assert 0 < row["precision"] < 1.0, row  # fixture contains irrelevant entries
    expected_dedup = 4  # (3,32) + (5,31) + two drops in the (11,33,34) triple
    async def keep_all2(mvp, batch):
        return [True] * len(batch)
    result, _ = await run_once(product, entries, keep_all2)
    assert result.dedup_count == expected_dedup, result.dedup_count
    print(f"self-test OK: recall={row['recall']}, precision={row['precision']}, "
          f"dedup={result.dedup_count} as designed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--json", default=None)
    parser.add_argument("--self-test", action="store_true", help="no-network plumbing check")
    args = parser.parse_args()

    if args.self_test:
        asyncio.run(self_test())
        return

    from goneby.config.settings import GoneByConfig
    from goneby.mcp_server import create_llm_fn

    config = GoneByConfig.from_env()
    if not config.llm_base_url or not config.llm_model:
        sys.exit("eval needs a configured judge: set GONEBY_LLM_BASE_URL + GONEBY_LLM_MODEL\n"
                 "(or run --self-test for the no-network check)")
    label_by_content, fid_by_content = load_fixture(Path(args.fixture))[2:]
    asyncio.run(evaluate(args, LLMJudge(create_llm_fn(config)).judge, label_by_content, fid_by_content))


if __name__ == "__main__":
    main()
