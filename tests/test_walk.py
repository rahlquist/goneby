"""Tests for GoneBy backward walk filter."""

from __future__ import annotations

from goneby.walk.backward import BackwardWalker, LLMJudge


class TestBackwardWalker:
    """Test backward walk logic."""

    def test_empty_entries(self):
        async def judge_fn(mvp, entries):
            return []

        walker = BackwardWalker(judge_fn)

        async def run():
            result = await walker.walk("MVP", [])
            assert result.kept_count == 0

        import asyncio
        asyncio.run(run())

    def test_all_relevant(self):
        entries = [
            {"content": "Decision A", "timestamp": "2026-01-01"},
            {"content": "Decision B", "timestamp": "2026-01-02"},
        ]

        async def judge_fn(mvp, ents):
            return [True] * len(ents)

        walker = BackwardWalker(judge_fn)

        async def run():
            result = await walker.walk("MVP", entries)
            assert result.kept_count == 2
            assert result.discarded_count == 0

        import asyncio
        asyncio.run(run())

    def test_some_irrelevant(self):
        entries = [
            {"content": "Decision A", "timestamp": "2026-01-01"},
            {"content": "Random thought", "timestamp": "2026-01-02"},
            {"content": "Decision B", "timestamp": "2026-01-03"},
        ]

        async def judge_fn(mvp, ents):
            # Entries are reversed in the walk
            return [True, False, True]  # Decision B, Random thought, Decision A

        walker = BackwardWalker(judge_fn)

        async def run():
            result = await walker.walk("MVP", entries)
            assert result.kept_count == 2
            assert result.discarded_count == 1

        import asyncio
        asyncio.run(run())

    def test_deduplication(self):
        entries = [
            {"content": "Same decision", "timestamp": "2026-01-01"},
            {"content": "Same decision", "timestamp": "2026-01-02"},
            {"content": "Different decision", "timestamp": "2026-01-03"},
        ]

        async def judge_fn(mvp, ents):
            return [True, True, True]

        walker = BackwardWalker(judge_fn)

        async def run():
            result = await walker.walk("MVP", entries)
            assert result.kept_count == 2
            assert result.dedup_count == 1

        import asyncio
        asyncio.run(run())


class TestLLMJudge:
    """Test LLM judge parsing."""

    def test_parse_json_response(self):
        async def llm_fn(prompt):
            return "```json\n[true, false, true]\n```"

        judge = LLMJudge(llm_fn)

        async def run():
            entries = [{"content": "A"}, {"content": "B"}, {"content": "C"}]
            flags = await judge.judge("MVP", entries)
            assert flags == [True, False, True]

        import asyncio
        asyncio.run(run())

    def test_parse_plain_response(self):
        async def llm_fn(prompt):
            return "[true, false]"

        judge = LLMJudge(llm_fn)

        async def run():
            entries = [{"content": "A"}, {"content": "B"}]
            flags = await judge.judge("MVP", entries)
            assert flags == [True, False]

        import asyncio
        asyncio.run(run())

    def test_fallback_on_invalid_response(self):
        async def llm_fn(prompt):
            return "I don't know"

        judge = LLMJudge(llm_fn)

        async def run():
            entries = [{"content": "A"}, {"content": "B"}]
            flags = await judge.judge("MVP", entries)
            assert flags == [True, True]  # Fallback: keep all

        import asyncio
        asyncio.run(run())
