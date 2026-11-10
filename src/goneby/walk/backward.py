"""Backward walk — retroactively filter WAL entries using the shipped MVP as oracle.

The algorithm:
1. Start with the shipped MVP (the "answer key")
2. Walk backward through WAL entries
3. For each entry, use an LLM to determine if it contributed to the final result
4. Keep relevant entries, discard irrelevant ones
5. Deduplicate what remains
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WalkResult:
    """Result of a backward walk."""

    kept_entries: list[dict[str, Any]] = field(default_factory=list)
    discarded_count: int = 0
    total_count: int = 0
    dedup_count: int = 0

    @property
    def kept_count(self) -> int:
        return len(self.kept_entries)


class BackwardWalker:
    """Walk backward through WAL entries and filter using the MVP as oracle.

    Uses an LLM judge to determine relevance. The judge is injected
    as a callable to keep the walker testable and model-agnostic.
    """

    def __init__(
        self,
        judge_fn: Callable[[str, list[dict[str, Any]]], list[bool]],
    ):
        """Initialize the backward walker.

        Args:
            judge_fn: Callable that takes (mvp_text, entries) and returns
                      a list of booleans indicating which entries are relevant.
        """
        self._judge_fn = judge_fn

    async def walk(
        self,
        mvp_text: str,
        entries: list[dict[str, Any]],
    ) -> WalkResult:
        """Run the backward walk.

        Args:
            mvp_text: The shipped MVP or final artifact (the "answer key").
            entries: WAL entries in chronological order (oldest first).

        Returns:
            WalkResult with kept entries and statistics.
        """
        if not entries:
            return WalkResult()

        # Walk backward: reverse the entries
        reversed_entries = list(reversed(entries))

        # Ask the judge which entries are relevant
        relevance_flags = await self._judge_fn(mvp_text, reversed_entries)

        # Keep entries marked as relevant (in original order)
        kept = []
        discarded = 0
        seen_contents: set[str] = set()
        dedup_count = 0

        for entry, is_relevant in zip(reversed_entries, relevance_flags):
            if not is_relevant:
                discarded += 1
                continue

            # Simple deduplication by content hash
            content = entry.get("content", "")
            content_hash = self._content_hash(content)

            if content_hash in seen_contents:
                dedup_count += 1
                continue

            seen_contents.add(content_hash)
            kept.append(entry)

        # Reverse back to chronological order
        kept.reverse()

        return WalkResult(
            kept_entries=kept,
            discarded_count=discarded,
            total_count=len(entries),
            dedup_count=dedup_count,
        )

    @staticmethod
    def _content_hash(content: str) -> str:
        """Generate a simple hash for deduplication."""
        # Normalize whitespace and lowercase for dedup
        normalized = re.sub(r"\s+", " ", content.strip().lower())
        return str(hash(normalized))


class LLMJudge:
    """LLM-based judge for backward walk relevance determination.

    Uses a simple prompt to ask the LLM which entries contributed
    to the final result. Designed to work with any LLM that supports
    structured output.
    """

    def __init__(self, llm_fn: Callable[[str], str]):
        """Initialize with an LLM function.

        Args:
            llm_fn: Callable that takes a prompt string and returns
                    the LLM response as a string.
        """
        self._llm_fn = llm_fn

    async def judge(
        self,
        mvp_text: str,
        entries: list[dict[str, Any]],
    ) -> list[bool]:
        """Ask the LLM which entries are relevant to the MVP.

        Args:
            mvp_text: The shipped MVP text.
            entries: WAL entries to evaluate.

        Returns:
            List of booleans indicating relevance.
        """
        # Build the prompt
        entries_text = "\n\n".join(
            f"--- Entry {i} ---\nTimestamp: {e.get('timestamp', 'unknown')}\n"
            f"Content: {e.get('content', '')}\n"
            f"Metadata: {json.dumps(e.get('metadata', {}))}"
            for i, e in enumerate(entries)
        )

        prompt = f"""You are a relevance judge for a project memory system.

Given the following shipped MVP (the final product), determine which of the
following project log entries contributed to the final result.

An entry is relevant if:
- It represents a decision that shaped the final product
- It contains information that directly influenced the architecture or design
- It documents a finding that led to a change in direction
- It captures a constraint that affected the implementation

An entry is NOT relevant if:
- It was a random thought that didn't lead anywhere
- It was a dead-end exploration
- It was superseded by a later decision
- It doesn't relate to the final product

## Shipped MVP

{mvp_text}

## Project Log Entries (in reverse chronological order)

{entries_text}

## Instructions

Respond with ONLY a JSON array of booleans, one per entry, in the same order.
Use true for relevant entries and false for irrelevant entries.

Example: [true, false, true, true, false]

JSON array:"""

        response = await self._llm_fn(prompt)

        # Parse the response
        try:
            # Extract JSON array from response
            match = re.search(r"\[[\s,true|false]*\]", response)
            if match:
                flags = json.loads(match.group())
                # Ensure we have the right number of flags
                if len(flags) < len(entries):
                    flags.extend([False] * (len(entries) - len(flags)))
                return flags[: len(entries)]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: if parsing fails, keep everything
        return [True] * len(entries)
