"""Metadata filtering shared by vector backends."""

from __future__ import annotations

from typing import Any


def matches_filter(metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
    """True when `metadata` satisfies every equality constraint in `filter`.

    A missing key never matches — an empty filter matches everything.
    """
    for key, want in (filter or {}).items():
        if key not in metadata:
            return False
        got = metadata[key]
        if isinstance(want, (list, tuple, set)):
            # List constraint means "value is one of these".
            if got not in want:
                return False
        elif got != want:
            return False
    return True


__all__ = ["matches_filter"]
