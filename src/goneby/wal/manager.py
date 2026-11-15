"""WAL manager — high-level interface for WAL operations."""

from __future__ import annotations

from pathlib import Path

from goneby.crypto.cipher import Cipher
from goneby.wal.reader import WALReader
from goneby.wal.writer import WALWriter


class WALManager:
    """High-level WAL operations."""

    def __init__(self, wal_path: Path, cipher: Cipher):
        self._writer = WALWriter(wal_path, cipher)
        self._reader = WALReader(wal_path, cipher)

    def append(self, content: str, metadata: dict | None = None) -> None:
        """Append an entry to the WAL."""
        self._writer.append(content, metadata)

    def read_all(self) -> list[dict]:
        """Read all entries."""
        return self._reader.read_all()

    def read_for_walk(self) -> tuple[list[dict], list[str]]:
        """Entries safe to judge, plus integrity notes.

        Markers ([CORRUPTED], [GAP]) and out-of-order lines never reach the
        judge — a splice or tampered line must not get a model vote on it.
        Unbound legacy entries ARE kept (accept-and-flag): excluding them
        would silently lose history during migration.
        """
        clean: list[dict] = []
        notes: list[str] = []
        for entry in self.read_all():
            md = entry.get("metadata") or {}
            if md.get("corrupted"):
                notes.append(entry.get("content") or "corrupted line")
                continue
            if md.get("gap"):
                notes.append(entry.get("content") or "sequence gap")
                continue
            if md.get("out_of_order"):
                notes.append(
                    f"line at position {md.get('wal_seq')} is out of order; excluded"
                )
                continue
            if md.get("integrity") == "unbound":
                notes.append(
                    f"unbound legacy entry kept (no position binding): "
                    f"{entry.get('timestamp')}"
                )
            clean.append(entry)
        return clean, notes

    def iter_entries(self):
        """Iterate over entries."""
        return self._reader.iter_entries()

    def clear(self) -> None:
        """Clear the WAL (for testing/reset)."""
        if self._writer._wal_path.exists():
            self._writer._wal_path.unlink()

    @property
    def path(self) -> Path:
        return self._writer._wal_path

    @property
    def entry_count(self) -> int:
        """Count entries in the WAL."""
        return sum(1 for _ in self._reader.iter_entries())
