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
