"""WAL reader — read and decrypt entries from disk."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path

from goneby.crypto.cipher import Cipher


class WALReader:
    """Read and decrypt WAL entries from disk."""

    def __init__(self, wal_path: Path, cipher: Cipher):
        self._wal_path = wal_path
        self._cipher = cipher

    def read_all(self) -> list[dict]:
        """Read and decrypt all entries from the WAL."""
        return list(self.iter_entries())

    def iter_entries(self) -> Iterator[dict]:
        """Iterate over decrypted WAL entries.

        A damaged or undecryptable line yields a CORRUPTED marker entry rather
        than aborting — one bad append must not hide the rest of the log.
        """
        if not self._wal_path.exists():
            return

        with open(self._wal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    encrypted = base64.b64decode(line)
                    plaintext = self._cipher.decrypt(encrypted)
                    entry = json.loads(plaintext.decode("utf-8"))
                    yield entry
                except Exception as e:  # noqa: BLE001 - must not abort the whole log
                    # Mark corrupted but continue
                    yield {
                        "timestamp": None,
                        "content": f"[CORRUPTED: {type(e).__name__}]",
                        "metadata": {"corrupted": True},
                    }

    def read_raw_lines(self) -> Iterator[str]:
        """Iterate over raw base64 lines (for backup/transfer)."""
        if not self._wal_path.exists():
            return

        with open(self._wal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
