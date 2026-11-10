"""WAL writer — append encrypted entries to disk."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from goneby.crypto.cipher import Cipher


class WALWriter:
    """Append-only encrypted WAL writer.

    Each entry is a JSON object with timestamp + content, encrypted
    independently with ChaCha20-Poly1305, and written as a single
    base64-encoded line.
    """

    def __init__(self, wal_path: Path, cipher: Cipher):
        self._wal_path = wal_path
        self._cipher = cipher
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Create WAL directory if it doesn't exist."""
        self._wal_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, content: str, metadata: dict | None = None) -> None:
        """Append an encrypted entry to the WAL.

        Args:
            content: The text content to store.
            metadata: Optional dict of metadata (tags, type, etc.).
        """
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "content": content,
            "metadata": metadata or {},
        }
        plaintext = json.dumps(entry, ensure_ascii=False).encode("utf-8")
        encrypted = self._cipher.encrypt(plaintext)

        # Write as base64-encoded line for easy parsing
        import base64
        encoded = base64.b64encode(encrypted).decode("ascii")

        with open(self._wal_path, "a", encoding="utf-8") as f:
            f.write(encoded + "\n")

    def append_raw(self, encrypted_line: str) -> None:
        """Append a pre-encrypted base64 line (for import/replay)."""
        with open(self._wal_path, "a", encoding="utf-8") as f:
            f.write(encrypted_line + "\n")
