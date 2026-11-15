"""WAL writer — append encrypted, position-bound entries to disk."""

from __future__ import annotations

import base64
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from goneby.crypto.cipher import Cipher
from goneby.wal import framing


class WALWriter:
    """Append-only encrypted WAL writer (v2 framing).

    Each entry is a JSON object with timestamp + content, encrypted with
    ChaCha20-Poly1305 whose AAD binds file name, parent-dir token, and
    sequence (see ``goneby.wal.framing``), then written as ``v2:<seq>:<b64>``.
    A sidecar ``<stem>.keyid`` records the key fingerprint.
    """

    def __init__(self, wal_path: Path, cipher: Cipher):
        self._wal_path = Path(wal_path)
        self._cipher = cipher
        self._seq: int | None = None
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Create WAL directory if it doesn't exist."""
        self._wal_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _next_seq(self) -> int:
        if self._seq is None:
            self._seq = framing.next_seq(self._wal_path) - 1
        return self._seq + 1

    def _write_key_id(self) -> None:
        header = framing.key_id_path(self._wal_path)
        try:
            if not header.exists() or header.read_text().strip() != self._cipher.key_id:
                header.write_text(self._cipher.key_id + "\n", encoding="utf-8")
                os.chmod(header, 0o600)
        except OSError:
            # The key id is an operational nicety; never block a capture.
            pass

    def append(self, content: str, metadata: dict | None = None) -> None:
        """Append an encrypted, position-bound entry to the WAL.

        Args:
            content: The text content to store.
            metadata: Optional dict of metadata (tags, type, etc.).
        """
        seq = self._next_seq()
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "content": content,
            "metadata": metadata or {},
        }
        plaintext = json.dumps(entry, ensure_ascii=False).encode()
        aad = framing.aad_for(self._wal_path, seq)
        blob = self._cipher.encrypt(plaintext, aad)
        line = framing.encode_line(seq, blob)

        new_file = not self._wal_path.exists()
        with open(self._wal_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if new_file:
            # Transcript material, even encrypted: keep it owner-only.
            os.chmod(self._wal_path, 0o600)
        self._seq = seq
        self._write_key_id()

    def append_raw(self, encrypted_line: str, seq: int | None = None) -> None:
        """Append a pre-encrypted line (for import/replay).

        A v2 line must already carry the binding for THIS file's name, dir
        token, and sequence — a line encrypted for another WAL simply fails
        its tag on read. ``seq`` given with a bare-base64 blob wraps it as
        ``v2:<seq>:<blob>`` (encrypted for this file); a legacy line without
        ``seq`` imports as unbound."""
        line = encrypted_line.strip()
        if seq is not None and not line.startswith(framing.V2_PREFIX):
            line = framing.encode_line(seq, base64.b64decode(line))
        with open(self._wal_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
