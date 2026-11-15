"""WAL reader — read and decrypt position-bound entries from disk.

v2 lines decrypt only when file name, parent dir, and sequence match what
was bound at write time. Deletions (gaps) and reorders (backwards sequence)
are reported; trailing deletion cannot be (no end-of-file marker is bound) —
a known, documented limitation. Legacy bare-base64 lines stay readable and
are flagged ``integrity: unbound`` (accept-and-flag, D4).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from goneby.crypto.cipher import Cipher
from goneby.wal import framing


class WALReader:
    """Read and decrypt WAL entries from disk."""

    def __init__(self, wal_path: Path, cipher: Cipher):
        self._wal_path = Path(wal_path)
        self._cipher = cipher

    def read_all(self) -> list[dict]:
        """Read and decrypt all entries from the WAL."""
        return list(self.iter_entries())

    def _key_hint(self) -> str:
        """'wrong key?' annotation when the sidecar says a different key."""
        header = framing.key_id_path(self._wal_path)
        try:
            saved = header.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        if saved and saved != self._cipher.key_id:
            return f" (wrong key? file says {saved}, current key is {self._cipher.key_id})"
        return ""

    def _corrupted(self, detail: str) -> dict:
        return {
            "timestamp": None,
            "content": f"[CORRUPTED: {detail}{self._key_hint()}]",
            "metadata": {"corrupted": True},
        }

    def iter_entries(self) -> Iterator[dict]:
        """Iterate over decrypted WAL entries.

        A damaged or undecryptable line yields a CORRUPTED marker entry
        rather than aborting — one bad append must not hide the rest of the
        log. Gaps yield [GAP] markers; out-of-order v2 lines are kept but
        flagged ``out_of_order`` so the walk can exclude them."""
        if not self._wal_path.exists():
            return

        prev_seq = 0
        with open(self._wal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    seq, blob = framing.decode_line(line)
                except ValueError as exc:
                    yield self._corrupted(f"unparseable line: {type(exc).__name__}")
                    continue
                aad = framing.aad_for(self._wal_path, seq) if seq is not None else None
                try:
                    plaintext = self._cipher.decrypt(blob, aad)
                    entry = json.loads(plaintext.decode("utf-8"))
                except Exception as exc:  # noqa: BLE001 - must not abort the whole log
                    yield self._corrupted(type(exc).__name__)
                    continue

                metadata = entry.setdefault("metadata", {})
                if seq is None:
                    # Legacy line: content authentic, position never bound.
                    metadata["integrity"] = "unbound"
                    yield entry
                    continue

                metadata["wal_seq"] = seq
                if seq > prev_seq + 1:
                    yield {
                        "timestamp": None,
                        "content": (
                            f"[GAP] {seq - prev_seq - 1} entr"
                            f"{'y' if seq - prev_seq == 2 else 'ies'} "
                            f"between sequence {prev_seq} and {seq} are missing"
                        ),
                        "metadata": {"gap": True, "missing_after": prev_seq, "before": seq},
                    }
                if seq <= prev_seq:
                    metadata["out_of_order"] = True
                prev_seq = max(prev_seq, seq)
                yield entry

    def read_raw_lines(self) -> Iterator[str]:
        """Iterate over raw lines (for backup/transfer)."""
        if not self._wal_path.exists():
            return

        with open(self._wal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
