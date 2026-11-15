"""WAL line framing — v2 format and sequence bookkeeping.

v2 line: ``v2:<seq>:<base64(nonce||ciphertext||tag)>``

The AEAD additional-authenticated-data binds the WAL file NAME, a token of
its PARENT DIRECTORY, and the line SEQUENCE. Consequences (ctx-audit:
"damaged entries unnoticed" for deletions, reorders, and cross-log splices):

- a line copied into another file (or another directory's same-named file)
  fails its tag — splicing is caught;
- gaps and backwards sequence jumps are visible without decrypting twice;
- the parent-dir token is a hash, not a path, so no location leaks into
  ciphertext metadata.

Legacy v1 lines are bare base64. They stay readable (accept-and-flag,
D4) as ``integrity: unbound`` — position and file were never bound.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

V2_PREFIX = "v2:"
_V2_RE = re.compile(r"^v2:(\d+):([A-Za-z0-9+/=]+)$")


def dir_token(path: Path) -> str:
    """12 hex chars of SHA-256 over the resolved parent directory."""
    return hashlib.sha256(str(path.resolve().parent).encode()).hexdigest()[:12]


def aad_for(path: Path, seq: int) -> bytes:
    """AAD binding: file name | sequence | parent-dir token."""
    return f"{path.name}|{seq}|{dir_token(path)}".encode()


def encode_line(seq: int, blob: bytes) -> str:
    return f"{V2_PREFIX}{seq}:{base64.b64encode(blob).decode('ascii')}"


def decode_line(line: str) -> tuple[int | None, bytes]:
    """Return (seq, raw nonce||ct||tag bytes); seq is None for a legacy line.

    Raises ValueError on anything that is not valid v2 framing or base64 —
    the reader turns that into a CORRUPTED marker.
    """
    m = _V2_RE.match(line)
    if m:
        return int(m.group(1)), base64.b64decode(m.group(2))
    return None, base64.b64decode(line)


def next_seq(path: Path) -> int:
    """Highest usable sequence in an existing WAL + 1.

    Mixed files: legacy lines had no sequence, so they occupy the first N
    numbers by file order; v2 numbers win when they are higher."""
    if not path.exists():
        return 1
    legacy_count = 0
    highest = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(V2_PREFIX):
            m = _V2_RE.match(line)
            if m:
                highest = max(highest, int(m.group(1)))
            else:
                legacy_count += 1
        else:
            legacy_count += 1
    return max(highest, legacy_count) + 1


def key_id_path(wal_path: Path) -> Path:
    """Sidecar carrying the WAL's key fingerprint (future rotation hook)."""
    return wal_path.with_suffix(".keyid")
