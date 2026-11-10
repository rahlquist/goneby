"""Settings loaded from environment variables or config file."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


def project_wal_path(base: Path, project: str) -> Path:
    """Per-project WAL path.

    Each project gets its own log so a backward walk for one project never
    sees another project's evidence. The directory is shared; only the file
    name is namespaced.
    """
    safe = project_slug(project)
    return base / f"{safe}.wal"


def project_slug(project: str) -> str:
    """Filesystem-safe project identifier.

    Absolute paths and dotted names are flattened rather than rejected, so a
    cwd like ``/home/rahlquist/goneby`` becomes a stable slug instead of an
    error.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", project.strip()).strip("._")
    return cleaned or "default"


@dataclass
class GoneByConfig:
    """Runtime configuration for GoneBy."""

    # Encryption
    encryption_key: bytes = field(default=b"")

    # WAL storage
    wal_path: Path = field(default=Path("~/.goneby/wal").expanduser())

    # Vector backend: "valkey" or "qdrant"
    vector_backend: str = "valkey"

    # Project namespace — keeps memories from different projects separated
    project: str = "default"

    # Valkey settings
    valkey_host: str = "localhost"
    valkey_port: int = 6379
    valkey_db: int = 0

    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "goneby"

    # Embedding backend (any OpenAI-compatible /v1/embeddings server)
    embedding_base_url: str = "http://localhost:8081/v1"
    embedding_model: str = "qwen3-embedding-0.6b"
    embedding_api_key: str | None = None
    embedding_dimensions: int = 512  # MRL truncation target
    embedding_batch_size: int = 16

    @classmethod
    def from_env(cls) -> GoneByConfig:
        """Load configuration from environment variables."""
        key_hex = os.environ.get("GONEBY_ENCRYPTION_KEY", "")
        key = bytes.fromhex(key_hex) if key_hex else b""

        return cls(
            encryption_key=key,
            wal_path=Path(os.environ.get("GONEBY_WAL_PATH", "~/.goneby/wal")).expanduser(),
            vector_backend=os.environ.get("GONEBY_VECTOR_BACKEND", "valkey"),
            project=os.environ.get("GONEBY_PROJECT", "default"),
            valkey_host=os.environ.get("GONEBY_VALKEY_HOST", "localhost"),
            valkey_port=int(os.environ.get("GONEBY_VALKEY_PORT", "6379")),
            valkey_db=int(os.environ.get("GONEBY_VALKEY_DB", "0")),
            qdrant_host=os.environ.get("GONEBY_QDRANT_HOST", "localhost"),
            qdrant_port=int(os.environ.get("GONEBY_QDRANT_PORT", "6333")),
            qdrant_collection=os.environ.get("GONEBY_QDRANT_COLLECTION", "goneby"),
            embedding_base_url=os.environ.get(
                "GONEBY_EMBEDDING_BASE_URL", "http://localhost:8081/v1"
            ),
            embedding_model=os.environ.get(
                "GONEBY_EMBEDDING_MODEL", "qwen3-embedding-0.6b"
            ),
            embedding_api_key=os.environ.get("GONEBY_EMBEDDING_API_KEY") or None,
            embedding_dimensions=int(os.environ.get("GONEBY_EMBEDDING_DIMENSIONS", "512")),
            embedding_batch_size=int(os.environ.get("GONEBY_EMBEDDING_BATCH_SIZE", "16")),
        )

    def validate(self) -> list[str]:
        """Return list of validation errors (empty if valid)."""
        errors = []
        if not self.encryption_key:
            errors.append("encryption_key is required (set GONEBY_ENCRYPTION_KEY)")
        elif len(self.encryption_key) != 32:
            errors.append(f"encryption_key must be 32 bytes, got {len(self.encryption_key)}")
        if self.vector_backend not in ("valkey", "qdrant", "memory"):
            errors.append(
                "vector_backend must be 'valkey', 'qdrant', or 'memory', "
                f"got '{self.vector_backend}'"
            )
        return errors
