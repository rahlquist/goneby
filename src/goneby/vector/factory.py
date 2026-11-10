"""Vector backend factory."""

from __future__ import annotations

from goneby.config.settings import GoneByConfig
from goneby.vector.base import VectorBackend
from goneby.vector.memory_backend import MemoryBackend
from goneby.vector.qdrant_backend import QdrantBackend
from goneby.vector.valkey_backend import ValkeyBackend


def create_backend(config: GoneByConfig) -> VectorBackend:
    """Create a vector backend from configuration.

    Args:
        config: GoneBy configuration.

    Returns:
        VectorBackend instance.
    """
    if config.vector_backend == "valkey":
        return ValkeyBackend(
            host=config.valkey_host,
            port=config.valkey_port,
            db=config.valkey_db,
        )
    elif config.vector_backend == "qdrant":
        return QdrantBackend(
            host=config.qdrant_host,
            port=config.qdrant_port,
            collection=config.qdrant_collection,
            dimensions=config.embedding_dimensions,
        )
    elif config.vector_backend == "memory":
        return MemoryBackend()
    else:
        raise ValueError(f"Unknown vector backend: {config.vector_backend}")
