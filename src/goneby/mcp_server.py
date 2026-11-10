"""GoneBy MCP server — exposes memory tools via Model Context Protocol."""

from __future__ import annotations

from typing import Any

from goneby.config.settings import GoneByConfig, project_wal_path
from goneby.crypto.cipher import Cipher
from goneby.memory.embedder import Embedder, EmbeddingConfig, EmbeddingError
from goneby.memory.store import MemoryStore
from goneby.vector.factory import create_backend
from goneby.wal.manager import WALManager
from goneby.walk.backward import BackwardWalker, LLMJudge


def load_mcp_server_class():
    """Return the FastMCP server class for whichever mcp SDK is installed.

    mcp 1.x exposes `mcp.server.fastmcp.FastMCP`; 2.x renamed it to
    `mcp.server.mcpserver.MCPServer`. Support both so a fresh install of
    either major version works.
    """
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP

        return FastMCP
    except ImportError as exc:
        raise ImportError(
            "MCP support requires the 'mcp' package. "
            "Install with: pip install 'goneby[mcp]'"
        ) from exc


def create_mcp_server(config: GoneByConfig | None = None):
    """Create and configure the GoneBy MCP server.

    Args:
        config: Optional configuration. Loads from env if not provided.

    Returns:
        Configured MCP server instance.
    """
    ServerClass = load_mcp_server_class()

    if config is None:
        config = GoneByConfig.from_env()

    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")

    # Initialize components
    cipher = Cipher(config.encryption_key)
    wal = WALManager(project_wal_path(config.wal_path, config.project), cipher)
    backend = create_backend(config)
    embedder = Embedder(
        EmbeddingConfig(
            base_url=config.embedding_base_url,
            model=config.embedding_model,
            api_key=config.embedding_api_key,
            dimensions=config.embedding_dimensions,
            batch_size=config.embedding_batch_size,
        )
    )
    store = MemoryStore(embedder, backend, project=config.project)

    # LLM function for backward walk (uses Anthropic by default)
    async def llm_fn(prompt: str) -> str:
        """Call LLM for backward walk judgment."""
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except ImportError:
            raise RuntimeError(
                "LLM support requires the 'anthropic' package. "
                "Install with: pip install anthropic"
            )

    judge = LLMJudge(llm_fn)
    walker = BackwardWalker(judge.judge)

    # Create MCP server
    mcp = ServerClass(
        name="goneby",
        instructions="Retroactive project memory for LLM agents. "
        "Capture project output between context compactions, "
        "then retroactively filter and store relevant memories.",
    )

    @mcp.tool()
    async def wall_append(content: str, entry_type: str = "note") -> dict[str, Any]:
        """Append an entry to the Write-Ahead Log.

        Call this at each context compaction boundary to capture
        all project output. No filtering happens here — everything
        is stored. The backward walk at project close will determine
        what was actually relevant.

        Args:
            content: The text content to store.
            entry_type: Type of entry (decision, finding, note, constraint).

        Returns:
            Entry confirmation with timestamp and ID.
        """
        metadata = {"type": entry_type}
        wal.append(content, metadata)

        return {
            "status": "ok",
            "timestamp": wal.read_all()[-1]["timestamp"] if wal.entry_count > 0 else None,
            "entry_type": entry_type,
            "wal_entry_count": wal.entry_count,
        }

    @mcp.tool()
    async def wall_read() -> dict[str, Any]:
        """Read all entries from the WAL.

        Returns all decrypted entries in the WAL. Use this to
        review what has been captured before running the backward walk.
        """
        entries = wal.read_all()
        return {
            "count": len(entries),
            "entries": entries,
        }

    @mcp.tool()
    async def walk_backward(mvp_text: str) -> dict[str, Any]:
        """Run the backward walk filter.

        This is the core intelligence of GoneBy. Walk backward through
        the WAL entries using the shipped MVP as the oracle to determine
        which entries were relevant to the final result.

        Args:
            mvp_text: The shipped MVP or final artifact (the "answer key").

        Returns:
            Walk results with kept entries and statistics.
        """
        entries = wal.read_all()

        if not entries:
            return {
                "status": "empty",
                "message": "No WAL entries to process.",
                "kept_count": 0,
                "discarded_count": 0,
            }

        result = await walker.walk(mvp_text, entries)

        return {
            "status": "ok",
            "kept_count": result.kept_count,
            "discarded_count": result.discarded_count,
            "total_count": result.total_count,
            "dedup_count": result.dedup_count,
            "kept_entries": result.kept_entries,
        }

    @mcp.tool()
    async def store_memory(entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Store filtered entries in the vector database.

        Run walk_backward first, then pass its kept_entries here. Each entry is
        embedded with the configured model and upserted into Valkey or Qdrant.
        IDs are content-derived, so re-storing the same entry is idempotent.

        Args:
            entries: The kept entries from walk_backward.

        Returns:
            Storage summary with counts, vector dimensions, and backend.
        """
        try:
            return await store.index_entries(entries)
        except EmbeddingError as exc:
            return {"status": "error", "stage": "embedding", "error": str(exc), "stored": 0}
        except Exception as exc:  # noqa: BLE001 - tool must not crash the server
            return {
                "status": "error",
                "stage": "vector_store",
                "backend": config.vector_backend,
                "error": f"{type(exc).__name__}: {exc}",
                "stored": 0,
            }

    @mcp.tool()
    async def query_memory(
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        scope: str = "project",
    ) -> dict[str, Any]:
        """Query the memory store with semantic search.

        The query is embedded with the same model used at store time, then
        matched by cosine similarity against stored memories.

        By default only this project's memories are searched. Pass
        ``scope="all"`` to search every project — useful for "how did I solve
        this in another repo?" lookups, at the cost of possible cross-project
        noise.

        Args:
            query: Natural-language question about the project.
            top_k: Maximum results to return.
            min_score: Minimum similarity score (0.0-1.0) to include a hit.
            scope: ``"project"`` (default) or ``"all"``.

        Returns:
            Ranked memories with scores.
        """
        try:
            return await store.search(
                query, top_k=top_k, min_score=min_score, scope=scope
            )
        except EmbeddingError as exc:
            return {
                "status": "error",
                "stage": "embedding",
                "error": str(exc),
                "results": [],
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "stage": "vector_store",
                "backend": config.vector_backend,
                "error": f"{type(exc).__name__}: {exc}",
                "results": [],
            }

    @mcp.tool()
    async def generate_key() -> dict[str, Any]:
        """Generate a new encryption key.

        Returns a hex-encoded 256-bit key for WAL encryption.
        Store this securely (e.g., in Hermes vault or Bitwarden).
        """
        key = Cipher.generate_key()
        return {
            "key_hex": key.hex(),
            "key_size_bytes": len(key),
            "algorithm": "ChaCha20-Poly1305",
            "warning": "Store this key securely. If lost, WAL entries cannot be decrypted.",
        }

    @mcp.tool()
    async def status() -> dict[str, Any]:
        """Get GoneBy server status.

        Returns configuration, WAL statistics, and a live probe of the
        embedding and vector backends.
        """
        return {
            "version": "0.1.0",
            "project": config.project,
            "wal_path": str(project_wal_path(config.wal_path, config.project)),
            "wal_entry_count": wal.entry_count,
            "vector_backend": config.vector_backend,
            "embedding": {
                "base_url": config.embedding_base_url,
                "model": config.embedding_model,
                "dimensions": config.embedding_dimensions,
            },
            "encryption": "ChaCha20-Poly1305",
            "health": await store.health(),
        }

    return mcp


def main():
    """Run the GoneBy MCP server."""

    config = GoneByConfig.from_env()
    errors = config.validate()

    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nSet the required environment variables and try again.")
        raise SystemExit(1)

    server = create_mcp_server(config)
    server.run()


if __name__ == "__main__":
    main()
