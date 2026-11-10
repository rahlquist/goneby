"""CLI entry point for GoneBy."""

from __future__ import annotations

import argparse
import sys

from goneby.config.settings import GoneByConfig
from goneby.crypto.cipher import Cipher
from goneby.wal.manager import WALManager


def cmd_generate_key(args: argparse.Namespace) -> int:
    """Generate a new encryption key."""
    key = Cipher.generate_key()
    print(f"Encryption key (hex): {key.hex()}")
    print(f"Key size: {len(key)} bytes")
    print("Algorithm: ChaCha20-Poly1305")
    print()
    print("Store this key securely. If lost, WAL entries cannot be decrypted.")
    print()
    print("Set it as an environment variable:")
    print(f"  export GONEBY_ENCRYPTION_KEY={key.hex()}")
    return 0


def cmd_append(args: argparse.Namespace) -> int:
    """Append an entry to the WAL."""
    config = GoneByConfig.from_env()
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    cipher = Cipher(config.encryption_key)
    wal = WALManager(config.wal_path, cipher)

    content = args.content
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()

    entry_type = args.type or "note"
    wal.append(content, {"type": entry_type})

    print(f"Entry appended to {config.wal_path}")
    print(f"Total entries: {wal.entry_count}")
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    """Read all WAL entries."""
    config = GoneByConfig.from_env()
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    cipher = Cipher(config.encryption_key)
    wal = WALManager(config.wal_path, cipher)

    entries = wal.read_all()
    print(f"WAL: {config.wal_path}")
    print(f"Entries: {len(entries)}")
    print()

    for i, entry in enumerate(entries):
        ts = entry.get("timestamp", "unknown")
        content = entry.get("content", "")
        metadata = entry.get("metadata", {})
        entry_type = metadata.get("type", "note")

        print(f"[{i}] {ts} ({entry_type})")
        print(f"    {content[:200]}")
        if len(content) > 200:
            print(f"    ... ({len(content)} chars total)")
        print()

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show GoneBy status."""
    import asyncio

    from goneby.memory.embedder import Embedder, EmbeddingConfig

    config = GoneByConfig.from_env()

    print("GoneBy Status")
    print("=" * 46)
    print("Version:     0.1.0")
    print(f"Project:     {config.project}")
    print(f"WAL path:    {config.wal_path}")
    print(f"Vector:      {config.vector_backend}")
    print(f"Embedding:   {config.embedding_model}")
    print(f"  endpoint:  {config.embedding_base_url}")
    print(f"  dims (MRL): {config.embedding_dimensions}")
    print("Encryption:  ChaCha20-Poly1305")

    if config.encryption_key:
        cipher = Cipher(config.encryption_key)
        wal = WALManager(config.wal_path, cipher)
        print(f"WAL entries: {wal.entry_count}")
    else:
        print("WAL entries: (GONEBY_ENCRYPTION_KEY not set)")

    # Live embedder probe — never fatal.
    async def probe() -> None:
        embedder = Embedder(
            EmbeddingConfig(
                base_url=config.embedding_base_url,
                model=config.embedding_model,
                api_key=config.embedding_api_key,
                dimensions=config.embedding_dimensions,
            )
        )
        report = await embedder.health()
        await embedder.aclose()
        if report.get("ok"):
            print(f"Embedder:    UP via {report.get('endpoint')}")
            models = report.get("models") or []
            if models:
                print(f"  serving:   {', '.join(models)}")
        else:
            print(f"Embedder:    DOWN — {report.get('detail')}")
            print("  start one with:")
            print(
                "    llama-server -m qwen3-embedding-0.6b.gguf --embeddings "
                "--port 8081"
            )

    try:
        asyncio.run(probe())
    except Exception as exc:  # noqa: BLE001 - status must never crash
        print(f"Embedder:    probe failed — {exc}")

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the MCP server."""
    from goneby.mcp_server import main as serve_main
    return serve_main()


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="goneby",
        description="Retroactive project memory for LLM agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # generate-key
    gen_parser = subparsers.add_parser("generate-key", help="Generate encryption key")
    gen_parser.set_defaults(func=cmd_generate_key)

    # append
    append_parser = subparsers.add_parser("append", help="Append entry to WAL")
    append_parser.add_argument("content", nargs="?", help="Content to append")
    append_parser.add_argument("-f", "--file", help="Read content from file")
    append_parser.add_argument("-t", "--type", help="Entry type (decision, finding, note, constraint)")
    append_parser.set_defaults(func=cmd_append)

    # read
    read_parser = subparsers.add_parser("read", help="Read all WAL entries")
    read_parser.set_defaults(func=cmd_read)

    # status
    status_parser = subparsers.add_parser("status", help="Show GoneBy status")
    status_parser.set_defaults(func=cmd_status)

    # serve
    serve_parser = subparsers.add_parser("serve", help="Run MCP server")
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
