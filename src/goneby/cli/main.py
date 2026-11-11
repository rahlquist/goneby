"""CLI entry point for GoneBy."""

from __future__ import annotations

import argparse
import sys

from goneby import __version__
from goneby.config import policy
from goneby.config.settings import GoneByConfig, project_wal_path
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
    wal_file = project_wal_path(config.wal_path, config.project)
    wal = WALManager(wal_file, cipher)

    content = args.content
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()

    entry_type = args.type or "note"
    wal.append(content, {"type": entry_type})

    print(f"Entry appended to {wal_file}")
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
    wal_file = project_wal_path(config.wal_path, config.project)
    wal = WALManager(wal_file, cipher)

    entries = wal.read_all()
    print(f"WAL: {wal_file}")
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
    print(f"Version:     {__version__}")
    print(f"Project:     {config.project}")
    print(f"WAL base:    {config.wal_path}")
    print(f"Project WAL: {project_wal_path(config.wal_path, config.project)}")
    print(f"Vector:      {config.vector_backend}")
    print(f"Embedding:   {config.embedding_model}")
    print(f"  endpoint:  {config.embedding_base_url}")
    print(f"  dims (MRL): {config.embedding_dimensions}")
    print("Encryption:  ChaCha20-Poly1305")

    if config.encryption_key:
        cipher = Cipher(config.encryption_key)
        wal = WALManager(project_wal_path(config.wal_path, config.project), cipher)
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


def cmd_project(args: argparse.Namespace) -> int:
    """Enable/disable/inspect capture for one project."""
    from pathlib import Path

    cfg = GoneByConfig.from_env()
    slug = args.project or cfg.project
    root = Path(args.path).expanduser() if args.path else None

    if args.action in ("enable", "disable"):
        enabled = args.action == "enable"
        written = None
        if root is not None and root.is_dir():
            try:
                written = f"marker: {policy.write_marker(root, enabled)}"
            except OSError:
                written = None
        if written is None:
            user_cfg = policy.load_config()
            policy.set_project_override(user_cfg, slug, enabled)
            written = f"user config: projects[{slug}]"
        print(f"Capture {args.action}d for project '{slug}' -> {written}")
        return 0

    # status
    user_cfg = policy.load_config()
    marker = policy.read_marker(root)
    override = policy.project_override(user_cfg, slug)
    mode = policy.effective_mode(user_cfg)
    verdict = policy.capture_allowed(
        slug,
        "",
        mode=mode,
        disabled_platforms=policy.effective_disabled_platforms(user_cfg),
        master=policy.master_enabled(user_cfg),
        marker=marker,
        override=override,
    )
    print(f"Project:   {slug}")
    print(f"Root:      {root if root else '(cwd-derived at capture time)'}")
    print(f"Marker:    {marker if marker is not None else 'none'}")
    print(f"Override:  {override if override is not None else 'none'}")
    print(f"Mode:      {mode}")
    print(f"Capturing: {verdict}")
    return 0


def cmd_mode(args: argparse.Namespace) -> int:
    cfg = policy.load_config()
    cfg["mode"] = args.mode
    policy.save_config(cfg)
    detail = (
        "capture everything not excluded by gate/platform/marker/override"
        if args.mode == policy.MODE_ALL
        else "capture only explicitly enabled projects"
    )
    print(f"Capture mode set to '{args.mode}' ({detail}).")
    return 0


def cmd_platform(args: argparse.Namespace) -> int:
    if args.action == "list":
        disabled = sorted(policy.effective_disabled_platforms(policy.load_config()))
        print("Disabled platforms: " + (", ".join(disabled) if disabled else "(none)"))
        return 0
    if not args.name:
        print(f"Configuration errors:\n  - platform {args.action} requires a platform id")
        return 1
    cfg = policy.load_config()
    cfg = policy.set_platform(args.name, args.action == "enable")
    disabled = sorted(cfg.get("disabled_platforms", []))
    print(f"Platform '{args.name}' capture {args.action}d. Disabled list: {', '.join(disabled) or '(none)'}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    enabled = args.state == "on"
    policy.set_master_enabled(enabled)
    print(f"Master capture gate {args.state}: capture is now "
          f"{'possible everywhere not excluded' if enabled else 'OFF for every project and gateway'}.")
    return 0


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

    # project
    project_parser = subparsers.add_parser(
        "project", help="Enable/disable/inspect capture for a project"
    )
    project_parser.add_argument(
        "action", choices=["enable", "disable", "status"], help="What to do"
    )
    project_parser.add_argument("path", nargs="?", default="", help="Project root path")
    project_parser.add_argument("--project", default="", help="Project slug (override target)")
    project_parser.set_defaults(func=cmd_project)

    # mode
    mode_parser = subparsers.add_parser("mode", help="Set capture mode: all | opt-in")
    mode_parser.add_argument("mode", choices=list(policy.MODES), help="Capture mode")
    mode_parser.set_defaults(func=cmd_mode)

    # platform
    platform_parser = subparsers.add_parser(
        "platform", help="Per-gateway capture kill (e.g. telegram)"
    )
    platform_parser.add_argument("action", choices=["enable", "disable", "list"])
    platform_parser.add_argument("name", nargs="?", default="", help="Platform id")
    platform_parser.set_defaults(func=cmd_platform)

    # gate
    gate_parser = subparsers.add_parser("gate", help="Master capture switch")
    gate_parser.add_argument("state", choices=["on", "off"], help="Capture everywhere or nowhere")
    gate_parser.set_defaults(func=cmd_gate)

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
