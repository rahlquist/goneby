"""Tests for the MCP tool surface.

Constructs the real server and calls the real tools. The vector store is
patched so no external service is needed; the WAL and cipher are real.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from goneby.config.settings import GoneByConfig, project_wal_path
from goneby.mcp_server import create_mcp_server, load_mcp_server_class


def run(coro):
    return asyncio.run(coro)


def payload(result) -> dict:
    """Extract the JSON body from an MCP CallToolResult."""
    obj = result[1] if isinstance(result, tuple) else result
    for block in obj.content:
        return json.loads(block.text)
    raise AssertionError("tool returned no content blocks")


def make_server(tmp_path, **overrides):
    """Build a real server against a real per-project WAL on a temp path."""
    cfg = GoneByConfig(
        encryption_key=_new_key(),
        wal_path=tmp_path / "wal",
        **overrides,
    )
    return create_mcp_server(cfg), cfg


def wal_file(cfg: GoneByConfig):
    """Resolve the per-project WAL file the server actually writes."""
    return project_wal_path(cfg.wal_path, cfg.project)


def _new_key() -> bytes:
    from goneby.crypto.cipher import Cipher

    return Cipher.generate_key()


class TestLLMProviderConfig:
    """ctx-audit: the MCP server hardcoded the Anthropic interface with one
    fixed model, and the package was not a declared dependency — a fresh
    install could not walk, and the 'no provider hard-coded' promise was
    false. Provider must come from config; unset must be an actionable
    error, not an ImportError."""

    def test_walk_without_llm_config_returns_actionable_error(self, tmp_path):
        server, _ = make_server(tmp_path)
        run(server.call_tool("wall_append", {"content": "a log line"}))
        out = payload(run(server.call_tool("walk_backward", {"mvp_text": "mvp"})))
        assert out["status"] == "error"
        assert out["stage"] == "judge"
        assert "GONEBY_LLM_BASE_URL" in out["error"]

    def test_no_vendor_or_model_default(self):
        cfg = GoneByConfig()
        assert cfg.llm_model == ""
        assert cfg.llm_provider != "anthropic"

    def test_openai_compat_endpoint_receives_prompt(self):
        import asyncio
        import socket

        import uvicorn
        from fastapi import FastAPI

        from goneby.mcp_server import create_llm_fn

        received: dict = {}
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat(body: dict):
            received["payload"] = body
            return {"choices": [{"message": {"content": "[true, false]"}}]}

        async def go():
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            ucfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
            srv = uvicorn.Server(ucfg)
            task = asyncio.create_task(srv.serve())
            for _ in range(100):
                if srv.started:
                    break
                await asyncio.sleep(0.05)
            try:
                cfg = GoneByConfig(
                    llm_base_url=f"http://127.0.0.1:{port}/v1",
                    llm_model="test-judge",
                    llm_api_key="k",
                )
                out = await create_llm_fn(cfg)("judge this")
                assert out == "[true, false]"
                assert received["payload"]["model"] == "test-judge"
                assert received["payload"]["messages"][0]["content"] == "judge this"
            finally:
                srv.should_exit = True
                await task

        run(go())

    def test_anthropic_requires_declared_extra(self):
        from pathlib import Path
        from tomllib import loads as toml_loads

        data = toml_loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
        assert "anthropic" in data["project"]["optional-dependencies"], (
            "provider may be opt-in, but the dependency must be declared"
        )

    def test_anthropic_path_names_the_extra_when_missing(self):
        try:
            import anthropic  # noqa: F401

            pytest.skip("anthropic installed in this env; ImportError path unreachable")
        except ImportError:
            pass
        from goneby.mcp_server import create_llm_fn

        cfg = GoneByConfig(llm_provider="anthropic", llm_model="claude-x")

        async def go():
            with pytest.raises(RuntimeError, match=r"goneby\[anthropic\]"):
                await create_llm_fn(cfg)("p")

        run(go())


class TestServerConstruction:
    def test_server_class_loads(self):
        assert load_mcp_server_class().__name__ in {"MCPServer", "FastMCP"}

    def test_exposes_seven_tools(self, tmp_path):
        server, _ = make_server(tmp_path)
        tools = run(server.list_tools())
        names = {t.name for t in tools}
        assert names == {
            "wall_append",
            "wall_read",
            "walk_backward",
            "store_memory",
            "query_memory",
            "generate_key",
            "status",
        }

    def test_tool_schemas_declare_required_args(self, tmp_path):
        server, _ = make_server(tmp_path)
        tools = {t.name: t for t in run(server.list_tools())}
        assert tools["wall_append"].input_schema["required"] == ["content"]
        assert tools["walk_backward"].input_schema["required"] == ["mvp_text"]
        assert tools["query_memory"].input_schema["required"] == ["query"]

    def test_missing_key_is_rejected(self, tmp_path):
        cfg = GoneByConfig(encryption_key=b"", wal_path=tmp_path / "x")
        with pytest.raises(ValueError, match="encryption_key"):
            create_mcp_server(cfg)

    def test_bad_key_length_is_rejected(self, tmp_path):
        cfg = GoneByConfig(encryption_key=b"tooshort", wal_path=tmp_path / "x")
        with pytest.raises(ValueError, match="32 bytes"):
            create_mcp_server(cfg)


class TestWALTools:
    def test_append_then_read(self, tmp_path):
        server, _ = make_server(tmp_path)

        async def go():
            await server.call_tool("wall_append", {"content": "first", "entry_type": "decision"})
            await server.call_tool("wall_append", {"content": "second", "entry_type": "note"})
            out = payload(await server.call_tool("wall_read", {}))
            assert out["count"] == 2
            assert [e["content"] for e in out["entries"]] == ["first", "second"]
            assert out["entries"][0]["metadata"]["type"] == "decision"

        run(go())

    def test_read_empty_wal(self, tmp_path):
        server, _ = make_server(tmp_path)
        out = payload(run(server.call_tool("wall_read", {})))
        assert out["count"] == 0

    def test_wal_file_is_encrypted_on_disk(self, tmp_path):
        server, cfg = make_server(tmp_path)
        run(server.call_tool("wall_append", {"content": "PLAINTEXT_CANARY"}))
        raw = wal_file(cfg).read_text()
        assert "PLAINTEXT_CANARY" not in raw

    def test_projects_do_not_share_a_wal(self, tmp_path):
        """Two projects against one server config must write separate files."""
        server_a, cfg_a = make_server(tmp_path, project="alpha")
        server_b, cfg_b = make_server(tmp_path, project="beta")

        run(server_a.call_tool("wall_append", {"content": "alpha secret"}))
        run(server_b.call_tool("wall_append", {"content": "beta secret"}))

        a_raw = wal_file(cfg_a).read_text()
        b_raw = wal_file(cfg_b).read_text()
        assert "alpha secret" not in b_raw
        assert "beta secret" not in a_raw
        assert wal_file(cfg_a) != wal_file(cfg_b)

    def test_walk_on_empty_wal_returns_status(self, tmp_path):
        server, _ = make_server(tmp_path)
        out = payload(run(server.call_tool("walk_backward", {"mvp_text": "some mvp"})))
        assert out["status"] == "empty"
        assert out["kept_count"] == 0


class TestKeyAndStatus:
    def test_generate_key_shape(self, tmp_path):
        server, _ = make_server(tmp_path)
        out = payload(run(server.call_tool("generate_key", {})))
        assert out["key_size_bytes"] == 32
        assert out["algorithm"] == "ChaCha20-Poly1305"
        assert len(bytes.fromhex(out["key_hex"])) == 32

    def test_status_reports_embedding_config(self, tmp_path):
        server, _ = make_server(tmp_path, embedding_model="test-embed", embedding_dimensions=256)
        out = payload(run(server.call_tool("status", {})))
        assert out["embedding"]["model"] == "test-embed"
        assert out["embedding"]["dimensions"] == 256
        assert out["encryption"] == "ChaCha20-Poly1305"
        assert "embedder" in out["health"]

    def test_status_never_raises_when_backends_down(self, tmp_path):
        server, _ = make_server(tmp_path)
        out = payload(run(server.call_tool("status", {})))
        assert "health" in out
        assert "wal_entry_count" in out


class TestFailurePaths:
    def test_query_memory_reports_stage_not_crash(self, tmp_path):
        """With no embedder or vector store up, the tool must still answer."""
        server, _ = make_server(tmp_path)
        out = payload(run(server.call_tool("query_memory", {"query": "anything"})))
        assert out["status"] == "error"
        assert out["stage"] in {"embedding", "vector_store"}
        assert out["results"] == []

    def test_store_memory_reports_stage_not_crash(self, tmp_path):
        server, _ = make_server(tmp_path)
        out = payload(
            run(
                server.call_tool(
                    "store_memory", {"entries": [{"content": "x", "metadata": {}}]}
                )
            )
        )
        assert out["status"] == "error"
        assert out["stage"] in {"embedding", "vector_store"}
        assert out["stored"] == 0
