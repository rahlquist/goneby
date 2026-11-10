"""Tests for the Hermes plugin adapter.

A fake ``ctx`` stands in for Hermes' PluginContext, recording registrations and
providing a stub host LLM. The goneby library, crypto, WAL, and embedder are real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goneby.crypto.cipher import Cipher

# --------------------------------------------------------------------------
# Fake Hermes plugin context
# --------------------------------------------------------------------------


class FakeLlmResult:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class FakeLlm:
    def __init__(self, parsed=None, text=""):
        self._parsed = parsed
        self._text = text
        self.calls: list[dict] = []

    def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        return FakeLlmResult(parsed=self._parsed, text=self._text)


class FakeCtx:
    def __init__(self, llm: FakeLlm | None = None):
        self.tools: dict[str, dict] = {}
        self.hooks: dict[str, object] = {}
        self.llm = llm or FakeLlm()
        self.commands: dict[str, object] = {}

    def register_tool(self, name, toolset, schema, handler, **kwargs):
        self.tools[name] = {
            "toolset": toolset,
            "schema": schema,
            "handler": handler,
            "kwargs": kwargs,
        }

    def register_hook(self, hook_name, callback, **kwargs):
        self.hooks[hook_name] = callback

    def register_command(self, name, **kwargs):
        self.commands[name] = kwargs


def make_plugin(monkeypatch, tmp_path, ctx=None, **env) -> tuple:
    """Build a plugin with a temp config and a fake Hermes context."""
    monkeypatch.setenv("GONEBY_ENCRYPTION_KEY", Cipher.generate_key().hex())
    monkeypatch.setenv("GONEBY_WAL_PATH", str(tmp_path / "wal"))
    monkeypatch.setenv("GONEBY_PROJECT", env.get("GONEBY_PROJECT", "test-proj"))
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))

    from plugin.adapter import GoneByPlugin

    ctx = ctx or FakeCtx()
    plugin = GoneByPlugin(ctx)
    plugin.register()
    return plugin, ctx


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


class TestRegistration:
    def test_registers_declared_tools(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        assert set(ctx.tools) == {
            "wall_append",
            "wall_read",
            "walk_backward",
            "store_memory",
            "query_memory",
            "goneby_status",
        }

    def test_registers_declared_hooks(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        assert set(ctx.hooks) == {"pre_llm_call", "post_compaction"}

    def test_schemas_declare_required_args(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        assert ctx.tools["wall_append"]["schema"]["parameters"]["required"] == ["content"]
        assert ctx.tools["walk_backward"]["schema"]["parameters"]["required"] == ["mvp_text"]
        assert ctx.tools["query_memory"]["schema"]["parameters"]["required"] == ["query"]

    def test_entry_type_enum_is_declared(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        props = ctx.tools["wall_append"]["schema"]["parameters"]["properties"]
        assert set(props["entry_type"]["enum"]) == {
            "decision",
            "finding",
            "note",
            "constraint",
        }

    def test_missing_key_still_registers(self, monkeypatch, tmp_path):
        """A missing key must not prevent registration — status stays usable."""
        monkeypatch.delenv("GONEBY_ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("GONEBY_WAL_PATH", str(tmp_path / "wal"))

        from plugin.adapter import GoneByPlugin

        ctx = FakeCtx()
        GoneByPlugin(ctx).register()
        assert set(ctx.tools) == {
            "wall_append",
            "wall_read",
            "walk_backward",
            "store_memory",
            "query_memory",
            "goneby_status",
        }


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


class TestCapture:
    def _history(self, *turns):
        return [
            {"role": role, "content": text, "message_uid": f"uid-{i}"}
            for i, (role, text) in enumerate(turns)
        ]

    def test_pre_llm_call_captures_new_messages(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        hook = ctx.hooks["pre_llm_call"]

        hook(
            session_id="s1",
            user_message="let's build it",
            conversation_history=self._history(
                ("user", "build a memory system"),
                ("assistant", "starting now"),
            ),
            is_first_turn=False,
            model="test",
            platform="cli",
        )

        out = json.loads(ctx.tools["wall_read"]["handler"]({}))
        assert out["count"] == 2

    def test_capture_is_idempotent_per_message(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        hook = ctx.hooks["pre_llm_call"]
        history = self._history(("user", "same message"))

        for _ in range(3):
            hook(
                session_id="s1",
                user_message="same message",
                conversation_history=history,
                is_first_turn=False,
                model="test",
                platform="cli",
            )

        out = json.loads(ctx.tools["wall_read"]["handler"]({}))
        assert out["count"] == 1

    def test_capture_appends_incrementally(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        hook = ctx.hooks["pre_llm_call"]

        hook(
            session_id="s1",
            user_message="a",
            conversation_history=self._history(("user", "first")),
            is_first_turn=True,
            model="test",
            platform="cli",
        )
        hook(
            session_id="s1",
            user_message="b",
            conversation_history=self._history(
                ("user", "first"), ("assistant", "second")
            ),
            is_first_turn=False,
            model="test",
            platform="cli",
        )

        out = json.loads(ctx.tools["wall_read"]["handler"]({}))
        assert out["count"] == 2
        assert [e["content"] for e in out["entries"]] == ["first", "second"]

    def test_system_and_tool_messages_are_not_captured(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        ctx.hooks["pre_llm_call"](
            session_id="s1",
            user_message="x",
            conversation_history=[
                {"role": "system", "content": "you are a helpful agent", "message_uid": "s1"},
                {"role": "tool", "content": '{"output": "noise"}', "message_uid": "t1"},
                {"role": "user", "content": "real content", "message_uid": "u1"},
            ],
            is_first_turn=True,
            model="test",
            platform="cli",
        )
        out = json.loads(ctx.tools["wall_read"]["handler"]({}))
        assert [e["content"] for e in out["entries"]] == ["real content"]

    def test_compaction_boundary_is_recorded(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        ctx.hooks["post_compaction"](
            session_id="s1", kind="summarize", reason="overflow", messages_removed=40
        )
        out = json.loads(ctx.tools["wall_read"]["handler"]({}))
        assert any("compaction boundary" in e["content"] for e in out["entries"])

    def test_auto_capture_can_be_disabled(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(
            monkeypatch, tmp_path, GONEBY_AUTO_CAPTURE="false"
        )
        ctx.hooks["pre_llm_call"](
            session_id="s1",
            user_message="x",
            conversation_history=self._history(("user", "should not be stored")),
            is_first_turn=True,
            model="test",
            platform="cli",
        )
        out = json.loads(ctx.tools["wall_read"]["handler"]({}))
        assert out["count"] == 0

    def test_capture_failure_does_not_raise(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        hook = ctx.hooks["pre_llm_call"]
        # A non-list history must not blow up the turn.
        hook(
            session_id="s1",
            user_message="x",
            conversation_history="not a list",
            is_first_turn=True,
            model="test",
            platform="cli",
        )


# --------------------------------------------------------------------------
# Recall
# --------------------------------------------------------------------------


class TestRecall:
    def test_recall_off_by_default(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        result = ctx.hooks["pre_llm_call"](
            session_id="s1",
            user_message="a sufficiently long question about the project",
            conversation_history=[],
            is_first_turn=True,
            model="test",
            platform="cli",
        )
        assert result is None

    def test_short_turns_are_not_searched(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path, GONEBY_AUTO_RECALL="true")
        result = ctx.hooks["pre_llm_call"](
            session_id="s1",
            user_message="ok",
            conversation_history=[],
            is_first_turn=True,
            model="test",
            platform="cli",
        )
        assert result is None


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


class TestTools:
    def test_wall_append_and_read_roundtrip(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        out = json.loads(
            ctx.tools["wall_append"]["handler"](
                {"content": "chose qdrant", "entry_type": "decision"}
            )
        )
        assert out["status"] == "ok"
        assert out["wal_entry_count"] == 1

        read = json.loads(ctx.tools["wall_read"]["handler"]({}))
        assert read["entries"][0]["content"] == "chose qdrant"
        assert read["entries"][0]["metadata"]["type"] == "decision"

    def test_append_rejects_empty_content(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        out = json.loads(ctx.tools["wall_append"]["handler"]({"content": "   "}))
        assert out["status"] == "error"

    def test_append_rejects_unknown_type(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        out = json.loads(
            ctx.tools["wall_append"]["handler"](
                {"content": "x", "entry_type": "vibes"}
            )
        )
        assert out["status"] == "error"
        assert "entry_type" in out["error"]

    def test_wal_is_encrypted_on_disk(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        ctx.tools["wall_append"]["handler"]({"content": "PLAINTEXT_CANARY"})
        wal_file = tmp_path / "wal" / "test-proj.wal"
        assert wal_file.exists()
        assert "PLAINTEXT_CANARY" not in wal_file.read_text()

    def test_projects_write_separate_files(self, monkeypatch, tmp_path):
        _, ctx_a = make_plugin(monkeypatch, tmp_path, GONEBY_PROJECT="alpha")
        _, ctx_b = make_plugin(monkeypatch, tmp_path, GONEBY_PROJECT="beta")

        ctx_a.tools["wall_append"]["handler"]({"content": "alpha secret"})
        ctx_b.tools["wall_append"]["handler"]({"content": "beta secret"})

        a = (tmp_path / "wal" / "alpha.wal").read_text()
        b = (tmp_path / "wal" / "beta.wal").read_text()
        assert "alpha secret" not in b
        assert "beta secret" not in a

    def test_status_reports_project_and_config(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        out = json.loads(ctx.tools["goneby_status"]["handler"]({}))
        assert out["status"] == "ok"
        assert out["project"] == "test-proj"
        assert out["wal_entry_count"] == 0
        assert out["encryption"] == "ChaCha20-Poly1305"
        assert out["auto_capture"] is True
        assert out["auto_recall"] is False

    def test_status_without_key_reports_error_not_crash(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GONEBY_ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("GONEBY_WAL_PATH", str(tmp_path / "wal"))

        from plugin.adapter import GoneByPlugin

        ctx = FakeCtx()
        GoneByPlugin(ctx).register()
        out = json.loads(ctx.tools["goneby_status"]["handler"]({}))
        assert out["status"] == "error"
        assert "GONEBY_ENCRYPTION_KEY" in out["error"]

    def test_tools_report_error_without_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GONEBY_ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("GONEBY_WAL_PATH", str(tmp_path / "wal"))

        from plugin.adapter import GoneByPlugin

        ctx = FakeCtx()
        GoneByPlugin(ctx).register()
        out = json.loads(ctx.tools["wall_append"]["handler"]({"content": "x"}))
        assert out["status"] == "error"

    def test_walk_backward_uses_host_llm(self, monkeypatch, tmp_path):
        """The judge must go through ctx.llm, not a hardcoded provider."""
        llm = FakeLlm(parsed=[True, False, True])
        ctx = FakeCtx(llm=llm)
        _, ctx = make_plugin(monkeypatch, tmp_path, ctx=ctx)

        for text in ["keep one", "drop this", "keep two"]:
            ctx.tools["wall_append"]["handler"]({"content": text})

        out = json.loads(
            ctx.tools["walk_backward"]["handler"]({"mvp_text": "shipped thing"})
        )
        assert out["status"] == "ok"
        assert out["kept_count"] == 2
        assert out["discarded_count"] == 1
        assert llm.calls, "host LLM was never consulted"
        assert "json_schema" in llm.calls[0]

    def test_walk_backward_handles_unparseable_judge(self, monkeypatch, tmp_path):
        llm = FakeLlm(parsed=None, text="I refuse to answer")
        ctx = FakeCtx(llm=llm)
        _, ctx = make_plugin(monkeypatch, tmp_path, ctx=ctx)
        ctx.tools["wall_append"]["handler"]({"content": "only entry"})

        out = json.loads(
            ctx.tools["walk_backward"]["handler"]({"mvp_text": "mvp"})
        )
        # Fail open: never lose evidence to a bad judge response.
        assert out["kept_count"] == 1

    def test_walk_backward_on_empty_wal(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        out = json.loads(
            ctx.tools["walk_backward"]["handler"]({"mvp_text": "mvp"})
        )
        assert out["status"] == "empty"

    def test_store_and_query_roundtrip(self, monkeypatch, tmp_path):
        """Store then query through the plugin, against a live embedder.

        The fake server runs on its own thread with its own event loop. If it
        shared the main loop, the plugin's blocking ``future.result()`` would
        starve the server and deadlock the test.
        """
        import socket
        import threading

        from tests.test_memory import make_fake_app

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        base_url = f"http://127.0.0.1:{port}/v1"

        import uvicorn

        config = uvicorn.Config(
            make_fake_app(), host="127.0.0.1", port=port, log_level="error"
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(100):
            if server.started:
                break
            import time

            time.sleep(0.05)

        try:
            monkeypatch.setenv("GONEBY_ENCRYPTION_KEY", Cipher.generate_key().hex())
            monkeypatch.setenv("GONEBY_WAL_PATH", str(tmp_path / "wal"))
            monkeypatch.setenv("GONEBY_PROJECT", "test-proj")
            monkeypatch.setenv("GONEBY_EMBEDDING_BASE_URL", base_url)
            monkeypatch.setenv("GONEBY_VECTOR_BACKEND", "memory")

            from plugin.adapter import GoneByPlugin

            ctx = FakeCtx()
            GoneByPlugin(ctx).register()

            ctx.tools["wall_append"]["handler"](
                {"content": "chose qdrant for the vector backend",
                 "entry_type": "decision"}
            )
            stored = json.loads(
                ctx.tools["store_memory"]["handler"](
                    {"entries": [
                        {"content": "chose qdrant for the vector backend",
                         "metadata": {"type": "decision"}}
                    ]}
                )
            )
            assert stored["status"] == "ok", stored
            assert stored["stored"] == 1
            assert stored["project"] == "test-proj"

            hits = json.loads(
                ctx.tools["query_memory"]["handler"](
                    {"query": "vector backend", "top_k": 3}
                )
            )
            assert hits["status"] == "ok"
            assert hits["scope"] == "project"
            assert hits["count"] >= 1
        finally:
            server.should_exit = True
            thread.join(timeout=5)

    def test_store_reports_error_when_backend_down(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(
            monkeypatch, tmp_path, GONEBY_EMBEDDING_BASE_URL="http://127.0.0.1:1/v1"
        )
        out = json.loads(
            ctx.tools["store_memory"]["handler"](
                {"entries": [{"content": "x", "metadata": {}}]}
            )
        )
        assert out["status"] == "error"

    def test_query_rejects_empty_query(self, monkeypatch, tmp_path):
        _, ctx = make_plugin(monkeypatch, tmp_path)
        out = json.loads(ctx.tools["query_memory"]["handler"]({"query": "  "}))
        assert out["status"] == "error"


# --------------------------------------------------------------------------
# Message helpers
# --------------------------------------------------------------------------


class TestMessageText:
    def test_string_content(self):
        from plugin.adapter import message_text

        assert message_text({"content": "hello"}) == "hello"

    def test_list_content_extracts_text_parts(self):
        from plugin.adapter import message_text

        msg = {
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image", "source": "..."},
                {"type": "text", "text": "second"},
            ]
        }
        assert message_text(msg) == "first\nsecond"

    def test_multimodal_user_message(self):
        from plugin.adapter import message_text

        msg = {
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ]
        }
        assert message_text(msg) == "what is this"

    def test_none_content(self):
        from plugin.adapter import message_text

        assert message_text({}) == ""

    def test_identity_prefers_uid(self):
        from plugin.adapter import message_identity

        a = message_identity({"role": "user", "content": "x", "message_uid": "abc"})
        b = message_identity({"role": "user", "content": "x", "message_uid": "abc"})
        assert a == b == "uid:abc"

    def test_identity_falls_back_to_content_hash(self):
        from plugin.adapter import message_identity

        a = message_identity({"role": "user", "content": "same"})
        b = message_identity({"role": "assistant", "content": "same"})
        assert a != b  # role is part of the identity
