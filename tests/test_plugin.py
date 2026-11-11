"""Tests for the Hermes plugin adapter.

A fake ``ctx`` stands in for Hermes' PluginContext, recording registrations and
providing a stub host LLM. The goneby library, crypto, WAL, and embedder are real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
    def __init__(self, llm: Any | None = None):
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

    def register_command(self, name, handler=None, **kwargs):
        self.commands[name] = handler


def make_plugin(monkeypatch, tmp_path, ctx=None, **env) -> tuple:
    """Build a plugin with a temp config and a fake Hermes context."""
    monkeypatch.setenv("GONEBY_ENCRYPTION_KEY", Cipher.generate_key().hex())
    monkeypatch.setenv("GONEBY_WAL_PATH", str(tmp_path / "wal"))
    monkeypatch.setenv("GONEBY_PROJECT", env.get("GONEBY_PROJECT", "test-proj"))
    monkeypatch.setenv("GONEBY_CONFIG_PATH", str(tmp_path / "goneby-config.json"))
    monkeypatch.setenv("GONEBY_HERMES_STATE_DB", str(tmp_path / "absent-state.db"))
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


class TestJudgeHostContract:
    """The host's complete_structured takes typed blocks, not raw strings.

    The fake LLM in older tests accepted anything, which hid a ValueError
    ('Unsupported input block: str') on every real walk_backward. Pin the
    exact shape agent/plugin_llm.py requires.
    """

    def test_judge_passes_text_blocks(self, monkeypatch, tmp_path):
        llm = FakeLlm(parsed=[True])
        plugin, ctx = make_plugin(monkeypatch, tmp_path, ctx=FakeCtx(llm=llm))
        handler = ctx.tools["wall_append"]["handler"]
        handler({"content": "decided: use WAL", "entry_type": "decision"}, session_id="s1")
        plugin._tool_walk_backward({"mvp_text": "shipped thing"}, session_id="s1")
        assert llm.calls, "host LLM was never consulted"
        blocks = llm.calls[-1]["input"]
        assert isinstance(blocks, list) and len(blocks) == 1
        assert isinstance(blocks[0], dict)
        assert blocks[0]["type"] == "text"
        assert isinstance(blocks[0]["text"], str) and "decided: use WAL" in blocks[0]["text"]
        assert "shipped thing" in blocks[0]["text"]


class TestJudgeBatching:
    """Large WALs must be judged in bounded batches, fail-open per batch."""

    class ScriptedLlm:
        def __init__(self, seq):
            self.seq = list(seq)
            self.calls = []

        def complete_structured(self, **kwargs):
            self.calls.append(kwargs)
            return FakeLlmResult(parsed=self.seq.pop(0))

    @staticmethod
    def _append(ctx, session_id, contents):
        handler = ctx.tools["wall_append"]["handler"]
        for c in contents:
            handler({"content": c, "entry_type": "note"}, session_id=session_id)

    def test_multiple_batches_align(self, monkeypatch, tmp_path):
        llm = self.ScriptedLlm([[True, False], [True]])
        plugin, ctx = make_plugin(monkeypatch, tmp_path, ctx=FakeCtx(llm=llm))
        monkeypatch.setenv("GONEBY_JUDGE_BATCH", "2")
        self._append(ctx, "s1", ["kept one", "dead end", "kept two"])
        import json as _json
        result = _json.loads(
            plugin._tool_walk_backward({"mvp_text": "shipped"}, session_id="s1")
        )
        assert result["status"] == "ok"
        assert len(llm.calls) == 2, "3 entries at batch=2 must make two judge calls"
        assert result["kept_count"] == 2
        assert result["discarded_count"] == 1

    def test_batch_failure_keeps_only_that_batch(self, monkeypatch, tmp_path):
        class BoomThenOk:
            def __init__(self):
                self.n = 0

            def complete_structured(self, **kwargs):
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("APITimeoutError: Request timed out.")
                return FakeLlmResult(parsed=[False])

        llm = BoomThenOk()
        plugin, ctx = make_plugin(monkeypatch, tmp_path, ctx=FakeCtx(llm=llm))
        monkeypatch.setenv("GONEBY_JUDGE_BATCH", "1")
        self._append(ctx, "s1", ["boom batch", "judged batch"])
        import json as _json
        result = _json.loads(
            plugin._tool_walk_backward({"mvp_text": "shipped"}, session_id="s1")
        )
        assert result["status"] == "ok"
        assert result["kept_count"] == 1
        # Batches arrive newest-first (walker pairs flags with that order):
        # batch 1 = "judged batch" raises -> fail-open KEEP; batch 2 = "boom
        # batch" answers False -> discarded.
        kept_texts = [e["content"] for e in result["kept_entries"]]
        assert kept_texts == ["judged batch"]

    def test_garbage_env_falls_back_to_default(self, monkeypatch, tmp_path):
        llm = FakeLlm(parsed=[True])
        plugin, ctx = make_plugin(monkeypatch, tmp_path, ctx=FakeCtx(llm=llm))
        monkeypatch.setenv("GONEBY_JUDGE_BATCH", "not-a-number")
        self._append(ctx, "s1", ["only entry"])
        import json as _json
        result = _json.loads(
            plugin._tool_walk_backward({"mvp_text": "shipped"}, session_id="s1")
        )
        assert result["status"] == "ok"


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
            "goneby_project",
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
            "goneby_project",
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


# --------------------------------------------------------------------------
# Capture policy: opt-in mode, project resolution, /goneby control
# --------------------------------------------------------------------------


class TestCapturePolicy:
    @staticmethod
    def _history(*pairs):
        return [{"role": r, "content": c, "message_uid": f"{r}-{c}"} for r, c in pairs]

    @staticmethod
    def _make_state_db(tmp_path, rows):
        import sqlite3

        db = tmp_path / "state.db"
        con = sqlite3.connect(db)
        con.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT, git_repo_root TEXT)"
        )
        con.executemany("INSERT INTO sessions VALUES (?, ?, ?)", rows)
        con.commit()
        con.close()
        return db

    def _capture_hook(self, ctx, session_id="s1", platform="cli"):
        ctx.hooks["pre_llm_call"](
            session_id=session_id,
            user_message="x",
            conversation_history=self._history(("user", "policy probe")),
            is_first_turn=True,
            model="test",
            platform=platform,
        )

    def test_opt_in_mode_captures_nothing_without_marker(self, monkeypatch, tmp_path):
        plugin, ctx = make_plugin(monkeypatch, tmp_path, GONEBY_CAPTURE_MODE="opt-in")
        self._capture_hook(ctx)
        assert not plugin._get_wal().read_all()
        assert not plugin._wal_path.exists()

    def test_enable_writes_marker_and_capture_flows(self, monkeypatch, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        _plugin, ctx = make_plugin(monkeypatch, tmp_path, GONEBY_CAPTURE_MODE="opt-in")

        out = ctx.tools["goneby_project"]["handler"](
            {"action": "enable", "path": str(repo)}, session_id="s1"
        )
        import json as _json

        verdict = _json.loads(out)
        assert verdict["status"] == "ok"
        assert (repo / ".goneby.json").exists()
        # marker at repo path, but session id is unknown => slug stays default project
        assert verdict["written_to"].endswith(".goneby.json")

    def test_project_slug_comes_from_state_db(self, monkeypatch, tmp_path):
        repo = tmp_path / "realrepo"
        repo.mkdir()
        db = self._make_state_db(tmp_path, [("s1", str(repo), str(repo))])
        plugin, ctx = make_plugin(
            monkeypatch, tmp_path, GONEBY_CAPTURE_MODE="all", GONEBY_HERMES_STATE_DB=db
        )
        self._capture_hook(ctx)
        wal = plugin._get_wal("realrepo")
        entries = wal.read_all()
        assert entries and entries[0]["content"] == "policy probe"
        assert not plugin._get_wal().read_all()  # default project untouched

    def test_opt_in_marker_enables_named_project(self, monkeypatch, tmp_path):
        repo = tmp_path / "optrepo"
        repo.mkdir()
        (repo / ".goneby.json").write_text('{"tool": "goneby", "enabled": true}')
        db = self._make_state_db(tmp_path, [("s1", str(repo), str(repo))])
        plugin, ctx = make_plugin(
            monkeypatch, tmp_path, GONEBY_CAPTURE_MODE="opt-in", GONEBY_HERMES_STATE_DB=db
        )
        self._capture_hook(ctx)
        assert plugin._get_wal("optrepo").read_all()

    def test_marker_disable_beats_mode_all(self, monkeypatch, tmp_path):
        repo = tmp_path / "quietrepo"
        repo.mkdir()
        (repo / ".goneby.json").write_text('{"enabled": false}')
        db = self._make_state_db(tmp_path, [("s1", str(repo), str(repo))])
        plugin, ctx = make_plugin(
            monkeypatch, tmp_path, GONEBY_CAPTURE_MODE="all", GONEBY_HERMES_STATE_DB=db
        )
        self._capture_hook(ctx)
        assert not plugin._get_wal("quietrepo").read_all()

    def test_platform_kill_gates_by_platform(self, monkeypatch, tmp_path):
        plugin, ctx = make_plugin(monkeypatch, tmp_path)
        run = ctx.commands["goneby"]
        reply = run("platform telegram off", session_id="s1", platform="cli")
        assert "DISABLED" in reply

        # telegram session captures nothing
        self._capture_hook(ctx, platform="telegram")
        assert not plugin._get_wal().read_all()
        # cli session still captures
        self._capture_hook(ctx, platform="cli")
        assert plugin._get_wal().read_all()

    def test_gate_off_stops_everything(self, monkeypatch, tmp_path):
        plugin, ctx = make_plugin(monkeypatch, tmp_path)
        reply = ctx.commands["goneby"]("gate off", session_id="s1")
        assert "OFF" in reply
        self._capture_hook(ctx)
        assert not plugin._get_wal().read_all()

    def test_mode_switch_via_command(self, monkeypatch, tmp_path):
        plugin, ctx = make_plugin(monkeypatch, tmp_path)
        run = ctx.commands["goneby"]
        assert "opt-in" in run("mode opt-in", session_id="s1")
        self._capture_hook(ctx)
        assert not plugin._get_wal().read_all()
        assert "all" in run("mode all", session_id="s1")
        self._capture_hook(ctx, platform="desktop")
        assert plugin._get_wal().read_all()

    def test_status_reports_policy(self, monkeypatch, tmp_path):
        _plugin, ctx = make_plugin(monkeypatch, tmp_path, GONEBY_CAPTURE_MODE="opt-in")
        out = json.loads(ctx.tools["goneby_status"]["handler"]({}, session_id="s1"))
        assert out["capture"]["policy"]["mode"] == "opt-in"
        assert out["capture"]["allowed_for_this_session"] is False

    def test_unknown_command_shows_usage(self, monkeypatch, tmp_path):
        _plugin, ctx = make_plugin(monkeypatch, tmp_path)
        assert "usage" in ctx.commands["goneby"]("frobnicate", session_id="s1")
