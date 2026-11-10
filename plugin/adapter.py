"""GoneBy plugin adapter.

Bridges Hermes' plugin surface (tools + hooks + host LLM) onto the goneby
library. The library owns crypto, WAL, embeddings, and the vector store; this
module only translates calls and lifecycle events.

Design notes
------------
**Capture is per-turn, not per-compaction.** ``post_compaction`` fires *after*
the transcript is rewritten, so the messages it dropped are already gone. Instead
``pre_llm_call`` hands us the full ``conversation_history`` each turn, and we
append the messages we have not seen before. That is a superset of per-compaction
capture and loses nothing to the rewrite.

**Recall is opt-in.** ``auto_recall`` defaults off: a semantic search on every
turn costs latency, and the agent already has ``query_memory``. Flip it on when
you want memories surfaced without being asked.

**Hooks must stay fast and must never raise.** Hermes logs and skips a failing
callback, and observer callbacks are bounded by ``plugins.hook_callback_timeout``.
Capture is synchronous file I/O; anything touching the embedder runs on a
dedicated background loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

ENTRY_TYPES = ("decision", "finding", "note", "constraint")


def message_text(message: dict[str, Any]) -> str:
    """Flatten an OpenAI-format message to plain text.

    Content may be a string or a list of parts (text/image/tool_use). Only text
    parts are kept — image blocks carry no memory value and would bloat the WAL.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "input_text", "output_text"):
                    chunks.append(str(part.get("text", "")))
                elif part.get("type") == "tool_result":
                    inner = part.get("content")
                    if isinstance(inner, str):
                        chunks.append(inner)
                    elif isinstance(inner, list):
                        for sub in inner:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                chunks.append(str(sub.get("text", "")))
            elif isinstance(part, str):
                chunks.append(part)
        return "\n".join(c for c in chunks if c)
    return ""


def message_identity(message: dict[str, Any]) -> str:
    """Stable identity for a message, for capture de-duplication.

    Prefers the host's ``message_uid`` (survives compaction and rotation). Falls
    back to a hash of role + text, which is stable across turns for unchanged
    messages and changes when the host rewrites one.
    """
    uid = message.get("message_uid")
    if uid:
        return f"uid:{uid}"
    role = message.get("role", "")
    text = message_text(message)
    return f"hash:{role}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


class GoneByPlugin:
    """Hermes plugin surface for GoneBy."""

    def __init__(self, ctx):
        self._ctx = ctx
        self._lock = threading.RLock()
        self._seen: set[str] = set()
        self._loop: Any = None
        self._loop_thread: threading.Thread | None = None

        from goneby.config.settings import GoneByConfig, project_wal_path

        self._config = GoneByConfig.from_env()
        self._wal_path = project_wal_path(self._config.wal_path, self._config.project)

        # Opt-in recall: a search on every turn costs latency.
        self.auto_recall = self._env_flag("GONEBY_AUTO_RECALL", default=False)
        self.auto_capture = self._env_flag("GONEBY_AUTO_CAPTURE", default=True)
        # Skip recall for very short turns ("ok", "continue") — rarely a query.
        self.recall_min_chars = int(
            self._env("GONEBY_RECALL_MIN_CHARS", "12")
        )
        self.recall_top_k = int(self._env("GONEBY_RECALL_TOP_K", "3"))

        self._cipher = None
        self._store = None
        self._init_error: str | None = None
        try:
            if not self._config.encryption_key:
                raise ValueError("GONEBY_ENCRYPTION_KEY is not set")
            from goneby.crypto.cipher import Cipher

            self._cipher = Cipher(self._config.encryption_key)
        except Exception as exc:  # noqa: BLE001 - reported by tools, not raised
            self._init_error = f"{type(exc).__name__}: {exc}"
            logger.warning("goneby: unavailable — %s", self._init_error)

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _env(name: str, default: str = "") -> str:
        import os

        return os.environ.get(name, default)

    @classmethod
    def _env_flag(cls, name: str, *, default: bool) -> bool:
        raw = cls._env(name).strip().lower()
        if not raw:
            return default
        return raw in ("1", "true", "yes", "on")

    @property
    def project(self) -> str:
        return self._config.project

    @property
    def available(self) -> bool:
        return self._cipher is not None

    # ------------------------------------------------------------------
    # Background event loop (for the async embedder/vector store)
    # ------------------------------------------------------------------

    def _get_loop(self):
        """A dedicated loop on a daemon thread.

        Hermes hooks are synchronous; the embedder and vector backends are async.
        One long-lived loop avoids spinning up an event loop per call.
        """
        if self._loop is not None:
            return self._loop
        with self._lock:
            if self._loop is not None:
                return self._loop
            import asyncio

            loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=loop.run_forever, name="goneby-plugin-loop", daemon=True
            )
            self._loop_thread.start()
            self._loop = loop
            return loop

    def _run_async(self, coro, timeout: float = 60.0):
        """Run a coroutine on the background loop and block for its result."""
        import asyncio

        loop = self._get_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    def shutdown(self) -> None:
        loop, thread = self._loop, self._loop_thread
        self._loop = None
        self._loop_thread = None
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                logger.debug("goneby: loop stop failed — %s", exc)
        if thread is not None:
            thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Lazy components
    # ------------------------------------------------------------------

    def _get_store(self):
        if self._store is not None:
            return self._store
        from goneby.memory.embedder import Embedder, EmbeddingConfig
        from goneby.memory.store import MemoryStore
        from goneby.vector.factory import create_backend

        embedder = Embedder(
            EmbeddingConfig(
                base_url=self._config.embedding_base_url,
                model=self._config.embedding_model,
                api_key=self._config.embedding_api_key,
                dimensions=self._config.embedding_dimensions,
                batch_size=self._config.embedding_batch_size,
            )
        )
        self._store = MemoryStore(
            embedder, create_backend(self._config), project=self._config.project
        )
        return self._store

    def _get_wal(self):
        from goneby.wal.manager import WALManager

        return WALManager(self._wal_path, self._cipher)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self) -> None:
        ctx = self._ctx
        ctx.register_tool(
            name="wall_append",
            toolset="goneby",
            schema=self._schema(
                "wall_append",
                "Append an entry to this project's encrypted Write-Ahead Log. "
                "Call at context compaction boundaries to capture project output. "
                "No filtering happens here — everything is stored; the backward "
                "walk decides relevance later.",
                {
                    "content": {
                        "type": "string",
                        "description": "The text content to store.",
                    },
                    "entry_type": {
                        "type": "string",
                        "enum": list(ENTRY_TYPES),
                        "description": "Kind of entry.",
                    },
                },
                required=["content"],
            ),
            handler=self._tool_wall_append,
            is_async=False,
        )
        ctx.register_tool(
            name="wall_read",
            toolset="goneby",
            schema=self._schema(
                "wall_read",
                "Read all entries from this project's WAL.",
                {},
                required=[],
            ),
            handler=self._tool_wall_read,
            is_async=False,
        )
        ctx.register_tool(
            name="walk_backward",
            toolset="goneby",
            schema=self._schema(
                "walk_backward",
                "Filter the WAL by walking backward from the shipped product, "
                "keeping only entries that contributed to it. Run at project close, "
                "then pass kept_entries to store_memory.",
                {
                    "mvp_text": {
                        "type": "string",
                        "description": "The shipped product or final artifact — the ground truth.",
                    }
                },
                required=["mvp_text"],
            ),
            handler=self._tool_walk_backward,
            is_async=False,
        )
        ctx.register_tool(
            name="store_memory",
            toolset="goneby",
            schema=self._schema(
                "store_memory",
                "Embed and store walked entries in the vector backend. Idempotent.",
                {
                    "entries": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "kept_entries from walk_backward.",
                    }
                },
                required=["entries"],
            ),
            handler=self._tool_store_memory,
            is_async=False,
        )
        ctx.register_tool(
            name="query_memory",
            toolset="goneby",
            schema=self._schema(
                "query_memory",
                "Semantic search over stored memories. Defaults to this project; "
                "pass scope='all' to search every project.",
                {
                    "query": {"type": "string", "description": "Natural-language question."},
                    "top_k": {"type": "integer", "description": "Max results."},
                    "min_score": {
                        "type": "number",
                        "description": "Minimum similarity (0.0-1.0).",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["project", "all"],
                        "description": "'project' (default) or 'all'.",
                    },
                },
                required=["query"],
            ),
            handler=self._tool_query_memory,
            is_async=False,
        )
        ctx.register_tool(
            name="goneby_status",
            toolset="goneby",
            schema=self._schema(
                "goneby_status",
                "Show GoneBy configuration, WAL statistics, and backend health.",
                {},
                required=[],
            ),
            handler=self._tool_status,
            is_async=False,
        )

        ctx.register_hook("pre_llm_call", self._on_pre_llm_call)
        ctx.register_hook("post_compaction", self._on_post_compaction)
        logger.info(
            "goneby: hooks registered (auto_capture=%s auto_recall=%s)",
            self.auto_capture,
            self.auto_recall,
        )

    @staticmethod
    def _schema(name: str, description: str, properties: dict, required: list) -> dict:
        return {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture(self, conversation_history: list) -> int:
        """Append unseen messages to the WAL. Returns count appended.

        Pure synchronous file I/O — safe to call from a hook.
        """
        if not self.available or not conversation_history:
            return 0

        appended = 0
        try:
            wal = self._get_wal()
            with self._lock:
                for message in conversation_history:
                    if not isinstance(message, dict):
                        continue
                    role = message.get("role")
                    # System prompts and tool scaffolding are not project memory.
                    if role in ("system", "tool"):
                        continue
                    text = message_text(message).strip()
                    if not text:
                        continue
                    identity = message_identity(message)
                    if identity in self._seen:
                        continue
                    self._seen.add(identity)
                    wal.append(text, {"type": "note", "role": role})
                    appended += 1
        except Exception as exc:  # noqa: BLE001 - capture must never break a turn
            logger.warning("goneby: capture failed — %s", exc)
            return 0
        return appended

    def _on_pre_llm_call(
        self,
        session_id: str,
        user_message,
        conversation_history: list,
        is_first_turn: bool,
        model: str,
        platform: str,
        **kwargs,
    ):
        """Capture new turns, and optionally inject recall.

        Never raises: a failing hook is logged and skipped by the host, but a
        raise here would also skip any later plugin's hook.
        """
        try:
            if self.auto_capture:
                self._capture(conversation_history or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("goneby: pre_llm_call capture error — %s", exc)

        if not self.auto_recall:
            return None

        query = user_message if isinstance(user_message, str) else message_text(
            {"content": user_message}
        )
        if not query or len(query.strip()) < self.recall_min_chars:
            return None

        try:
            return self._recall(query.strip())
        except Exception as exc:  # noqa: BLE001 - recall must never break a turn
            logger.warning("goneby: recall failed — %s", exc)
            return None

    def _recall(self, query: str) -> str | None:
        store = self._get_store()
        result = self._run_async(
            store.search(query, top_k=self.recall_top_k, scope="project")
        )
        if result.get("status") != "ok" or not result.get("results"):
            return None

        lines = []
        for hit in result["results"]:
            text = (hit.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"- {text}")
        if not lines:
            return None
        body = "\n".join(lines)
        return (
            f"[goneby recall — {result['count']} relevant memor"
            f"{'y' if result['count'] == 1 else 'ies'} from this project]\n{body}"
        )

    def _on_post_compaction(
        self,
        session_id: str,
        kind: str,
        reason: str,
        **kwargs,
    ):
        """Record the compaction boundary.

        Content capture already happened in ``pre_llm_call`` before the rewrite;
        this notes the boundary so the WAL reflects the project's shape.
        """
        try:
            if not self.available:
                return
            # A compaction boundary is worth one line of context for the walk.
            self._capture(
                [
                    {
                        "role": "user",
                        "content": (
                            f"[compaction boundary] kind={kind} reason={reason} "
                            f"messages_removed={kwargs.get('messages_removed')}"
                        ),
                        "message_uid": f"compaction-{session_id}-{kind}-{reason}",
                    }
                ]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("goneby: post_compaction error — %s", exc)

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _require_ready(self) -> dict | None:
        if not self.available:
            return {
                "status": "error",
                "error": self._init_error or "goneby is not available",
            }
        return None

    def _tool_wall_append(self, params: dict, **kwargs) -> str:
        not_ready = self._require_ready()
        if not_ready:
            return json.dumps(not_ready)

        content = (params.get("content") or "").strip()
        if not content:
            return json.dumps({"status": "error", "error": "content must not be empty"})

        entry_type = params.get("entry_type") or "note"
        if entry_type not in ENTRY_TYPES:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"entry_type must be one of {', '.join(ENTRY_TYPES)}",
                }
            )

        try:
            wal = self._get_wal()
            with self._lock:
                wal.append(content, {"type": entry_type})
            return json.dumps(
                {
                    "status": "ok",
                    "project": self.project,
                    "entry_type": entry_type,
                    "wal_entry_count": wal.entry_count,
                }
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"})

    def _tool_wall_read(self, params: dict, **kwargs) -> str:
        not_ready = self._require_ready()
        if not_ready:
            return json.dumps(not_ready)
        try:
            entries = self._get_wal().read_all()
            return json.dumps(
                {"status": "ok", "project": self.project, "count": len(entries), "entries": entries}
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"})

    def _tool_walk_backward(self, params: dict, **kwargs) -> str:
        not_ready = self._require_ready()
        if not_ready:
            return json.dumps(not_ready)

        mvp_text = (params.get("mvp_text") or "").strip()
        if not mvp_text:
            return json.dumps({"status": "error", "error": "mvp_text must not be empty"})

        try:
            entries = self._get_wal().read_all()
            if not entries:
                return json.dumps(
                    {"status": "empty", "kept_count": 0, "kept_entries": []}
                )

            flags = self._judge(mvp_text, entries)
            from goneby.walk.backward import BackwardWalker

            async def judge(_mvp, batch):
                return flags

            result = self._run_async(BackwardWalker(judge).walk(mvp_text, entries))

            return json.dumps(
                {
                    "status": "ok",
                    "project": self.project,
                    "kept_count": result.kept_count,
                    "discarded_count": result.discarded_count,
                    "total_count": result.total_count,
                    "dedup_count": result.dedup_count,
                    "kept_entries": result.kept_entries,
                }
            )
        except Exception as exc:
            logger.exception("goneby: walk_backward failed")
            return json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"})

    def _judge(self, mvp_text: str, entries: list) -> list[bool]:
        """Ask the host's model which entries contributed to the shipped product.

        Uses ``ctx.llm.complete_structured`` so the user's own model and auth are
        borrowed — no separate API key, no hardcoded provider.
        """

        entries_text = "\n\n".join(
            f"--- Entry {i} ---\n{e.get('content', '')}" for i, e in enumerate(entries)
        )
        instructions = (
            "You are a relevance judge for a project memory system. Given the "
            "shipped product, decide which project log entries contributed to it. "
            "An entry is relevant if it records a decision, finding, or constraint "
            "that shaped the final product. It is irrelevant if it was a dead end, "
            "a random thought, or was superseded. Respond with one boolean per "
            "entry, in order."
        )
        input_text = f"## Shipped product\n\n{mvp_text}\n\n## Log entries\n\n{entries_text}"

        result = self._ctx.llm.complete_structured(
            instructions=instructions,
            input=[input_text],
            json_schema={"type": "array", "items": {"type": "boolean"}},
            purpose="goneby backward walk",
        )

        parsed = getattr(result, "parsed", None)
        if isinstance(parsed, list):
            flags = [bool(x) for x in parsed]
            if len(flags) < len(entries):
                flags.extend([False] * (len(entries) - len(flags)))
            return flags[: len(entries)]

        # Fall back to parsing the raw text if structured parsing was unavailable.
        raw = getattr(result, "text", None) or ""
        import re

        match = re.search(r"\[[\s,true|false]*\]", raw)
        if match:
            try:
                flags = json.loads(match.group())
                if len(flags) < len(entries):
                    flags.extend([False] * (len(entries) - len(flags)))
                return flags[: len(entries)]
            except json.JSONDecodeError:
                pass

        logger.warning("goneby: judge returned unusable output; keeping all entries")
        return [True] * len(entries)

    def _tool_store_memory(self, params: dict, **kwargs) -> str:
        not_ready = self._require_ready()
        if not_ready:
            return json.dumps(not_ready)

        entries = params.get("entries") or []
        if not isinstance(entries, list):
            return json.dumps({"status": "error", "error": "entries must be a list"})

        try:
            result = self._run_async(self._get_store().index_entries(entries))
            result["project"] = self.project
            return json.dumps(result)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"})

    def _tool_query_memory(self, params: dict, **kwargs) -> str:
        not_ready = self._require_ready()
        if not_ready:
            return json.dumps(not_ready)

        query = (params.get("query") or "").strip()
        if not query:
            return json.dumps({"status": "error", "error": "query must not be empty"})

        scope = params.get("scope") or "project"
        try:
            result = self._run_async(
                self._get_store().search(
                    query,
                    top_k=int(params.get("top_k") or 5),
                    min_score=float(params.get("min_score") or 0.0),
                    scope=scope,
                )
            )
            return json.dumps(result)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"})

    def _tool_status(self, params: dict, **kwargs) -> str:
        payload = {
            "status": "ok" if self.available else "error",
            "project": self.project,
            "wal_path": str(self._wal_path),
            "wal_entry_count": 0,
            "vector_backend": self._config.vector_backend,
            "embedding": {
                "base_url": self._config.embedding_base_url,
                "model": self._config.embedding_model,
                "dimensions": self._config.embedding_dimensions,
            },
            "encryption": "ChaCha20-Poly1305",
            "auto_capture": self.auto_capture,
            "auto_recall": self.auto_recall,
        }
        if self._init_error:
            payload["error"] = self._init_error
            return json.dumps(payload)

        try:
            payload["wal_entry_count"] = self._get_wal().entry_count
        except Exception as exc:  # noqa: BLE001
            payload["wal_error"] = str(exc)

        try:
            health = self._run_async(self._get_store().health(), timeout=15.0)
            payload["health"] = health
        except Exception as exc:  # noqa: BLE001
            payload["health"] = {"ok": False, "detail": str(exc)}

        return json.dumps(payload)
