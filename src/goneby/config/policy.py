"""Capture policy — who/where/when GoneBy records turns.

Four knobs, evaluated in a fixed order (see `capture_allowed`):

1. **master switch** — `goneby gate off` / `/goneby gate off` disables capture
   everywhere, everywhere, without touching env. (Legacy
   ``GONEBY_AUTO_CAPTURE=false`` behaves identically.)
2. **per-gateway kill** — platform ids (``telegram``, ``discord``, ``cli``,
   ``desktop``...) blocked outright: a gateway serving those platforms never
   captures, regardless of project state.
3. **per-project marker** — ``.goneby.json`` in the project root. Team
   opt-in/opt-out that travels with the repo.
4. **user-level override** — ``~/.goneby/config.json`` ``projects`` map:
   personal, works on repos you don't own and can't write into.
5. **mode** — ``all`` (default: capture everything not excluded above) or
   ``opt-in`` (capture only projects explicitly enabled).

Everything persists in ``~/.goneby/config.json`` (override with
``GONEBY_CONFIG_PATH``); the env vars remain process-scoped fallbacks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MODE_ALL = "all"
MODE_OPT_IN = "opt-in"
MODES = (MODE_ALL, MODE_OPT_IN)

MARKER_FILENAME = ".goneby.json"


def default_config_path() -> Path:
    override = os.environ.get("GONEBY_CONFIG_PATH", "")
    if override:
        return Path(override).expanduser()
    return Path("~/.goneby/config.json").expanduser()


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read the user config; missing or corrupt reads as {} (never raises)."""
    p = path or default_config_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_config(cfg: dict[str, Any], path: Path | None = None) -> Path:
    """Atomic write of the user config; returns the path written."""
    p = path or default_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def effective_mode(cfg: dict[str, Any], env_mode: str = "") -> str:
    """config.json wins over env; invalid values fall back to the default."""
    for candidate in (cfg.get("mode"), env_mode):
        if isinstance(candidate, str) and candidate in MODES:
            return candidate
    return MODE_ALL


def effective_disabled_platforms(cfg: dict[str, Any], env_value: str = "") -> frozenset[str]:
    """Persistent list ∪ env list (a gateway can kill itself without a file edit)."""
    out: set[str] = set()
    listed = cfg.get("disabled_platforms")
    if isinstance(listed, list):
        out.update(str(x).strip().lower() for x in listed if str(x).strip())
    if env_value:
        out.update(x.strip().lower() for x in env_value.split(",") if x.strip())
    return frozenset(out)


def master_enabled(cfg: dict[str, Any], env_flag: bool = True) -> bool:
    """Persistent master switch AND the legacy env kill switch."""
    flag = cfg.get("master_enabled", True)
    return bool(flag) and bool(env_flag)


def set_master_enabled(enabled: bool, path: Path | None = None) -> dict[str, Any]:
    cfg = load_config(path)
    cfg["master_enabled"] = bool(enabled)
    save_config(cfg, path)
    return cfg


def set_mode(mode: str, path: Path | None = None) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    cfg = load_config(path)
    cfg["mode"] = mode
    save_config(cfg, path)
    return cfg


def set_platform(platform: str, enabled: bool, path: Path | None = None) -> dict[str, Any]:
    """enabled=True REMOVES the platform from the kill list (capture allowed there)."""
    name = platform.strip().lower()
    if not name:
        raise ValueError("platform name must not be empty")
    cfg = load_config(path)
    listed = cfg.get("disabled_platforms")
    current = {str(x).strip().lower() for x in listed} if isinstance(listed, list) else set()
    if enabled:
        current.discard(name)
    else:
        current.add(name)
    cfg["disabled_platforms"] = sorted(current)
    save_config(cfg, path)
    return cfg


# ---------------------------------------------------------------------------
# Per-project state: repo marker file, then user-level override
# ---------------------------------------------------------------------------


def marker_path(project_root: Path) -> Path:
    return Path(project_root) / MARKER_FILENAME


def read_marker(project_root: Path | None) -> bool | None:
    """True/False from the marker's ``enabled`` field; None when absent/unusable."""
    if project_root is None:
        return None
    p = marker_path(Path(project_root))
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    flag = data.get("enabled") if isinstance(data, dict) else None
    return bool(flag) if isinstance(flag, bool) else None


def write_marker(project_root: Path, enabled: bool) -> Path:
    root = Path(project_root)
    if not root.is_dir():
        raise NotADirectoryError(f"project root does not exist: {root}")
    p = marker_path(root)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"tool": "goneby", "enabled": bool(enabled)}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, p)
    return p


def project_override(cfg: dict[str, Any], project: str) -> bool | None:
    """User-level per-project tri-state; None when unset."""
    entry = cfg.get("projects")
    if not isinstance(entry, dict):
        return None
    val = entry.get(project)
    flag = val.get("enabled") if isinstance(val, dict) else val
    return bool(flag) if isinstance(flag, bool) else None


def set_project_override(cfg: dict[str, Any], project: str, enabled: bool,
                         path: Path | None = None) -> dict[str, Any]:
    projects = cfg.setdefault("projects", {})
    if not isinstance(projects, dict):
        projects = cfg["projects"] = {}
    projects[project] = {"enabled": bool(enabled)}
    save_config(cfg, path)
    return cfg


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def capture_allowed(project: str, platform: str, *, mode: str,
                    disabled_platforms: frozenset[str], master: bool = True,
                    marker: bool | None = None, override: bool | None = None) -> bool:
    """The whole policy as one deterministic decision, most-specific wins.

    Order: master switch → platform kill → repo marker → user override → mode.
    """
    if not master:
        return False
    if platform and platform.strip().lower() in disabled_platforms:
        return False
    if marker is not None:
        return marker
    if override is not None:
        return override
    return mode == MODE_ALL


def resolve_project_slug(cwd: str, git_repo_root: str) -> tuple[str, Path | None]:
    """(slug, root) from session context. Repo root beats cwd; either falls
    back to the basename used as the project name. (0, None) when unknown."""
    from goneby.config.settings import project_slug

    root = (git_repo_root or cwd or "").strip()
    if not root:
        return "default", None
    name = Path(root).resolve().name or "default"
    return project_slug(name), Path(root)
