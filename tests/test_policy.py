"""Policy unit tests: the decision table and the persistence layer."""

from __future__ import annotations

import pytest

from goneby.config import policy


def _allowed(**kw):
    base = {"mode": policy.MODE_ALL, "disabled_platforms": frozenset()}
    base.update(kw)
    return policy.capture_allowed(base.pop("project", "p"), base.pop("platform", ""), **base)


class TestDecisionTable:
    def test_mode_all_captures(self):
        assert _allowed() is True

    def test_mode_opt_in_blocks(self):
        assert _allowed(mode=policy.MODE_OPT_IN) is False

    def test_master_off_beats_everything(self):
        assert _allowed(master=False, mode=policy.MODE_ALL) is False
        assert _allowed(master=False, mode=policy.MODE_OPT_IN, marker=True) is False

    def test_platform_kill_beats_mode_and_marker(self):
        disabled = frozenset({"telegram"})
        assert _allowed(platform="telegram", disabled_platforms=disabled) is False
        assert _allowed(platform="telegram", disabled_platforms=disabled, marker=True) is False

    def test_marker_beats_override_beats_mode(self):
        assert _allowed(marker=True, override=False, mode=policy.MODE_OPT_IN) is True
        assert _allowed(marker=False, override=True, mode=policy.MODE_ALL) is False
        assert _allowed(marker=None, override=True, mode=policy.MODE_OPT_IN) is True

    def test_platform_case_insensitive(self):
        assert _allowed(platform="Telegram", disabled_platforms=frozenset({"telegram"})) is False


class TestPersistence:
    def test_config_roundtrip_atomic(self, tmp_path):
        p = tmp_path / "cfg.json"
        cfg = policy.set_mode("opt-in", p)
        assert cfg["mode"] == "opt-in"
        assert policy.load_config(p) == cfg
        assert not list(tmp_path.glob("*.tmp"))

    def test_invalid_json_reads_empty(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text("{not json")
        assert policy.load_config(p) == {}

    def test_mode_validation(self, tmp_path):
        with pytest.raises(ValueError):
            policy.set_mode("sometimes", tmp_path / "c.json")

    def test_platform_add_and_remove(self, tmp_path):
        p = tmp_path / "cfg.json"
        cfg = policy.set_platform("Telegram", False, p)
        assert cfg["disabled_platforms"] == ["telegram"]
        cfg = policy.set_platform("telegram", True, p)
        assert cfg["disabled_platforms"] == []

    def test_env_and_file_platforms_union(self):
        cfg = {"disabled_platforms": ["discord"]}
        out = policy.effective_disabled_platforms(cfg, env_value="telegram, CLI ")
        assert out == frozenset({"discord", "telegram", "cli"})

    def test_file_mode_wins_over_env(self):
        assert policy.effective_mode({"mode": "opt-in"}, env_mode="all") == "opt-in"
        assert policy.effective_mode({}, env_mode="opt-in") == "opt-in"
        assert policy.effective_mode({}, env_mode="bogus") == "all"


class TestMarker:
    def test_absent_is_none(self, tmp_path):
        assert policy.read_marker(tmp_path) is None
        assert policy.read_marker(None) is None

    def test_write_read_roundtrip(self, tmp_path):
        path = policy.write_marker(tmp_path, True)
        assert policy.read_marker(tmp_path) is True
        policy.write_marker(tmp_path, False)
        assert policy.read_marker(tmp_path) is False
        assert path.name == ".goneby.json"

    def test_bogus_marker_is_none(self, tmp_path):
        (tmp_path / ".goneby.json").write_text('{"enabled": "yes please"}')
        assert policy.read_marker(tmp_path) is None

    def test_write_into_missing_root_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            policy.write_marker(tmp_path / "nope", True)


class TestUserOverrides:
    def test_override_roundtrip(self, tmp_path):
        p = tmp_path / "cfg.json"
        cfg = policy.set_project_override({}, "proj-a", True, p)
        assert policy.project_override(cfg, "proj-a") is True
        assert policy.project_override(cfg, "proj-b") is None

    def test_non_dict_projects_treated_absent(self):
        assert policy.project_override({"projects": ["nope"]}, "x") is None


class TestSlugResolution:
    def test_repo_root_beats_cwd(self, tmp_path):
        slug, root = policy.resolve_project_slug(str(tmp_path / "sub" / "dir"), str(tmp_path))
        assert slug == tmp_path.name
        assert root == tmp_path

    def test_cwd_fallback(self, tmp_path):
        slug, root = policy.resolve_project_slug(str(tmp_path), "")
        assert slug == tmp_path.name
        assert root == tmp_path

    def test_absolute_path_cwd_flattens(self):
        slug, root = policy.resolve_project_slug("/home/user/code/repoX", "")
        assert slug == "repoX"
        assert root.name == "repoX"

    def test_empty_everything(self):
        slug, root = policy.resolve_project_slug("", "")
        assert slug == "default"
        assert root is None
