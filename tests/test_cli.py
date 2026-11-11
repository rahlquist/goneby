"""CLI tests — these close the gap that hid a real bug: every CLI WAL command
used `config.wal_path` (the directory) directly instead of the per-project
file, so `goneby status`/`append`/`read` crashed with IsADirectoryError or
corrupted project isolation the moment the base dir existed.
"""

from __future__ import annotations

import pytest

from goneby.cli.main import main
from goneby.config.settings import project_wal_path

TEST_KEY = "00" * 32


@pytest.fixture()
def cli_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GONEBY_WAL_PATH", str(tmp_path))
    monkeypatch.setenv("GONEBY_ENCRYPTION_KEY", TEST_KEY)
    monkeypatch.setenv("GONEBY_PROJECT", "cli-test")
    # Unreachable-but-fast endpoint: the status probe must degrade, not raise.
    monkeypatch.setenv("GONEBY_EMBEDDING_BASE_URL", "http://127.0.0.1:1/v1")
    return tmp_path


def run_cli(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["goneby", *argv])
    return main()


def test_status_survives_existing_wal_dir(cli_env, monkeypatch, capsys):
    # The crash condition: base dir exists but the project file does not.
    (cli_env / "stray.txt").write_text("x")
    rc = run_cli(monkeypatch, ["status"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert "WAL entries: 0" in captured.out
    assert "cli-test.wal" in captured.out


def test_append_and_read_use_project_file(cli_env, monkeypatch, capsys):
    rc = run_cli(monkeypatch, ["append", "hello cli", "-t", "note"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cli-test.wal" in out

    wal_file = project_wal_path(cli_env, "cli-test")
    assert wal_file.exists()
    # encrypted at rest
    assert b"hello cli" not in wal_file.read_bytes()

    rc = run_cli(monkeypatch, ["read"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "hello cli" in out
    assert "Entries: 1" in out


def test_config_errors_reported_not_raised(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GONEBY_WAL_PATH", str(tmp_path))
    monkeypatch.delenv("GONEBY_ENCRYPTION_KEY", raising=False)
    rc = run_cli(monkeypatch, ["append", "nope"])
    assert rc == 1
    assert "Configuration errors" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Policy commands: project / mode / platform / gate
# ---------------------------------------------------------------------------


@pytest.fixture()
def policy_env(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.json"
    monkeypatch.setenv("GONEBY_CONFIG_PATH", str(cfg_path))
    monkeypatch.setenv("GONEBY_PROJECT", "pol-proj")
    return cfg_path


def test_mode_command_persists(policy_env, monkeypatch, capsys):
    assert run_cli(monkeypatch, ["mode", "opt-in"]) == 0
    assert "opt-in" in capsys.readouterr().out
    from goneby.config import policy

    assert policy.load_config(policy_env)["mode"] == "opt-in"


def test_project_enable_writes_marker(policy_env, monkeypatch, capsys):
    repo = policy_env.parent / "somerepo"
    repo.mkdir()
    assert run_cli(monkeypatch, ["project", "enable", str(repo)]) == 0
    assert (repo / ".goneby.json").exists()

    assert run_cli(monkeypatch, ["project", "status", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "Marker:    True" in out
    assert "Capturing: True" in out


def test_project_enable_without_path_uses_override(policy_env, monkeypatch):
    assert run_cli(monkeypatch, ["project", "enable"]) == 0
    from goneby.config import policy

    assert policy.project_override(policy.load_config(policy_env), "pol-proj") is True


def test_platform_and_gate_commands(policy_env, monkeypatch, capsys):
    assert run_cli(monkeypatch, ["platform", "disable", "telegram"]) == 0
    assert run_cli(monkeypatch, ["platform", "list"]) == 0
    assert "telegram" in capsys.readouterr().out

    assert run_cli(monkeypatch, ["gate", "off"]) == 0
    from goneby.config import policy

    assert policy.load_config(policy_env)["master_enabled"] is False

    assert run_cli(monkeypatch, ["platform", "enable"]) == 1  # missing id
