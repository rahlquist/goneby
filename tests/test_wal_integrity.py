"""WAL tamper detection (ctx-audit row: 'Damaged entries are noticed').

Per-line AEAD tags proved a tampered CONTENT, but GoneBy bound no additional
data — so deleting, reordering, or splicing lines from another project's log
passed unnoticed. v2 lines (``v2:<seq>:<base64>``) bind file name, parent-dir
token, and sequence into the AAD; legacy (bare base64) lines stay readable but
are flagged ``integrity: unbound`` (D4: accept-and-flag, never discard).

"""

from __future__ import annotations

import base64
import json

from goneby.crypto.cipher import Cipher
from goneby.wal.manager import WALManager
from goneby.wal.writer import WALWriter


def make_wal(tmp_path, name="test.wal", key=None):
    key = key if key is not None else Cipher.generate_key()
    return WALManager(tmp_path / name, Cipher(key)), key


def raw_lines(wal_file):
    return [ln for ln in wal_file.read_text().splitlines() if ln.strip()]


class TestV2Framing:
    def test_new_lines_are_v2_and_sequential(self, tmp_path):
        wal, _ = make_wal(tmp_path)
        wal.append("one", {})
        wal.append("two", {})
        lines = raw_lines(tmp_path / "test.wal")
        assert lines[0].startswith("v2:1:")
        assert lines[1].startswith("v2:2:")

    def test_reopened_writer_continues_the_sequence(self, tmp_path):
        _wal, key = make_wal(tmp_path)
        wal2, _ = make_wal(tmp_path, key=key)
        wal2.append("after restart", {})
        assert raw_lines(tmp_path / "test.wal")[-1].startswith("v2:1:")
        wal3, _ = make_wal(tmp_path, key=key)
        wal3.append("again", {})
        assert raw_lines(tmp_path / "test.wal")[-1].startswith("v2:2:")

    def test_roundtrip_is_transparent(self, tmp_path):
        wal, _ = make_wal(tmp_path)
        for i in range(5):
            wal.append(f"entry {i}", {"type": "note"})
        entries = wal.read_all()
        assert [e["content"] for e in entries] == [f"entry {i}" for i in range(5)]
        assert all(e["metadata"]["wal_seq"] == i + 1 for i, e in enumerate(entries))


class TestTamperDetection:
    def _three(self, tmp_path):
        wal, key = make_wal(tmp_path)
        for c in ("first", "second", "third"):
            wal.append(c, {})
        return key

    def test_deleted_middle_line_reports_gap(self, tmp_path):
        key = self._three(tmp_path)
        f = tmp_path / "test.wal"
        lines = raw_lines(f)
        f.write_text("\n".join([lines[0], lines[2]]) + "\n")
        wal, _ = make_wal(tmp_path, key=key)
        entries = wal.read_all()
        contents = [e["content"] for e in entries]
        assert "second" not in contents
        assert any("[GAP]" in c for c in contents), "a deleted line must be reported"
        clean, notes = wal.read_for_walk()
        assert [e["content"] for e in clean] == ["first", "third"]
        assert any("GAP" in n or "gap" in n for n in notes)

    def test_reordered_lines_are_excluded_and_noted(self, tmp_path):
        key = self._three(tmp_path)
        f = tmp_path / "test.wal"
        lines = raw_lines(f)
        f.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")
        wal, _ = make_wal(tmp_path, key=key)
        clean, notes = wal.read_for_walk()
        assert [e["content"] for e in clean] == ["second", "third"]
        assert any("first" in e["content"] for e in wal.read_all())  # readable...
        assert any("out of order" in n for n in notes), \
            "...but flagged out of order and kept out of the walk"

    def test_foreign_project_line_spliced_is_detected(self, tmp_path):
        key = self._three(tmp_path)
        other, _ = make_wal(tmp_path, name="other-project.wal", key=key)
        other.append("foreign secret", {})
        foreign = raw_lines(tmp_path / "other-project.wal")[0]
        # splice into our WAL at the position the sequence would allow:
        our_file = tmp_path / "test.wal"
        lines = raw_lines(our_file)
        our_file.write_text("\n".join(lines[:1] + [foreign] + lines[1:]) + "\n")
        wal, _ = make_wal(tmp_path, key=key)
        clean, notes = wal.read_for_walk()
        assert "foreign secret" not in [e["content"] for e in clean]
        # the three genuine entries survive — detection is surgical, not total
        assert [e["content"] for e in clean] == ["first", "second", "third"]
        assert any("CORRUPTED" in n or "corrupt" in n.lower() for n in notes)

    def test_same_name_different_directory_splice_detected(self, tmp_path):
        inner_a = tmp_path / "a"
        inner_b = tmp_path / "b"
        inner_a.mkdir()
        inner_b.mkdir()
        wal_a, key_a = make_wal(inner_a)
        for c in ("one", "two"):
            wal_a.append(c, {})
        wal_b, _ = make_wal(inner_b, key=key_a)  # same key on purpose
        wal_b.append("planted", {})
        planted = raw_lines(inner_b / "test.wal")[0]  # v2:1:<blob>
        lines = raw_lines(inner_a / "test.wal")
        (inner_a / "test.wal").write_text("\n".join(lines + [planted]) + "\n")
        wal_a2, _ = make_wal(inner_a, key=key_a)
        clean, notes = wal_a2.read_for_walk()
        assert "planted" not in [e["content"] for e in clean]
        assert notes

    def test_unbound_legacy_line_stays_readable_and_flagged(self, tmp_path):
        """D4: accept-and-flag. A pre-v2 line must decrypt, be reported
        unbound, and still reach the walk (no silent data loss in migration)."""
        key = Cipher.generate_key()
        cipher = Cipher(key)
        entry = {"timestamp": "2026-11-01T00:00:00+00:00", "content": "old faithful", "metadata": {}}
        blob = cipher.encrypt(json.dumps(entry).encode("utf-8"))  # no AAD, v1 path
        legacy_line = base64.b64encode(blob).decode("ascii")
        (tmp_path / "test.wal").write_text(legacy_line + "\n")
        wal, _ = make_wal(tmp_path, key=key)
        entries = wal.read_all()
        assert entries[0]["content"] == "old faithful"
        assert entries[0]["metadata"]["integrity"] == "unbound"
        clean, notes = wal.read_for_walk()
        assert [e["content"] for e in clean] == ["old faithful"]
        assert any("unbound" in n for n in notes)


class TestKeyId:
    def test_key_id_header_written_on_first_append(self, tmp_path):
        key = Cipher.generate_key()
        wal, _ = make_wal(tmp_path, key=key)
        wal.append("x", {})
        header = tmp_path / "test.keyid"
        assert header.exists(), "WAL must carry a key id for future rotation"
        assert header.read_text().strip() == Cipher(key).key_id

    def test_wrong_key_corruption_says_so(self, tmp_path):
        _wal, key_a = make_wal(tmp_path)
        WALWriter(tmp_path / "test.wal", Cipher(key_a)).append("secret", {})
        (tmp_path / "test.keyid").write_text(Cipher(key_a).key_id)
        wal_b, _ = make_wal(tmp_path, key=Cipher.generate_key())
        entries = wal_b.read_all()
        assert any("wrong key" in e["content"] for e in entries), \
            "key-id mismatch must be reported as wrong key, not generic corruption"
