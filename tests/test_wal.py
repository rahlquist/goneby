"""Tests for GoneBy WAL and encryption layer."""

from __future__ import annotations

import pytest

from goneby.crypto.cipher import Cipher
from goneby.wal.manager import WALManager


class TestCipher:
    """Test ChaCha20-Poly1305 encryption."""

    def test_generate_key(self):
        key = Cipher.generate_key()
        assert len(key) == 32

    def test_encrypt_decrypt(self):
        cipher = Cipher(Cipher.generate_key())
        plaintext = b"Hello, GoneBy!"
        encrypted = cipher.encrypt(plaintext)
        decrypted = cipher.decrypt(encrypted)
        assert decrypted == plaintext

    def test_encrypt_decrypt_unicode(self):
        cipher = Cipher(Cipher.generate_key())
        plaintext = "Unicode: 你好, 世界! 🌍".encode()
        encrypted = cipher.encrypt(plaintext)
        decrypted = cipher.decrypt(encrypted)
        assert decrypted == plaintext

    def test_different_nonces(self):
        cipher = Cipher(Cipher.generate_key())
        plaintext = b"Same plaintext"
        encrypted1 = cipher.encrypt(plaintext)
        encrypted2 = cipher.encrypt(plaintext)
        # Different nonces should produce different ciphertexts
        assert encrypted1 != encrypted2

    def test_invalid_key_size(self):
        with pytest.raises(ValueError):
            Cipher(b"too short")


class TestWALWriter:
    """Test WAL writer."""

    def test_append_and_read(self, tmp_path):
        key = Cipher.generate_key()
        cipher = Cipher(key)
        wal = WALManager(tmp_path / "test.wal", cipher)

        wal.append("First entry", {"type": "decision"})
        wal.append("Second entry", {"type": "finding"})

        entries = wal.read_all()
        assert len(entries) == 2
        assert entries[0]["content"] == "First entry"
        assert entries[1]["content"] == "Second entry"

    def test_encrypted_on_disk(self, tmp_path):
        key = Cipher.generate_key()
        cipher = Cipher(key)
        wal = WALManager(tmp_path / "test.wal", cipher)

        wal.append("Sensitive data: password123", {"type": "note"})

        # Read raw file
        raw = (tmp_path / "test.wal").read_text()
        assert "password123" not in raw
        assert "Sensitive" not in raw

    def test_multiple_appends(self, tmp_path):
        key = Cipher.generate_key()
        cipher = Cipher(key)
        wal = WALManager(tmp_path / "test.wal", cipher)

        for i in range(100):
            wal.append(f"Entry {i}", {"type": "note"})

        entries = wal.read_all()
        assert len(entries) == 100
        assert entries[0]["content"] == "Entry 0"
        assert entries[99]["content"] == "Entry 99"

    def test_clear(self, tmp_path):
        key = Cipher.generate_key()
        cipher = Cipher(key)
        wal = WALManager(tmp_path / "test.wal", cipher)

        wal.append("Test", {"type": "note"})
        assert wal.entry_count == 1

        wal.clear()
        assert wal.entry_count == 0


class TestWALReader:
    """Test WAL reader."""

    def test_empty_wal(self, tmp_path):
        key = Cipher.generate_key()
        cipher = Cipher(key)
        wal = WALManager(tmp_path / "nonexistent.wal", cipher)

        entries = wal.read_all()
        assert entries == []

    def test_corrupted_entry(self, tmp_path):
        key = Cipher.generate_key()
        cipher = Cipher(key)
        wal = WALManager(tmp_path / "test.wal", cipher)

        wal.append("Good entry", {"type": "note"})

        # Corrupt the file
        with open(tmp_path / "test.wal", "a") as f:
            f.write("corrupted-data\n")

        entries = wal.read_all()
        assert len(entries) == 2
        assert entries[0]["content"] == "Good entry"
        assert "CORRUPTED" in entries[1]["content"]
