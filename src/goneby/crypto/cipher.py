"""Per-entry ChaCha20-Poly1305 encryption.

Each WAL entry is encrypted independently with a random 96-bit nonce.
Format per entry: nonce (12 bytes) || ciphertext || tag (16 bytes).
"""

from __future__ import annotations

import os

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

NONCE_SIZE = 12  # 96-bit nonce for ChaCha20-Poly1305
KEY_SIZE = 32    # 256-bit key


class Cipher:
    """Encrypt and decrypt individual WAL entries."""

    def __init__(self, key: bytes):
        if len(key) != KEY_SIZE:
            raise ValueError(f"Key must be {KEY_SIZE} bytes, got {len(key)}")
        self._cipher = ChaCha20Poly1305(key)

    @staticmethod
    def generate_key() -> bytes:
        """Generate a new random 256-bit key."""
        return os.urandom(KEY_SIZE)

    def encrypt(self, plaintext: bytes, associated_data: bytes | None = None) -> bytes:
        """Encrypt a single entry. Returns nonce + ciphertext + tag."""
        nonce = os.urandom(NONCE_SIZE)
        ciphertext = self._cipher.encrypt(nonce, plaintext, associated_data)
        return nonce + ciphertext

    def decrypt(self, data: bytes, associated_data: bytes | None = None) -> bytes:
        """Decrypt a single entry. Expects nonce + ciphertext + tag."""
        if len(data) < NONCE_SIZE + 16:  # nonce + minimum tag
            raise ValueError(f"Ciphertext too short: {len(data)} bytes")
        nonce = data[:NONCE_SIZE]
        ciphertext = data[NONCE_SIZE:]
        return self._cipher.decrypt(nonce, ciphertext, associated_data)
