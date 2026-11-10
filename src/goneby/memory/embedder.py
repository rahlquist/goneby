"""Embedding client — OpenAI-compatible HTTP client for local or hosted embedders.

Talks to any server exposing `POST /v1/embeddings`:
- llama.cpp (`llama-server --embeddings`)
- vLLM, SGLang, TEI, Ollama
- OpenAI, Voyage, Cohere (OpenAI-compatible endpoints)

Supports Matryoshka truncation (MRL) so a 1024-dim model can emit 512-dim
vectors while keeping ~97% of retrieval quality at half the storage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend fails or returns an unusable payload."""


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding client."""

    base_url: str = "http://localhost:8081/v1"
    model: str = "qwen3-embedding-0.6b"
    api_key: str | None = None
    dimensions: int = 512
    timeout: float = 60.0
    batch_size: int = 16

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/embeddings"

    def validate(self) -> list[str]:
        errors = []
        if not self.base_url:
            errors.append("embedding base_url is required")
        if not self.model:
            errors.append("embedding model is required")
        if self.dimensions <= 0:
            errors.append(f"embedding dimensions must be > 0, got {self.dimensions}")
        return errors


def mrl_truncate(vector: list[float], dimensions: int) -> list[float]:
    """Truncate a Matryoshka embedding to `dimensions`.

    Qwen3-Embedding is trained with Matryoshka Representation Learning, so the
    leading `dimensions` values carry most of the semantic signal. The result
    is re-normalized so cosine similarity stays a plain dot product.
    """
    if dimensions >= len(vector):
        return vector
    return vector[:dimensions]


def l2_normalize(vector: list[float]) -> list[float]:
    """Normalize to unit length (guards against zero-norm vectors)."""
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class Embedder:
    """Async embedding client against an OpenAI-compatible endpoint."""

    def __init__(self, config: EmbeddingConfig):
        errors = config.validate()
        if errors:
            raise EmbeddingError("; ".join(errors))
        self._config = config
        self._client = httpx.AsyncClient(timeout=config.timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        """Probe the embedding server. Returns status dict, never raises.

        Tries the OpenAI-standard `/models` under the configured base first.
        Falls back to llama.cpp's `/health` probe, which needs no API path.
        """
        base = self._config.base_url.rstrip("/")
        candidates = [
            (f"{base}/models", "GET"),
            (f"{base}/health", "GET"),
        ]
        last_detail = "no probe endpoint answered"

        for url, method in candidates:
            try:
                response = await self._client.request(
                    method, url, timeout=5.0, headers=self._auth_headers()
                )
            except Exception as exc:  # noqa: BLE001 - health probe must not raise
                last_detail = f"{url}: {exc}"
                continue

            if response.status_code >= 400:
                last_detail = f"{url}: HTTP {response.status_code}"
                continue

            # A reachable, non-error response means the server is up.
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            models = [
                m.get("id")
                for m in (payload.get("data") or [])
                if isinstance(m, dict) and m.get("id")
            ]
            return {
                "ok": True,
                "endpoint": url,
                "models": models,
                "configured_model": self._config.model,
            }

        return {"ok": False, "detail": last_detail}

    def _auth_headers(self) -> dict[str, str]:
        if self._config.api_key:
            return {"Authorization": f"Bearer {self._config.api_key}"}
        return {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input, in order."""
        if not texts:
            return []
        if any(not t.strip() for t in texts):
            raise EmbeddingError("refusing to embed empty or whitespace-only text")

        headers = self._auth_headers()
        headers["Content-Type"] = "application/json"

        payload = {"model": self._config.model, "input": texts}
        try:
            response = await self._client.post(
                self._config.endpoint, json=payload, headers=headers
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"embedding server unreachable at {self._config.endpoint}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise EmbeddingError(
                f"embedding server returned HTTP {response.status_code}: {response.text[:300]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise EmbeddingError(f"embedding server returned non-JSON: {exc}") from exc

        items = body.get("data")
        if not isinstance(items, list):
            raise EmbeddingError(f"unexpected embeddings payload shape: {list(body)}")

        # The API may return items out of order; `index` is authoritative.
        indexed: list[tuple[int, list[float]]] = []
        for position, item in enumerate(items):
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise EmbeddingError(
                    f"embedding response item {position} has no usable vector"
                )
            indexed.append((item.get("index", position), vector))

        indexed.sort(key=lambda pair: pair[0])

        out: list[list[float]] = []
        for _, vector in indexed:
            truncated = mrl_truncate(vector, self._config.dimensions)
            out.append(l2_normalize(truncated))
        return out

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]

    async def embed_batched(
        self, texts: list[str], progress: callable | None = None
    ) -> list[list[float]]:
        """Embed in batches of `batch_size` to bound request size."""
        results: list[list[float]] = []
        for start in range(0, len(texts), self._config.batch_size):
            chunk = texts[start : start + self._config.batch_size]
            results.extend(await self.embed(chunk))
            if progress is not None:
                progress(min(start + len(chunk), len(texts)), len(texts))
        return results
