from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import httpx

from app.config import Settings, get_settings
from app.schemas import MultiModalInput

DEFAULT_INSTRUCTION = "Represent the user's input."


@dataclass
class EmbeddingResult:
    embeddings: list[list[float]]
    prompt_tokens: int = 0
    total_tokens: int = 0


class Embedder(Protocol):
    async def embed(self, items: list[MultiModalInput], dimensions: int | None = None) -> EmbeddingResult:
        ...


def apply_dimensions(embedding: list[float], dimensions: int | None) -> list[float]:
    if dimensions is None:
        return embedding
    if dimensions > len(embedding):
        raise ValueError(f"dimensions must be <= embedding size {len(embedding)}")
    return embedding[:dimensions]


def build_messages(item: MultiModalInput) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if item.image:
        content.append({"type": "image_url", "image_url": {"url": item.image}})
    if item.text:
        content.append({"type": "text", "text": item.text})
    elif item.image:
        content.append({"type": "text", "text": DEFAULT_INSTRUCTION})
    return [{"role": "user", "content": content}]


class UpstreamEmbeddingError(RuntimeError):
    pass


class UpstreamEmbeddingClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def embed(self, items: list[MultiModalInput], dimensions: int | None = None) -> EmbeddingResult:
        if all(item.image is None for item in items):
            return await self._embed_text_batch(items, dimensions)
        return await self._embed_multimodal_items(items, dimensions)

    async def _embed_text_batch(self, items: list[MultiModalInput], dimensions: int | None) -> EmbeddingResult:
        input_value: str | list[str]
        texts = [item.text or "" for item in items]
        input_value = texts[0] if len(texts) == 1 else texts
        return await self._post_embeddings({"input": input_value}, dimensions)

    async def _embed_multimodal_items(self, items: list[MultiModalInput], dimensions: int | None) -> EmbeddingResult:
        embeddings: list[list[float]] = []
        prompt_tokens = 0
        total_tokens = 0
        async with httpx.AsyncClient(timeout=self.settings.upstream_timeout) as client:
            for item in items:
                body = self._base_body(dimensions)
                if item.image:
                    body["messages"] = build_messages(item)
                else:
                    body["input"] = item.text or ""
                payload = await self._post(client, body)
                embeddings.extend(_extract_embeddings(payload, dimensions=None))
                usage = payload.get("usage") or {}
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                total_tokens += int(usage.get("total_tokens") or 0)
        return EmbeddingResult(embeddings=embeddings, prompt_tokens=prompt_tokens, total_tokens=total_tokens)

    async def _post_embeddings(self, body_part: dict[str, Any], dimensions: int | None) -> EmbeddingResult:
        body = self._base_body(dimensions)
        body.update(body_part)
        async with httpx.AsyncClient(timeout=self.settings.upstream_timeout) as client:
            payload = await self._post(client, body)
        usage = payload.get("usage") or {}
        return EmbeddingResult(
            embeddings=_extract_embeddings(payload, dimensions=None),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
        )

    def _base_body(self, dimensions: int | None) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.settings.upstream_model, "encoding_format": "float"}
        if dimensions is not None:
            body["dimensions"] = dimensions
        return body

    async def _post(self, client: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await client.post(str(self.settings.upstream_embeddings_url), json=body)
        except httpx.HTTPError as exc:
            raise UpstreamEmbeddingError(f"failed to call upstream embedding service: {exc}") from exc
        if response.status_code >= 400:
            raise UpstreamEmbeddingError(f"upstream embedding service returned {response.status_code}: {response.text}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamEmbeddingError("upstream embedding service returned non-JSON response") from exc
        if not isinstance(payload, dict):
            raise UpstreamEmbeddingError("upstream embedding service returned invalid response")
        return payload


def _extract_embeddings(payload: dict[str, Any], dimensions: int | None) -> list[list[float]]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise UpstreamEmbeddingError("upstream response missing data list")
    embeddings = []
    for item in sorted(data, key=lambda value: int(value.get("index", 0))):
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise UpstreamEmbeddingError("upstream response item missing embedding")
        embeddings.append(apply_dimensions([float(value) for value in embedding], dimensions))
    return embeddings


@lru_cache
def get_embedder() -> UpstreamEmbeddingClient:
    return UpstreamEmbeddingClient(get_settings())
