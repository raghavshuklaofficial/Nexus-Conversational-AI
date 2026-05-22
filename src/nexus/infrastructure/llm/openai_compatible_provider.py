"""
OpenAI-compatible API provider.

Works with OpenAI, Azure OpenAI, Ollama, vLLM, LocalAI, LM Studio —
anything exposing the /v1/chat/completions endpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import structlog

from nexus.domain.ports import LLMProviderPort

logger = structlog.get_logger(__name__)


class OpenAICompatibleProvider(LLMProviderPort):
    """Generate text via an OpenAI-compatible REST API."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = None

    async def initialize(self) -> None:
        import httpx
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        logger.info("openai_compatible_provider_initialized", model=self._model)

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self._client:
            raise RuntimeError("Provider not initialized")

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
            "stream": False,
        }

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        if not self._client:
            raise RuntimeError("Provider not initialized")

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
            "stream": True,
        }

        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    import json
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content

    async def health_check(self) -> bool:
        if not self._client:
            return False
        try:
            response = await self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    @property
    def model_name(self) -> str:
        return self._model
