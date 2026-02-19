"""OpenAI-compatible API provider using httpx async."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from evalkit.core.types import ModelResponse
from evalkit.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# Default cost per token (input/output) for common models — USD per token
_COST_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50e-6, 10.00e-6),
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-4-turbo": (10.00e-6, 30.00e-6),
    "gpt-4": (30.00e-6, 60.00e-6),
    "gpt-3.5-turbo": (0.50e-6, 1.50e-6),
    "o1": (15.00e-6, 60.00e-6),
    "o1-mini": (3.00e-6, 12.00e-6),
}


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI and OpenAI-compatible APIs.

    Supports any endpoint that follows the OpenAI chat completions format,
    including Azure OpenAI, Together AI, Groq, and local servers such as
    Ollama or vLLM.

    Args:
        api_key: API key for authentication.
        model: Model identifier (e.g., "gpt-4o").
        base_url: Base URL for the API. Defaults to OpenAI's public endpoint.
        temperature: Sampling temperature (0.0–2.0).
        max_tokens: Maximum tokens in the completion.
        timeout: HTTP request timeout in seconds.
        extra_headers: Additional HTTP headers to include.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )
        logger.debug("OpenAIProvider initialised for model '%s' at '%s'", model, base_url)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def _estimate_cost(self, tokens_in: int, tokens_out: int) -> float | None:
        costs = _COST_TABLE.get(self._model)
        if costs is None:
            return None
        return tokens_in * costs[0] + tokens_out * costs[1]

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Call the OpenAI chat completions endpoint."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }

        logger.debug("OpenAI request: model=%s, prompt_len=%d", self._model, len(prompt))
        t0 = time.monotonic()

        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "OpenAI API error %s: %s",
                exc.response.status_code,
                exc.response.text,
            )
            raise

        latency_ms = (time.monotonic() - t0) * 1000
        data = response.json()

        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        logger.debug(
            "OpenAI response: latency=%.1fms, tokens_in=%d, tokens_out=%d",
            latency_ms,
            tokens_in,
            tokens_out,
        )

        return ModelResponse(
            text=text,
            model=self._model,
            provider=self.provider_name,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._estimate_cost(tokens_in, tokens_out),
            raw=data,
        )

    async def close(self) -> None:
        await self._client.aclose()
        logger.debug("OpenAIProvider HTTP client closed")
