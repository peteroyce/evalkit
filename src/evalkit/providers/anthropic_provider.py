"""Anthropic Claude provider using httpx async."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from evalkit.core.types import ModelResponse
from evalkit.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

# Cost per token (input, output) in USD
_COST_TABLE: dict[str, tuple[float, float]] = {
    "claude-3-5-sonnet-20241022": (3.00e-6, 15.00e-6),
    "claude-3-5-haiku-20241022": (0.80e-6, 4.00e-6),
    "claude-3-opus-20240229": (15.00e-6, 75.00e-6),
    "claude-3-sonnet-20240229": (3.00e-6, 15.00e-6),
    "claude-3-haiku-20240307": (0.25e-6, 1.25e-6),
    "claude-2.1": (8.00e-6, 24.00e-6),
    "claude-2.0": (8.00e-6, 24.00e-6),
}


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Claude models.

    Uses the Anthropic messages API format via httpx async.

    Args:
        api_key: Anthropic API key (starts with "sk-ant-").
        model: Model identifier (e.g., "claude-3-5-sonnet-20241022").
        temperature: Sampling temperature (0.0–1.0 for Claude).
        max_tokens: Maximum tokens in the completion.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._client = httpx.AsyncClient(
            headers={
                "x-api-key": api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )
        logger.debug("AnthropicProvider initialised for model '%s'", model)

    @property
    def provider_name(self) -> str:
        return "anthropic"

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
        """Call the Anthropic messages endpoint."""
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        logger.debug("Anthropic request: model=%s, prompt_len=%d", self._model, len(prompt))
        t0 = time.monotonic()

        try:
            response = await self._client.post(_ANTHROPIC_API_URL, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Anthropic API error %s: %s",
                exc.response.status_code,
                exc.response.text,
            )
            raise

        latency_ms = (time.monotonic() - t0) * 1000
        data = response.json()

        # Extract text from the content blocks
        content_blocks = data.get("content", [])
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )

        usage = data.get("usage", {})
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)

        logger.debug(
            "Anthropic response: latency=%.1fms, tokens_in=%d, tokens_out=%d",
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
        logger.debug("AnthropicProvider HTTP client closed")
