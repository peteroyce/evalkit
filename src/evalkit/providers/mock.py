"""MockProvider for testing — returns canned or random responses."""

from __future__ import annotations

import asyncio
import logging
import random
import string
from typing import Any

from evalkit.core.types import ModelResponse
from evalkit.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_LOREM_WORDS = [
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
    "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
    "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
    "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi",
    "aliquip", "ex", "ea", "commodo", "consequat",
]


def _random_text(min_words: int = 10, max_words: int = 50) -> str:
    count = random.randint(min_words, max_words)
    words = [random.choice(_LOREM_WORDS) for _ in range(count)]
    return " ".join(words).capitalize() + "."


class MockProvider(BaseProvider):
    """A mock provider for unit testing and offline development.

    Supports three response modes:
    - ``canned``: Returns responses from a pre-supplied list (cycling).
    - ``echo``: Echoes the prompt back as the response.
    - ``random``: Generates random lorem-ipsum text.

    Optionally simulates a configurable latency and token counts.

    Args:
        model: Fake model name to embed in responses.
        responses: Pre-defined responses for canned mode.
        mode: One of "canned", "echo", "random".
        simulated_latency_ms: Simulated latency added to each call.
        tokens_in_per_char: Token estimate multiplier for input text.
        tokens_out_per_char: Token estimate multiplier for output text.
        fail_every_n: If set, raises RuntimeError on every Nth call (for
            testing error-handling logic).
    """

    def __init__(
        self,
        model: str = "mock-model",
        responses: list[str] | None = None,
        mode: str = "canned",
        simulated_latency_ms: float = 10.0,
        tokens_in_per_char: float = 0.25,
        tokens_out_per_char: float = 0.25,
        fail_every_n: int | None = None,
    ) -> None:
        self._model = model
        self._responses = responses or ["This is a mock response."]
        self._mode = mode
        self._simulated_latency_ms = simulated_latency_ms
        self._tokens_in_per_char = tokens_in_per_char
        self._tokens_out_per_char = tokens_out_per_char
        self._fail_every_n = fail_every_n
        self._call_count = 0
        self._response_index = 0
        logger.debug("MockProvider initialised: model=%s, mode=%s", model, mode)

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Return a mock response according to the configured mode."""
        self._call_count += 1

        if self._fail_every_n and self._call_count % self._fail_every_n == 0:
            raise RuntimeError(
                f"MockProvider: simulated failure on call #{self._call_count}"
            )

        # Simulate network latency
        if self._simulated_latency_ms > 0:
            await asyncio.sleep(self._simulated_latency_ms / 1000.0)

        if self._mode == "echo":
            text = prompt
        elif self._mode == "random":
            text = _random_text()
        else:  # canned (default)
            text = self._responses[self._response_index % len(self._responses)]
            self._response_index += 1

        tokens_in = max(1, int(len(prompt) * self._tokens_in_per_char))
        tokens_out = max(1, int(len(text) * self._tokens_out_per_char))

        logger.debug(
            "MockProvider.generate call #%d: mode=%s, text_len=%d",
            self._call_count,
            self._mode,
            len(text),
        )

        return ModelResponse(
            text=text,
            model=self._model,
            provider=self.provider_name,
            latency_ms=self._simulated_latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=None,
            raw={"mock": True, "call_count": self._call_count},
        )

    def reset(self) -> None:
        """Reset call counters and response index."""
        self._call_count = 0
        self._response_index = 0
