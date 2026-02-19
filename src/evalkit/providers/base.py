"""Abstract base class for LLM providers."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from evalkit.core.types import ModelResponse

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement `generate`. The `generate_batch` method
    provides a default implementation using asyncio.gather, but subclasses
    may override it for more efficient batching.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """The canonical name of this provider (e.g., 'openai', 'anthropic')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier used for API calls."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response to a single prompt.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: Optional system/instruction prompt.
            **kwargs: Additional provider-specific parameters
                      (temperature, max_tokens, etc.).

        Returns:
            A ModelResponse with text, latency, and token counts.
        """
        ...

    async def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        concurrency: int = 5,
        **kwargs: Any,
    ) -> list[ModelResponse]:
        """Generate responses for a batch of prompts.

        Default implementation uses asyncio.gather with a semaphore for
        concurrency control. Subclasses may override for native batching.

        Args:
            prompts: List of user prompts.
            system_prompt: Optional system prompt applied to all calls.
            concurrency: Maximum number of concurrent API calls.
            **kwargs: Additional parameters forwarded to generate().

        Returns:
            List of ModelResponse objects, one per prompt (in order).
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded_generate(p: str) -> ModelResponse:
            async with semaphore:
                return await self.generate(p, system_prompt=system_prompt, **kwargs)

        logger.debug(
            "Generating batch of %d prompts with concurrency=%d on %s/%s",
            len(prompts),
            concurrency,
            self.provider_name,
            self.model_name,
        )
        return list(await asyncio.gather(*[_bounded_generate(p) for p in prompts]))

    async def close(self) -> None:
        """Release any resources held by this provider (e.g., HTTP client)."""
        pass

    async def __aenter__(self) -> "BaseProvider":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name!r}, model={self.model_name!r})"
