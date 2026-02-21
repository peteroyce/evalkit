"""Abstract base class for all scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from evalkit.core.types import Score


class BaseScorer(ABC):
    """Abstract base class for evaluation scorers.

    A scorer takes the model's output text and optionally the expected
    reference answer, and returns a Score with a value between 0.0 and 1.0.

    Scorers may be synchronous or asynchronous. The base class provides a
    synchronous `score` method; subclasses that need async (e.g., LLM judge)
    should override `score_async` and have `score` call `asyncio.run`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """A short, unique identifier for this scorer."""
        ...

    @abstractmethod
    def score(
        self,
        output: str,
        expected: str | None = None,
        **kwargs: Any,
    ) -> Score:
        """Score a model output.

        Args:
            output: The model's generated text.
            expected: The reference/expected answer (may be None for
                      scorers that don't require it, e.g., toxicity checks).
            **kwargs: Additional context passed through from the runner.

        Returns:
            A Score object with value in [0.0, 1.0].
        """
        ...

    async def score_async(
        self,
        output: str,
        expected: str | None = None,
        **kwargs: Any,
    ) -> Score:
        """Async version of score(). Default wraps the sync implementation."""
        return self.score(output, expected, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
