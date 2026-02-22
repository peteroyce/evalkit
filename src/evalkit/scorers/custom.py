"""CustomScorer — wraps a user-provided Python callable."""

from __future__ import annotations

import logging
from typing import Any, Callable

from evalkit.core.types import Score
from evalkit.scorers.base import BaseScorer

logger = logging.getLogger(__name__)

# Type alias for the scoring function signature
ScoringFn = Callable[[str, str | None], float]


class CustomScorer(BaseScorer):
    """Wraps an arbitrary Python function as a scorer.

    The provided callable must accept ``(output: str, expected: str | None)``
    and return a float in [0.0, 1.0].

    Args:
        fn: The scoring function to wrap.
        scorer_name: The name to give this scorer (default: "custom").
        description: Optional human-readable description of what this
            scorer evaluates.

    Example::

        def my_scorer(output: str, expected: str | None) -> float:
            return 1.0 if "Paris" in output else 0.0

        scorer = CustomScorer(fn=my_scorer, scorer_name="contains_paris")
    """

    def __init__(
        self,
        fn: ScoringFn,
        scorer_name: str = "custom",
        description: str | None = None,
    ) -> None:
        self._fn = fn
        self._scorer_name = scorer_name
        self._description = description
        logger.debug("CustomScorer initialised: name=%r", scorer_name)

    @property
    def name(self) -> str:
        return self._scorer_name

    def score(
        self,
        output: str,
        expected: str | None = None,
        **kwargs: Any,
    ) -> Score:
        try:
            value = self._fn(output, expected)
        except Exception as exc:
            logger.error("CustomScorer '%s' raised an exception: %s", self.name, exc)
            return Score(
                value=0.0,
                scorer=self.name,
                reasoning=f"Scorer function raised an exception: {exc}",
                metadata={"error": str(exc)},
            )

        # Clamp to [0, 1] and warn if out of range
        if not 0.0 <= value <= 1.0:
            logger.warning(
                "CustomScorer '%s' returned value %f outside [0, 1]; clamping.",
                self.name,
                value,
            )
            value = max(0.0, min(1.0, value))

        return Score(
            value=value,
            scorer=self.name,
            reasoning=self._description or f"Custom scorer '{self.name}'.",
            metadata={"fn": self._fn.__name__},
        )
