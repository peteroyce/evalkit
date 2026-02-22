"""CompositeScorer — weighted combination of multiple scorers."""

from __future__ import annotations

import logging
from typing import Any

from evalkit.core.types import Score
from evalkit.scorers.base import BaseScorer

logger = logging.getLogger(__name__)


class CompositeScorer(BaseScorer):
    """Combines multiple scorers with configurable weights.

    The aggregate score is a weighted average:
    ``score = sum(weight_i * scorer_i.score) / sum(weight_i)``

    Args:
        scorers: List of (BaseScorer, weight) tuples. Weights do not need to
                 sum to 1 — they are normalized internally.
        name_override: Override the scorer name (default: "composite").
    """

    def __init__(
        self,
        scorers: list[tuple[BaseScorer, float]],
        name_override: str = "composite",
    ) -> None:
        if not scorers:
            raise ValueError("CompositeScorer requires at least one scorer.")
        if any(w <= 0 for _, w in scorers):
            raise ValueError("All scorer weights must be positive.")
        self._scorers = scorers
        self._name = name_override
        total_weight = sum(w for _, w in scorers)
        self._normalized_weights = [(scorer_inst, w / total_weight) for scorer_inst, w in scorers]
        logger.debug(
            "CompositeScorer initialised with scorers: %s",
            [(scorer_inst.name, w) for scorer_inst, w in self._normalized_weights],
        )

    @property
    def name(self) -> str:
        return self._name

    def score(
        self,
        output: str,
        expected: str | None = None,
        **kwargs: Any,
    ) -> Score:
        component_scores: list[Score] = []
        aggregate = 0.0

        for scorer_inst, weight in self._normalized_weights:
            component = scorer_inst.score(output, expected, **kwargs)
            component_scores.append(component)
            aggregate += weight * component.value
            logger.debug(
                "CompositeScorer: scorer=%s, weight=%.3f, score=%.3f",
                scorer_inst.name,
                weight,
                component.value,
            )

        components_summary = ", ".join(
            f"{c.scorer}={c.value:.3f}" for c in component_scores
        )
        return Score(
            value=round(aggregate, 6),
            scorer=self.name,
            reasoning=f"Weighted average of [{components_summary}].",
            metadata={
                "components": [c.to_dict() for c in component_scores],
                "weights": {scorer_inst.name: w for scorer_inst, w in self._normalized_weights},
            },
        )

    async def score_async(
        self,
        output: str,
        expected: str | None = None,
        **kwargs: Any,
    ) -> Score:
        import asyncio

        tasks = [
            scorer_inst.score_async(output, expected, **kwargs)
            for scorer_inst, _ in self._normalized_weights
        ]
        component_scores: list[Score] = list(await asyncio.gather(*tasks))

        aggregate = sum(
            w * component.value
            for (_, w), component in zip(self._normalized_weights, component_scores)
        )

        components_summary = ", ".join(
            f"{c.scorer}={c.value:.3f}" for c in component_scores
        )
        return Score(
            value=round(aggregate, 6),
            scorer=self.name,
            reasoning=f"Weighted average of [{components_summary}].",
            metadata={
                "components": [c.to_dict() for c in component_scores],
                "weights": {scorer_inst.name: w for scorer_inst, w in self._normalized_weights},
            },
        )
