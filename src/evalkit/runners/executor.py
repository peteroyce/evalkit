"""EvalExecutor — runs a single EvalCase against a provider with scoring."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from evalkit.core.types import EvalCase, EvalResult, Score
from evalkit.providers.base import BaseProvider
from evalkit.scorers.base import BaseScorer

logger = logging.getLogger(__name__)


class EvalExecutor:
    """Executes a single EvalCase against a provider and scores the output.

    Handles timeouts, retries, and aggregates scores from multiple scorers
    into a single EvalResult.

    Args:
        provider: The LLM provider to run evaluations against.
        scorers: List of scorer instances to apply to each output.
        timeout_seconds: Seconds before a provider call is cancelled.
        max_retries: Number of times to retry on transient failures.
        retry_delay_seconds: Base delay between retries (exponential backoff).
        scorer_weights: Optional mapping of scorer name to weight for
            computing aggregate_score. Scorers not in the map get weight 1.0.
    """

    def __init__(
        self,
        provider: BaseProvider,
        scorers: list[BaseScorer],
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        retry_delay_seconds: float = 1.0,
        scorer_weights: dict[str, float] | None = None,
    ) -> None:
        self._provider = provider
        self._scorers = scorers
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._retry_delay = retry_delay_seconds
        self._scorer_weights = scorer_weights or {}

    def _compute_aggregate(self, scores: list[Score]) -> float:
        """Compute a weighted average aggregate score."""
        if not scores:
            return 0.0
        total_weight = 0.0
        weighted_sum = 0.0
        for s in scores:
            weight = self._scorer_weights.get(s.scorer, 1.0)
            weighted_sum += weight * s.value
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def run(
        self,
        case: EvalCase,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Run a single EvalCase and return an EvalResult.

        Args:
            case: The evaluation case to run.
            generate_kwargs: Additional kwargs forwarded to provider.generate().

        Returns:
            An EvalResult with the response and all scorer outputs.
        """
        generate_kwargs = generate_kwargs or {}
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = self._retry_delay * (2 ** (attempt - 1))
                logger.info(
                    "EvalExecutor: retry %d/%d for case '%s' after %.1fs",
                    attempt,
                    self._max_retries,
                    case.id,
                    delay,
                )
                await asyncio.sleep(delay)

            try:
                response = await asyncio.wait_for(
                    self._provider.generate(
                        case.prompt,
                        system_prompt=case.system_prompt,
                        **generate_kwargs,
                    ),
                    timeout=self._timeout,
                )
                break
            except asyncio.TimeoutError as exc:
                logger.warning(
                    "EvalExecutor: timeout on case '%s' (attempt %d)",
                    case.id,
                    attempt + 1,
                )
                last_exc = exc
            except Exception as exc:
                logger.warning(
                    "EvalExecutor: error on case '%s' (attempt %d): %s",
                    case.id,
                    attempt + 1,
                    exc,
                )
                last_exc = exc
        else:
            # All retries exhausted
            logger.error(
                "EvalExecutor: case '%s' failed after %d attempts: %s",
                case.id,
                self._max_retries + 1,
                last_exc,
            )
            raise RuntimeError(
                f"EvalCase '{case.id}' failed after {self._max_retries + 1} attempts: {last_exc}"
            ) from last_exc

        # Score the output using all scorers
        scores: list[Score] = []
        for scorer in self._scorers:
            try:
                score = await scorer.score_async(
                    output=response.text,
                    expected=case.expected,
                    prompt=case.prompt,
                    case_id=case.id,
                    metadata=case.metadata,
                )
                scores.append(score)
                logger.debug(
                    "EvalExecutor: case '%s', scorer '%s' -> %.3f",
                    case.id,
                    scorer.name,
                    score.value,
                )
            except Exception as exc:
                logger.error(
                    "EvalExecutor: scorer '%s' failed on case '%s': %s",
                    scorer.name,
                    case.id,
                    exc,
                )
                scores.append(
                    Score(
                        value=0.0,
                        scorer=scorer.name,
                        reasoning=f"Scorer error: {exc}",
                        metadata={"error": str(exc)},
                    )
                )

        aggregate = self._compute_aggregate(scores)

        return EvalResult(
            case=case,
            response=response,
            scores=scores,
            aggregate_score=aggregate,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
