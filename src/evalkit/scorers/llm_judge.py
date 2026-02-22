"""LLMJudgeScorer — uses an LLM to grade outputs with a rubric."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, TYPE_CHECKING

from evalkit.core.types import Score
from evalkit.scorers.base import BaseScorer

if TYPE_CHECKING:
    from evalkit.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_DEFAULT_RUBRIC = """You are an expert evaluator. Your task is to evaluate the quality of an AI model's response.

## Question / Task
{prompt}

## Model Response
{output}

{expected_section}

## Evaluation Rubric
Rate the response on a scale from 1 to 5:
- 5: Excellent — completely correct, comprehensive, and well-reasoned
- 4: Good — mostly correct with minor issues or missing nuances
- 3: Acceptable — partially correct but has notable gaps or errors
- 2: Poor — mostly incorrect or significantly incomplete
- 1: Unacceptable — completely wrong or irrelevant

## Instructions
Think step by step, then provide your rating as a JSON object on its own line:
{{"rating": <integer 1-5>, "reasoning": "<brief explanation>"}}

Your rating:"""

_EXPECTED_SECTION_TEMPLATE = "## Reference Answer\n{expected}\n"


def _extract_rating(text: str) -> tuple[int, str]:
    """Parse a 1-5 rating and reasoning from the judge LLM's response.

    Tries multiple parsing strategies in order:
    1. JSON block with "rating" key
    2. Plain integer on its own line
    3. Integer anywhere in the text
    """
    # Strategy 1: JSON with rating key
    json_match = re.search(
        r'\{[^}]*"rating"\s*:\s*([1-5])[^}]*"reasoning"\s*:\s*"([^"]*)"[^}]*\}',
        text,
        re.DOTALL,
    )
    if json_match:
        return int(json_match.group(1)), json_match.group(2)

    # Strategy 1b: JSON with rating only
    json_rating = re.search(r'\{[^}]*"rating"\s*:\s*([1-5])[^}]*\}', text)
    if json_rating:
        return int(json_rating.group(1)), "Extracted from JSON."

    # Strategy 2: "Rating: N" pattern
    rating_match = re.search(r"\brating\s*:\s*([1-5])\b", text, re.IGNORECASE)
    if rating_match:
        return int(rating_match.group(1)), "Extracted from 'Rating: N' pattern."

    # Strategy 3: standalone digit on its own line
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in {"1", "2", "3", "4", "5"}:
            return int(stripped), "Extracted from standalone digit."

    # Strategy 4: first digit in range found anywhere
    digit_match = re.search(r"\b([1-5])\b", text)
    if digit_match:
        return int(digit_match.group(1)), "Extracted via fallback digit search."

    raise ValueError(f"Could not parse a 1-5 rating from judge output: {text[:200]!r}")


class LLMJudgeScorer(BaseScorer):
    """Uses an LLM to evaluate model outputs against a configurable rubric.

    The judge LLM receives the prompt, the model's output, optionally the
    reference answer, and a rubric. It returns a 1-5 rating which is
    normalized to [0.0, 1.0].

    Args:
        provider: A BaseProvider instance to use as the judge.
        rubric_template: Jinja2-compatible template string with placeholders
            {prompt}, {output}, {expected_section}. Defaults to a built-in
            general-purpose rubric.
        temperature: Temperature for the judge LLM (low for consistency).
        include_expected: Whether to include the reference answer in the
            judge prompt (if available).
    """

    def __init__(
        self,
        provider: "BaseProvider",
        rubric_template: str = _DEFAULT_RUBRIC,
        temperature: float = 0.0,
        include_expected: bool = True,
    ) -> None:
        self._provider = provider
        self._rubric_template = rubric_template
        self._temperature = temperature
        self._include_expected = include_expected

    @property
    def name(self) -> str:
        return "llm_judge"

    def _build_prompt(
        self,
        output: str,
        expected: str | None,
        prompt_context: str | None,
    ) -> str:
        expected_section = ""
        if self._include_expected and expected:
            expected_section = _EXPECTED_SECTION_TEMPLATE.format(expected=expected)

        return self._rubric_template.format(
            prompt=prompt_context or "(not provided)",
            output=output,
            expected_section=expected_section,
        )

    async def score_async(
        self,
        output: str,
        expected: str | None = None,
        **kwargs: Any,
    ) -> Score:
        """Async score — calls the judge provider."""
        prompt_context = kwargs.get("prompt")  # original eval case prompt
        judge_prompt = self._build_prompt(output, expected, prompt_context)

        logger.debug(
            "LLMJudgeScorer: calling judge model %s/%s",
            self._provider.provider_name,
            self._provider.model_name,
        )

        try:
            response = await self._provider.generate(
                judge_prompt,
                temperature=self._temperature,
            )
            raw_text = response.text
        except Exception as exc:
            logger.error("LLMJudgeScorer: judge call failed: %s", exc)
            return Score(
                value=0.0,
                scorer=self.name,
                reasoning=f"Judge call failed: {exc}",
                metadata={"error": str(exc)},
            )

        try:
            rating, reasoning = _extract_rating(raw_text)
        except ValueError as exc:
            logger.warning("LLMJudgeScorer: rating parse failed: %s", exc)
            return Score(
                value=0.5,
                scorer=self.name,
                reasoning=f"Could not parse rating: {exc}. Raw: {raw_text[:200]}",
                metadata={"raw_judge_output": raw_text, "parse_error": str(exc)},
            )

        # Normalize 1-5 to 0.0-1.0
        value = (rating - 1) / 4.0

        logger.debug("LLMJudgeScorer: rating=%d, normalized=%.2f", rating, value)

        return Score(
            value=value,
            scorer=self.name,
            reasoning=reasoning,
            metadata={
                "raw_rating": rating,
                "raw_judge_output": raw_text,
                "judge_model": self._provider.model_name,
            },
        )

    def score(
        self,
        output: str,
        expected: str | None = None,
        **kwargs: Any,
    ) -> Score:
        """Synchronous wrapper around score_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If called from within an async context, create a new thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run, self.score_async(output, expected, **kwargs)
                    )
                    return future.result()
            return loop.run_until_complete(self.score_async(output, expected, **kwargs))
        except RuntimeError:
            return asyncio.run(self.score_async(output, expected, **kwargs))
