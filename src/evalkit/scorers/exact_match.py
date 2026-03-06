"""ExactMatchScorer — case-insensitive exact string match."""

from __future__ import annotations

import logging
import re
from typing import Any

from evalkit.core.types import Score
from evalkit.scorers.base import BaseScorer

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Normalize text: lowercase, collapse whitespace, strip punctuation edges."""
    text = text.lower().strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


class ExactMatchScorer(BaseScorer):
    """Scores 1.0 if the normalized output exactly matches the expected text.

    Normalization: lowercase, strip leading/trailing whitespace, collapse
    internal whitespace to single spaces.

    Args:
        strip_punctuation: If True, also strips leading/trailing punctuation
            before comparison (e.g., trailing periods or commas).
    """

    def __init__(self, strip_punctuation: bool = False) -> None:
        self._strip_punctuation = strip_punctuation

    @property
    def name(self) -> str:
        return "exact_match"

    def _prepare(self, text: str) -> str:
        normalized = _normalize(text)
        if self._strip_punctuation:
            normalized = normalized.strip(".,!?;:\"'")
        return normalized

    def score(
        self,
        output: str,
        expected: str | None = None,
        **kwargs: Any,
    ) -> Score:
        if expected is None:
            logger.warning("ExactMatchScorer called without expected; returning 0.0")
            return Score(
                value=0.0,
                scorer=self.name,
                reasoning="No expected answer provided.",
            )

        norm_output = self._prepare(output)
        norm_expected = self._prepare(expected)
        is_match = norm_output == norm_expected
        value = 1.0 if is_match else 0.0

        logger.debug(
            "ExactMatch: match=%s, output=%r, expected=%r",
            is_match,
            norm_output[:80],
            norm_expected[:80],
        )

        return Score(
            value=value,
            scorer=self.name,
            reasoning=(
                "Exact match." if is_match
                else f"Output '{norm_output[:60]}' != expected '{norm_expected[:60]}'."
            ),
            metadata={
                "normalized_output": norm_output,
                "normalized_expected": norm_expected,
            },
        )
