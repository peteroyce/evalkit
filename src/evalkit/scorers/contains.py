"""ContainsScorer — checks if output contains expected substrings."""

from __future__ import annotations

import logging
from typing import Any

from evalkit.core.types import Score
from evalkit.scorers.base import BaseScorer

logger = logging.getLogger(__name__)


class ContainsScorer(BaseScorer):
    """Scores based on the fraction of required substrings found in the output.

    The score is ``found / total`` where ``total`` is the number of required
    substrings and ``found`` is the number that appear in the model output.

    Args:
        case_sensitive: If False (default), comparisons ignore case.
        substrings: A fixed list of substrings to check for. When provided,
            the ``expected`` parameter to ``score()`` is ignored. When None,
            the ``expected`` value is treated as a single required substring.
        require_all: If True and multiple substrings are configured, the score
            is binary (1.0 only if all are found, else 0.0).
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        substrings: list[str] | None = None,
        require_all: bool = False,
    ) -> None:
        self._case_sensitive = case_sensitive
        self._fixed_substrings = substrings
        self._require_all = require_all

    @property
    def name(self) -> str:
        return "contains"

    def _normalize(self, text: str) -> str:
        return text if self._case_sensitive else text.lower()

    def score(
        self,
        output: str,
        expected: str | None = None,
        **kwargs: Any,
    ) -> Score:
        # Determine the list of substrings to check
        if self._fixed_substrings is not None:
            targets = self._fixed_substrings
        elif expected is not None:
            targets = [expected]
        else:
            logger.warning("ContainsScorer called without expected or fixed substrings; returning 0.0")
            return Score(
                value=0.0,
                scorer=self.name,
                reasoning="No expected answer or fixed substrings provided.",
            )

        norm_output = self._normalize(output)
        found = [t for t in targets if self._normalize(t) in norm_output]
        missing = [t for t in targets if self._normalize(t) not in norm_output]

        if self._require_all:
            value = 1.0 if len(missing) == 0 else 0.0
        else:
            value = len(found) / len(targets) if targets else 0.0

        logger.debug(
            "ContainsScorer: found=%d/%d, require_all=%s",
            len(found),
            len(targets),
            self._require_all,
        )

        return Score(
            value=value,
            scorer=self.name,
            reasoning=(
                f"Found {len(found)}/{len(targets)} required substrings."
                + (f" Missing: {missing[:3]}" if missing else "")
            ),
            metadata={
                "found": found,
                "missing": missing,
                "total": len(targets),
            },
        )
