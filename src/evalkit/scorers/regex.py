"""RegexScorer — pattern matching with named group extraction."""

from __future__ import annotations

import logging
import re
from typing import Any

from evalkit.core.types import Score
from evalkit.scorers.base import BaseScorer

logger = logging.getLogger(__name__)


class RegexScorer(BaseScorer):
    """Scores 1.0 if the model output matches a regular expression, 0.0 otherwise.

    Named capture groups from the match are included in the score metadata,
    making it easy to extract structured data (e.g., a JSON field, a number)
    from free-form model output.

    Args:
        pattern: The regular expression pattern to match against.
        flags: Regex flags (e.g., re.IGNORECASE | re.DOTALL). Defaults to
               re.IGNORECASE.
        search: If True (default), uses ``re.search`` (match anywhere). If
                False, uses ``re.fullmatch`` (entire string must match).
        partial_credit: If True, score is the ratio of named groups that
                        matched to total named groups (useful when the regex
                        has multiple captures). Defaults to False.
    """

    def __init__(
        self,
        pattern: str,
        flags: int = re.IGNORECASE,
        search: bool = True,
        partial_credit: bool = False,
    ) -> None:
        self._pattern_str = pattern
        self._compiled = re.compile(pattern, flags)
        self._search = search
        self._partial_credit = partial_credit
        logger.debug("RegexScorer compiled pattern: %r", pattern)

    @property
    def name(self) -> str:
        return "regex"

    def score(
        self,
        output: str,
        expected: str | None = None,
        **kwargs: Any,
    ) -> Score:
        if self._search:
            match = self._compiled.search(output)
        else:
            match = self._compiled.fullmatch(output)

        if match is None:
            return Score(
                value=0.0,
                scorer=self.name,
                reasoning=f"Pattern {self._pattern_str!r} not found in output.",
                metadata={"pattern": self._pattern_str, "matched": False},
            )

        named_groups = match.groupdict()
        logger.debug("RegexScorer match found; groups=%s", named_groups)

        if self._partial_credit and named_groups:
            filled = sum(1 for v in named_groups.values() if v is not None)
            value = filled / len(named_groups)
            reasoning = (
                f"Pattern matched with {filled}/{len(named_groups)} named groups captured."
            )
        else:
            value = 1.0
            reasoning = f"Pattern {self._pattern_str!r} matched."

        return Score(
            value=value,
            scorer=self.name,
            reasoning=reasoning,
            metadata={
                "pattern": self._pattern_str,
                "matched": True,
                "groups": named_groups,
                "span": list(match.span()),
            },
        )
