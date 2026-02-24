"""EloRating — Elo rating system from pairwise comparisons."""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_INITIAL_RATING = 1500.0
_DEFAULT_K = 32.0


class EloRating:
    """Implements an Elo rating system for LLM model comparison.

    Models start with an initial rating of 1500. Ratings are updated after
    each pairwise comparison using the standard Elo formula.

    Args:
        models: Initial list of model names to register.
        initial_rating: Starting Elo rating for all models.
        k: K-factor (controls how quickly ratings change per game).
    """

    def __init__(
        self,
        models: list[str] | None = None,
        initial_rating: float = _DEFAULT_INITIAL_RATING,
        k: float = _DEFAULT_K,
    ) -> None:
        self._initial_rating = initial_rating
        self._k = k
        self._ratings: dict[str, float] = {}
        self._match_counts: dict[str, int] = {}
        self._history: list[dict[str, Any]] = []

        for model in (models or []):
            self.add_model(model)

    def add_model(self, model: str) -> None:
        """Register a new model with the initial rating."""
        if model not in self._ratings:
            self._ratings[model] = self._initial_rating
            self._match_counts[model] = 0
            logger.debug("EloRating: added model '%s' with rating %.1f", model, self._initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Compute the expected score for player A against player B.

        Returns a value in (0, 1) representing the probability that A wins.
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def update(self, winner: str, loser: str, outcome: float = 1.0) -> tuple[float, float]:
        """Update Elo ratings after a match.

        Args:
            winner: Name of the winning model (or model A in case of a draw).
            loser: Name of the losing model (or model B in case of a draw).
            outcome: Actual score for the winner (1.0 = win, 0.5 = draw, 0.0 = loss).
                     Pass 0.5 for ties.

        Returns:
            Tuple of (new_winner_rating, new_loser_rating).
        """
        self.add_model(winner)
        self.add_model(loser)

        ra = self._ratings[winner]
        rb = self._ratings[loser]

        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)

        # outcome is from winner's perspective; loser gets (1 - outcome)
        new_ra = ra + self._k * (outcome - ea)
        new_rb = rb + self._k * ((1 - outcome) - eb)

        self._ratings[winner] = new_ra
        self._ratings[loser] = new_rb
        self._match_counts[winner] += 1
        self._match_counts[loser] += 1

        self._history.append({
            "winner": winner,
            "loser": loser,
            "outcome": outcome,
            "before": {"winner": ra, "loser": rb},
            "after": {"winner": new_ra, "loser": new_rb},
            "delta": {"winner": new_ra - ra, "loser": new_rb - rb},
        })

        logger.debug(
            "EloRating: %s vs %s (outcome=%.1f) -> %s: %.1f->%.1f, %s: %.1f->%.1f",
            winner, loser, outcome,
            winner, ra, new_ra,
            loser, rb, new_rb,
        )

        return new_ra, new_rb

    def get_ratings(self) -> dict[str, float]:
        """Return a copy of the current ratings dict."""
        return dict(self._ratings)

    def get_rating(self, model: str) -> float:
        """Get the current Elo rating for a model."""
        if model not in self._ratings:
            raise KeyError(f"Model '{model}' not registered in EloRating.")
        return self._ratings[model]

    def get_match_count(self, model: str) -> int:
        """Get the number of matches played by a model."""
        return self._match_counts.get(model, 0)

    def get_leaderboard(self) -> list[tuple[str, float, int]]:
        """Return models sorted by rating descending.

        Returns:
            List of (model_name, rating, match_count) tuples.
        """
        return sorted(
            [(m, r, self._match_counts[m]) for m, r in self._ratings.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    def get_history(self) -> list[dict[str, Any]]:
        """Return the full match history."""
        return list(self._history)

    def reset(self) -> None:
        """Reset all ratings to the initial value and clear history."""
        for model in self._ratings:
            self._ratings[model] = self._initial_rating
            self._match_counts[model] = 0
        self._history.clear()
        logger.debug("EloRating: reset all ratings to %.1f", self._initial_rating)

    def __repr__(self) -> str:
        leaderboard = self.get_leaderboard()
        parts = [f"{m}: {r:.1f}" for m, r, _ in leaderboard[:5]]
        return f"EloRating(k={self._k}, ratings=[{', '.join(parts)}])"
