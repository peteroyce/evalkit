"""ComparisonAnalyzer — model-vs-model statistics and win rates."""

from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict
from typing import Any

from evalkit.comparison.elo import EloRating
from evalkit.core.types import ComparisonResult, EvalResult

logger = logging.getLogger(__name__)


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted data list (linear interpolation)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    index = (p / 100) * (n - 1)
    lower = int(index)
    upper = min(lower + 1, n - 1)
    fraction = index - lower
    return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])


class ComparisonAnalyzer:
    """Analyzes and compares evaluation results across multiple models.

    Takes a mapping of model label -> list of EvalResult (all run on the
    same suite) and computes per-model statistics, head-to-head win rates,
    and Elo ratings derived from pairwise score comparisons.

    Args:
        results: Dict mapping model label to list of EvalResult objects.
        elo_k: K-factor for the Elo rating system.
        tie_threshold: Score difference below which a match is considered a tie.
    """

    def __init__(
        self,
        results: dict[str, list[EvalResult]],
        elo_k: float = 32.0,
        tie_threshold: float = 0.01,
    ) -> None:
        if len(results) < 2:
            raise ValueError(
                "ComparisonAnalyzer requires results for at least 2 models."
            )
        self._results = results
        self._models = sorted(results.keys())
        self._elo = EloRating(models=self._models, k=elo_k)
        self._tie_threshold = tie_threshold

        # Build case-id -> model -> result index for efficient lookup
        self._case_index: dict[str, dict[str, EvalResult]] = defaultdict(dict)
        for model, res_list in results.items():
            for res in res_list:
                self._case_index[res.case.id][model] = res

    def _compute_stats(self, scores: list[float]) -> dict[str, float]:
        if not scores:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "p95": 0.0, "n": 0}
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        median = statistics.median(scores)
        p95 = _percentile(scores, 95)
        return {"mean": mean, "std": std, "median": median, "p95": p95, "n": len(scores)}

    def analyze(self) -> ComparisonResult:
        """Compute a full comparison result.

        Returns:
            ComparisonResult with win rates, Elo ratings, score summaries,
            and head-to-head matrices.
        """
        # Per-model aggregate score lists
        model_scores: dict[str, list[float]] = {m: [] for m in self._models}
        for model, res_list in self._results.items():
            for res in res_list:
                model_scores[model].append(res.aggregate_score)

        score_summary = {
            model: self._compute_stats(scores)
            for model, scores in model_scores.items()
        }

        # Head-to-head win rates and Elo updates
        head_to_head_wins: dict[str, dict[str, int]] = {
            m: {other: 0 for other in self._models if other != m}
            for m in self._models
        }
        head_to_head_total: dict[str, dict[str, int]] = {
            m: {other: 0 for other in self._models if other != m}
            for m in self._models
        }

        shared_cases = [
            case_id
            for case_id, model_map in self._case_index.items()
            if len(model_map) >= 2
        ]

        logger.info(
            "ComparisonAnalyzer: analyzing %d shared cases across %d models",
            len(shared_cases),
            len(self._models),
        )

        for case_id in shared_cases:
            model_map = self._case_index[case_id]
            available_models = [m for m in self._models if m in model_map]

            for i, model_a in enumerate(available_models):
                for model_b in available_models[i + 1:]:
                    score_a = model_map[model_a].aggregate_score
                    score_b = model_map[model_b].aggregate_score

                    head_to_head_total[model_a][model_b] += 1
                    head_to_head_total[model_b][model_a] += 1

                    diff = score_a - score_b
                    if abs(diff) < self._tie_threshold:
                        # Tie — update Elo with equal probability
                        self._elo.update(model_a, model_b, outcome=0.5)
                    elif diff > 0:
                        head_to_head_wins[model_a][model_b] += 1
                        self._elo.update(model_a, model_b, outcome=1.0)
                    else:
                        head_to_head_wins[model_b][model_a] += 1
                        self._elo.update(model_b, model_a, outcome=1.0)

        # Compute head-to-head win rates
        head_to_head: dict[str, dict[str, float]] = {}
        for model_a in self._models:
            head_to_head[model_a] = {}
            for model_b in self._models:
                if model_a == model_b:
                    continue
                total = head_to_head_total[model_a].get(model_b, 0)
                wins = head_to_head_wins[model_a].get(model_b, 0)
                head_to_head[model_a][model_b] = wins / total if total > 0 else 0.0

        # Overall win rates (average head-to-head win rate against all opponents)
        win_rates: dict[str, float] = {}
        for model in self._models:
            opponents = [m for m in self._models if m != model]
            if opponents:
                win_rates[model] = sum(head_to_head[model][opp] for opp in opponents) / len(opponents)
            else:
                win_rates[model] = 0.0

        elo_ratings = self._elo.get_ratings()

        result = ComparisonResult(
            models=self._models,
            win_rates=win_rates,
            elo_ratings=elo_ratings,
            score_summary=score_summary,
            head_to_head=head_to_head,
        )

        logger.info(
            "ComparisonAnalyzer: complete. Win rates: %s",
            {m: f"{v:.3f}" for m, v in win_rates.items()},
        )
        return result

    def per_case_deltas(self) -> list[dict[str, Any]]:
        """Compute per-case score differences between each pair of models.

        Returns:
            List of dicts with case_id, model scores, and deltas.
        """
        rows = []
        for case_id, model_map in sorted(self._case_index.items()):
            row: dict[str, Any] = {"case_id": case_id}
            for model in self._models:
                if model in model_map:
                    row[model] = model_map[model].aggregate_score
                else:
                    row[model] = None

            # Compute delta for each pair
            available = [m for m in self._models if row[m] is not None]
            for i, ma in enumerate(available):
                for mb in available[i + 1:]:
                    key = f"delta_{ma}_vs_{mb}"
                    row[key] = row[ma] - row[mb]  # type: ignore[operator]

            rows.append(row)
        return rows
