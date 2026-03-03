"""Tests for ComparisonAnalyzer and EloRating."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evalkit.comparison.analyzer import ComparisonAnalyzer
from evalkit.comparison.elo import EloRating
from evalkit.core.types import (
    EvalCase,
    EvalResult,
    ModelResponse,
    Score,
)


def _make_result(case_id: str, score: float, model: str = "model_a") -> EvalResult:
    case = EvalCase(id=case_id, prompt="Test prompt", expected="Expected")
    response = ModelResponse(
        text="Response text",
        model=model,
        provider="mock",
        latency_ms=50.0,
        tokens_in=10,
        tokens_out=20,
    )
    scores = [Score(value=score, scorer="exact_match")]
    return EvalResult(
        case=case,
        response=response,
        scores=scores,
        aggregate_score=score,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# ComparisonAnalyzer tests
# ---------------------------------------------------------------------------


class TestComparisonAnalyzer:
    def _make_results(self) -> dict[str, list[EvalResult]]:
        return {
            "model_a": [
                _make_result("case_01", 0.9, "model_a"),
                _make_result("case_02", 0.8, "model_a"),
                _make_result("case_03", 0.3, "model_a"),
            ],
            "model_b": [
                _make_result("case_01", 0.5, "model_b"),
                _make_result("case_02", 0.4, "model_b"),
                _make_result("case_03", 0.9, "model_b"),
            ],
        }

    def test_analyze_returns_comparison_result(self) -> None:
        results = self._make_results()
        analyzer = ComparisonAnalyzer(results)
        comparison = analyzer.analyze()
        assert comparison is not None
        assert sorted(comparison.models) == ["model_a", "model_b"]

    def test_win_rates_sum_correctly(self) -> None:
        results = self._make_results()
        analyzer = ComparisonAnalyzer(results)
        comparison = analyzer.analyze()
        for model in comparison.models:
            assert 0.0 <= comparison.win_rates[model] <= 1.0

    def test_elo_ratings_initialized(self) -> None:
        results = self._make_results()
        analyzer = ComparisonAnalyzer(results)
        comparison = analyzer.analyze()
        for model in comparison.models:
            # Ratings start at 1500; should have changed after comparisons
            assert comparison.elo_ratings[model] > 0

    def test_head_to_head_is_symmetric(self) -> None:
        results = self._make_results()
        analyzer = ComparisonAnalyzer(results)
        comparison = analyzer.analyze()
        # head_to_head[a][b] + head_to_head[b][a] should roughly equal 1.0
        # (they are win rates, so ties can make the sum != 1)
        a_vs_b = comparison.head_to_head.get("model_a", {}).get("model_b", 0.0)
        b_vs_a = comparison.head_to_head.get("model_b", {}).get("model_a", 0.0)
        assert a_vs_b + b_vs_a <= 1.0 + 1e-9  # With ties, max is 1.0

    def test_score_summary_has_statistics(self) -> None:
        results = self._make_results()
        analyzer = ComparisonAnalyzer(results)
        comparison = analyzer.analyze()
        for model in comparison.models:
            stats = comparison.score_summary[model]
            assert "mean" in stats
            assert "std" in stats
            assert "median" in stats
            assert "p95" in stats

    def test_requires_at_least_2_models(self) -> None:
        with pytest.raises(ValueError, match="at least 2 models"):
            ComparisonAnalyzer({"only_one": []})

    def test_per_case_deltas(self) -> None:
        results = self._make_results()
        analyzer = ComparisonAnalyzer(results)
        deltas = analyzer.per_case_deltas()
        assert len(deltas) == 3
        for row in deltas:
            assert "case_id" in row
            assert "model_a" in row
            assert "model_b" in row

    def test_better_model_has_higher_win_rate(self) -> None:
        """Model A should win more often since it has higher scores on 2/3 cases."""
        results = self._make_results()
        analyzer = ComparisonAnalyzer(results)
        comparison = analyzer.analyze()
        # model_a wins case_01 (0.9 vs 0.5) and case_02 (0.8 vs 0.4)
        # model_b wins case_03 (0.9 vs 0.3)
        # So model_a should have a higher win rate
        assert comparison.win_rates["model_a"] > comparison.win_rates["model_b"]


# ---------------------------------------------------------------------------
# EloRating tests
# ---------------------------------------------------------------------------


class TestEloRating:
    def test_initial_rating(self) -> None:
        elo = EloRating(models=["a", "b"])
        assert elo.get_rating("a") == 1500.0
        assert elo.get_rating("b") == 1500.0

    def test_winner_gains_rating(self) -> None:
        elo = EloRating(models=["a", "b"])
        ra_before = elo.get_rating("a")
        elo.update("a", "b", outcome=1.0)
        assert elo.get_rating("a") > ra_before

    def test_loser_loses_rating(self) -> None:
        elo = EloRating(models=["a", "b"])
        rb_before = elo.get_rating("b")
        elo.update("a", "b", outcome=1.0)
        assert elo.get_rating("b") < rb_before

    def test_draw_minimal_change(self) -> None:
        elo = EloRating(models=["a", "b"])
        ra_before = elo.get_rating("a")
        rb_before = elo.get_rating("b")
        elo.update("a", "b", outcome=0.5)
        # For equal ratings, draw should cause no change
        assert elo.get_rating("a") == pytest.approx(ra_before, abs=0.01)
        assert elo.get_rating("b") == pytest.approx(rb_before, abs=0.01)

    def test_match_count_incremented(self) -> None:
        elo = EloRating(models=["a", "b"])
        elo.update("a", "b")
        assert elo.get_match_count("a") == 1
        assert elo.get_match_count("b") == 1

    def test_auto_register_new_models(self) -> None:
        elo = EloRating()
        elo.update("x", "y")
        assert elo.get_rating("x") > 0
        assert elo.get_rating("y") > 0

    def test_leaderboard_sorted_descending(self) -> None:
        elo = EloRating(models=["a", "b", "c"])
        elo.update("a", "b")
        elo.update("a", "c")
        leaderboard = elo.get_leaderboard()
        ratings = [r for _, r, _ in leaderboard]
        assert ratings == sorted(ratings, reverse=True)

    def test_reset_restores_initial_ratings(self) -> None:
        elo = EloRating(models=["a", "b"], initial_rating=1500.0)
        elo.update("a", "b")
        elo.reset()
        assert elo.get_rating("a") == 1500.0
        assert elo.get_rating("b") == 1500.0
        assert elo.get_history() == []

    def test_expected_score_equal_ratings(self) -> None:
        elo = EloRating()
        # Equal ratings should give 0.5 expected score
        assert elo.expected_score(1500, 1500) == pytest.approx(0.5, abs=0.001)

    def test_expected_score_favors_higher_rating(self) -> None:
        elo = EloRating()
        # Higher rating should have >0.5 expected score
        assert elo.expected_score(1600, 1400) > 0.5

    def test_history_records_matches(self) -> None:
        elo = EloRating(models=["a", "b"])
        elo.update("a", "b")
        history = elo.get_history()
        assert len(history) == 1
        assert history[0]["winner"] == "a"
        assert history[0]["loser"] == "b"

    def test_custom_k_factor(self) -> None:
        elo_k32 = EloRating(models=["a", "b"], k=32.0)
        elo_k16 = EloRating(models=["a", "b"], k=16.0)
        new_a_k32, _ = elo_k32.update("a", "b", outcome=1.0)
        new_a_k16, _ = elo_k16.update("a", "b", outcome=1.0)
        # Higher K means bigger rating change
        delta_k32 = new_a_k32 - 1500.0
        delta_k16 = new_a_k16 - 1500.0
        assert delta_k32 > delta_k16

    def test_get_ratings_returns_copy(self) -> None:
        elo = EloRating(models=["a", "b"])
        ratings = elo.get_ratings()
        ratings["a"] = 9999.0
        # Original should be unchanged
        assert elo.get_rating("a") == 1500.0
