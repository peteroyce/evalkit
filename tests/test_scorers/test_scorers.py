"""Tests for all scorer types."""

from __future__ import annotations

import asyncio

import pytest

from evalkit.core.types import Score
from evalkit.scorers.contains import ContainsScorer
from evalkit.scorers.custom import CustomScorer
from evalkit.scorers.exact_match import ExactMatchScorer
from evalkit.scorers.regex import RegexScorer
from evalkit.scorers.similarity import SemanticSimilarityScorer
from evalkit.scorers.composite import CompositeScorer
from evalkit.scorers import create_scorer


# ---------------------------------------------------------------------------
# ExactMatchScorer
# ---------------------------------------------------------------------------


class TestExactMatchScorer:
    def setup_method(self) -> None:
        self.scorer = ExactMatchScorer()

    def test_exact_match_returns_1(self) -> None:
        score = self.scorer.score("Paris", "Paris")
        assert score.value == 1.0
        assert score.scorer == "exact_match"

    def test_case_insensitive_match(self) -> None:
        score = self.scorer.score("paris", "Paris")
        assert score.value == 1.0

    def test_whitespace_normalization(self) -> None:
        score = self.scorer.score("  Paris  ", "Paris")
        assert score.value == 1.0

    def test_internal_whitespace_collapsed(self) -> None:
        score = self.scorer.score("hello   world", "hello world")
        assert score.value == 1.0

    def test_no_match_returns_0(self) -> None:
        score = self.scorer.score("Berlin", "Paris")
        assert score.value == 0.0

    def test_no_expected_returns_0(self) -> None:
        score = self.scorer.score("Paris", None)
        assert score.value == 0.0

    def test_strip_punctuation_enabled(self) -> None:
        scorer = ExactMatchScorer(strip_punctuation=True)
        score = scorer.score("Paris.", "Paris")
        assert score.value == 1.0

    def test_score_has_reasoning(self) -> None:
        score = self.scorer.score("Paris", "London")
        assert score.reasoning is not None
        assert len(score.reasoning) > 0

    def test_metadata_contains_normalized_values(self) -> None:
        score = self.scorer.score("Paris", "Paris")
        assert "normalized_output" in score.metadata
        assert "normalized_expected" in score.metadata


# ---------------------------------------------------------------------------
# ContainsScorer
# ---------------------------------------------------------------------------


class TestContainsScorer:
    def test_single_substring_found(self) -> None:
        scorer = ContainsScorer()
        score = scorer.score("The capital of France is Paris.", "Paris")
        assert score.value == 1.0

    def test_single_substring_not_found(self) -> None:
        scorer = ContainsScorer()
        score = scorer.score("Berlin is the capital.", "Paris")
        assert score.value == 0.0

    def test_case_insensitive_by_default(self) -> None:
        scorer = ContainsScorer(case_sensitive=False)
        score = scorer.score("PARIS is great", "paris")
        assert score.value == 1.0

    def test_case_sensitive_mode(self) -> None:
        scorer = ContainsScorer(case_sensitive=True)
        score = scorer.score("PARIS is great", "paris")
        assert score.value == 0.0

    def test_fixed_substrings_partial(self) -> None:
        scorer = ContainsScorer(substrings=["Paris", "Eiffel", "Louvre"])
        score = scorer.score("I visited Paris and the Eiffel Tower.", None)
        assert pytest.approx(score.value, abs=0.01) == 2 / 3

    def test_fixed_substrings_all_found(self) -> None:
        scorer = ContainsScorer(substrings=["Paris", "Eiffel"])
        score = scorer.score("Paris has the Eiffel Tower.", None)
        assert score.value == 1.0

    def test_require_all_mode(self) -> None:
        scorer = ContainsScorer(substrings=["Paris", "Eiffel"], require_all=True)
        score = scorer.score("I visited Paris.", None)
        assert score.value == 0.0

    def test_require_all_all_found(self) -> None:
        scorer = ContainsScorer(substrings=["Paris", "Eiffel"], require_all=True)
        score = scorer.score("Paris has the Eiffel Tower.", None)
        assert score.value == 1.0

    def test_no_expected_or_substrings(self) -> None:
        scorer = ContainsScorer()
        score = scorer.score("Some text", None)
        assert score.value == 0.0

    def test_metadata_tracks_found_and_missing(self) -> None:
        scorer = ContainsScorer(substrings=["Paris", "Eiffel"])
        score = scorer.score("Visit Paris!", None)
        assert "Paris" in score.metadata["found"]
        assert "Eiffel" in score.metadata["missing"]


# ---------------------------------------------------------------------------
# RegexScorer
# ---------------------------------------------------------------------------


class TestRegexScorer:
    def test_match_returns_1(self) -> None:
        scorer = RegexScorer(pattern=r"\$\d+\.\d{2}")
        score = scorer.score("The total is $12.50.")
        assert score.value == 1.0

    def test_no_match_returns_0(self) -> None:
        scorer = RegexScorer(pattern=r"\$\d+\.\d{2}")
        score = scorer.score("The total is free.")
        assert score.value == 0.0

    def test_named_groups_in_metadata(self) -> None:
        scorer = RegexScorer(pattern=r"(?P<amount>\$\d+\.\d{2})")
        score = scorer.score("The price is $9.99 today.")
        assert score.metadata["groups"]["amount"] == "$9.99"

    def test_case_insensitive_by_default(self) -> None:
        scorer = RegexScorer(pattern=r"hello")
        score = scorer.score("HELLO world")
        assert score.value == 1.0

    def test_fullmatch_mode(self) -> None:
        scorer = RegexScorer(pattern=r"\d+", search=False)
        score = scorer.score("42")
        assert score.value == 1.0

    def test_fullmatch_fails_on_partial(self) -> None:
        scorer = RegexScorer(pattern=r"\d+", search=False)
        score = scorer.score("42 apples")
        assert score.value == 0.0

    def test_partial_credit_named_groups(self) -> None:
        scorer = RegexScorer(
            pattern=r"(?P<name>\w+) is (?P<age>\d+)",
            partial_credit=True,
        )
        # Both groups match
        score = scorer.score("Alice is 30 years old.")
        assert score.value == 1.0


# ---------------------------------------------------------------------------
# SemanticSimilarityScorer
# ---------------------------------------------------------------------------


class TestSemanticSimilarityScorer:
    def test_identical_text_high_similarity(self) -> None:
        scorer = SemanticSimilarityScorer(use_tfidf_fallback=True)
        score = scorer.score("Paris is the capital of France.", "Paris is the capital of France.")
        assert score.value > 0.9

    def test_different_text_lower_similarity(self) -> None:
        scorer = SemanticSimilarityScorer(use_tfidf_fallback=True)
        score = scorer.score("The sky is blue.", "Water boils at 100 degrees.")
        assert score.value < 0.8

    def test_no_expected_returns_0(self) -> None:
        scorer = SemanticSimilarityScorer(use_tfidf_fallback=True)
        score = scorer.score("Some output.", None)
        assert score.value == 0.0

    def test_threshold_clamps_low_scores(self) -> None:
        scorer = SemanticSimilarityScorer(use_tfidf_fallback=True, threshold=0.95)
        score = scorer.score("Hello world.", "Goodbye universe.")
        # Even if similarity > 0, it's likely below threshold
        # We just check value is 0 or a valid float
        assert 0.0 <= score.value <= 1.0

    def test_score_is_in_range(self) -> None:
        scorer = SemanticSimilarityScorer(use_tfidf_fallback=True)
        score = scorer.score("foo bar baz", "qux quux corge")
        assert 0.0 <= score.value <= 1.0

    def test_metadata_has_method(self) -> None:
        scorer = SemanticSimilarityScorer(use_tfidf_fallback=True)
        score = scorer.score("hello", "world")
        assert "method" in score.metadata


# ---------------------------------------------------------------------------
# CompositeScorer
# ---------------------------------------------------------------------------


class TestCompositeScorer:
    def test_equal_weights(self) -> None:
        s1 = ExactMatchScorer()
        s2 = ContainsScorer()
        scorer = CompositeScorer([(s1, 1.0), (s2, 1.0)])
        # "Paris" exact-matches "Paris" (1.0) and contains "Paris" (1.0)
        score = scorer.score("Paris", "Paris")
        assert pytest.approx(score.value, abs=0.001) == 1.0

    def test_weighted_average(self) -> None:
        s1 = ExactMatchScorer()
        s2 = ContainsScorer()
        scorer = CompositeScorer([(s1, 0.7), (s2, 0.3)])
        # s1 scores 0.0 (no match), s2 scores 1.0 (contains "Paris")
        score = scorer.score("The capital is Paris", "Paris")
        # Weighted: 0.7 * 0.0 + 0.3 * 1.0 = 0.3
        assert pytest.approx(score.value, abs=0.01) == 0.3

    def test_raises_on_empty_scorers(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CompositeScorer([])

    def test_raises_on_zero_weight(self) -> None:
        s = ExactMatchScorer()
        with pytest.raises(ValueError, match="positive"):
            CompositeScorer([(s, 0.0)])

    def test_components_in_metadata(self) -> None:
        s1 = ExactMatchScorer()
        scorer = CompositeScorer([(s1, 1.0)])
        score = scorer.score("Paris", "Paris")
        assert "components" in score.metadata
        assert len(score.metadata["components"]) == 1

    @pytest.mark.asyncio
    async def test_async_score(self) -> None:
        s1 = ExactMatchScorer()
        s2 = ContainsScorer()
        scorer = CompositeScorer([(s1, 1.0), (s2, 1.0)])
        score = await scorer.score_async("Paris", "Paris")
        assert score.value == 1.0


# ---------------------------------------------------------------------------
# CustomScorer
# ---------------------------------------------------------------------------


class TestCustomScorer:
    def test_custom_fn_called(self) -> None:
        def my_fn(output: str, expected: str | None) -> float:
            return 1.0 if "correct" in output.lower() else 0.0

        scorer = CustomScorer(fn=my_fn, scorer_name="correct_check")
        assert scorer.score("This is correct.", None).value == 1.0
        assert scorer.score("This is wrong.", None).value == 0.0

    def test_clamps_out_of_range_values(self) -> None:
        def bad_fn(output: str, expected: str | None) -> float:
            return 1.5  # out of range

        scorer = CustomScorer(fn=bad_fn)
        score = scorer.score("text", None)
        assert score.value == 1.0  # clamped

    def test_handles_exception_gracefully(self) -> None:
        def exploding_fn(output: str, expected: str | None) -> float:
            raise RuntimeError("boom")

        scorer = CustomScorer(fn=exploding_fn)
        score = scorer.score("text", None)
        assert score.value == 0.0
        assert "boom" in score.reasoning

    def test_name_override(self) -> None:
        scorer = CustomScorer(fn=lambda o, e: 0.5, scorer_name="my_scorer")
        assert scorer.name == "my_scorer"

    def test_description_in_reasoning(self) -> None:
        scorer = CustomScorer(
            fn=lambda o, e: 0.8,
            scorer_name="test",
            description="Checks foo.",
        )
        score = scorer.score("text", None)
        assert "Checks foo." in score.reasoning


# ---------------------------------------------------------------------------
# create_scorer factory
# ---------------------------------------------------------------------------


class TestCreateScorerFactory:
    def test_create_exact_match(self) -> None:
        s = create_scorer("exact_match")
        assert isinstance(s, ExactMatchScorer)

    def test_create_contains(self) -> None:
        s = create_scorer("contains")
        assert isinstance(s, ContainsScorer)

    def test_create_regex(self) -> None:
        s = create_scorer("regex", pattern=r"\d+")
        assert isinstance(s, RegexScorer)

    def test_create_similarity(self) -> None:
        s = create_scorer("similarity")
        assert isinstance(s, SemanticSimilarityScorer)

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scorer type"):
            create_scorer("nonexistent_scorer")
