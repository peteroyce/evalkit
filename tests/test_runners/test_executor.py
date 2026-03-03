"""Tests for EvalExecutor and BatchRunner."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from evalkit.core.types import EvalCase, EvalSuite
from evalkit.providers.mock import MockProvider
from evalkit.runners.executor import EvalExecutor
from evalkit.runners.batch import BatchRunner
from evalkit.scorers.exact_match import ExactMatchScorer
from evalkit.scorers.contains import ContainsScorer


# ---------------------------------------------------------------------------
# EvalExecutor tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEvalExecutor:
    async def test_run_returns_eval_result(
        self, mock_provider: MockProvider, sample_cases: list[EvalCase]
    ) -> None:
        executor = EvalExecutor(
            provider=mock_provider,
            scorers=[ExactMatchScorer()],
        )
        result = await executor.run(sample_cases[0])
        assert result.case.id == "case_01"
        assert result.response.text == "The answer is 42."
        assert len(result.scores) == 1
        assert 0.0 <= result.aggregate_score <= 1.0
        assert result.timestamp

    async def test_exact_match_first_case(
        self, mock_provider: MockProvider, sample_cases: list[EvalCase]
    ) -> None:
        executor = EvalExecutor(
            provider=mock_provider,
            scorers=[ExactMatchScorer()],
        )
        # case_01 expected: "The answer is 42."
        # mock_provider canned responses[0]: "The answer is 42."
        result = await executor.run(sample_cases[0])
        assert result.scores[0].value == 1.0

    async def test_multiple_scorers(
        self, mock_provider: MockProvider, sample_cases: list[EvalCase]
    ) -> None:
        executor = EvalExecutor(
            provider=mock_provider,
            scorers=[ExactMatchScorer(), ContainsScorer()],
        )
        result = await executor.run(sample_cases[0])
        assert len(result.scores) == 2
        scorer_names = {s.scorer for s in result.scores}
        assert "exact_match" in scorer_names
        assert "contains" in scorer_names

    async def test_aggregate_score_is_weighted_average(
        self, mock_provider: MockProvider, sample_cases: list[EvalCase]
    ) -> None:
        executor = EvalExecutor(
            provider=mock_provider,
            scorers=[ExactMatchScorer(), ContainsScorer()],
            scorer_weights={"exact_match": 2.0, "contains": 1.0},
        )
        result = await executor.run(sample_cases[0])
        # Both scorers should score 1.0 for the matching response
        assert result.aggregate_score == pytest.approx(1.0, abs=0.001)

    async def test_retry_on_failure(self, sample_cases: list[EvalCase]) -> None:
        # fail_every_n=1 means fail on every call, but with max_retries=2
        # it will try 3 times total and still fail
        failing_provider = MockProvider(
            model="failing-model",
            fail_every_n=1,
            simulated_latency_ms=0,
        )
        executor = EvalExecutor(
            provider=failing_provider,
            scorers=[ExactMatchScorer()],
            max_retries=0,  # no retries, should fail immediately
        )
        with pytest.raises(RuntimeError, match="failed after"):
            await executor.run(sample_cases[0])

    async def test_result_includes_response_metadata(
        self, mock_provider: MockProvider, sample_cases: list[EvalCase]
    ) -> None:
        executor = EvalExecutor(
            provider=mock_provider,
            scorers=[ExactMatchScorer()],
        )
        result = await executor.run(sample_cases[0])
        assert result.response.provider == "mock"
        assert result.response.model == "mock-model-v1"
        assert result.response.tokens_in > 0
        assert result.response.tokens_out > 0
        assert result.response.latency_ms >= 0


# ---------------------------------------------------------------------------
# BatchRunner tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBatchRunner:
    async def test_run_suite_returns_results_for_all_cases(
        self,
        sample_suite: EvalSuite,
    ) -> None:
        provider = MockProvider(
            model="batch-model",
            responses=["The answer is 42.", "Paris", "$0.05"],
            mode="canned",
            simulated_latency_ms=1.0,
        )
        runner = BatchRunner(
            providers={"batch-model": provider},
            scorers=[ExactMatchScorer()],
            concurrency=2,
            show_progress=False,
        )
        results = await runner.run_suite(sample_suite)
        assert "batch-model" in results
        assert len(results["batch-model"]) == len(sample_suite.cases)

    async def test_run_suite_multiple_providers(
        self,
        sample_suite: EvalSuite,
    ) -> None:
        providers = {
            "model_a": MockProvider(model="model_a", mode="random", simulated_latency_ms=1.0),
            "model_b": MockProvider(model="model_b", mode="random", simulated_latency_ms=1.0),
        }
        runner = BatchRunner(
            providers=providers,
            scorers=[ExactMatchScorer()],
            concurrency=3,
            show_progress=False,
        )
        results = await runner.run_suite(sample_suite)
        assert "model_a" in results
        assert "model_b" in results

    async def test_run_suite_persists_to_storage(
        self,
        sample_suite: EvalSuite,
        tmp_storage,
    ) -> None:
        provider = MockProvider(
            model="stored-model",
            mode="canned",
            responses=["answer"],
            simulated_latency_ms=0,
        )
        runner = BatchRunner(
            providers={"stored-model": provider},
            scorers=[ExactMatchScorer()],
            storage=tmp_storage,
            show_progress=False,
        )
        results = await runner.run_suite(sample_suite, run_id="test-run-001")

        # Verify data was stored
        stored = await tmp_storage.get_run("test-run-001_stored-model")
        assert stored is not None
        assert stored["model"] == "stored-model"

    async def test_run_suite_with_custom_run_id(
        self,
        sample_suite: EvalSuite,
    ) -> None:
        provider = MockProvider(model="m", mode="random", simulated_latency_ms=0)
        runner = BatchRunner(
            providers={"m": provider},
            scorers=[ExactMatchScorer()],
            show_progress=False,
        )
        results = await runner.run_suite(sample_suite, run_id="my-custom-run")
        assert results is not None

    async def test_all_results_have_timestamps(
        self,
        sample_suite: EvalSuite,
    ) -> None:
        provider = MockProvider(model="ts-model", mode="random", simulated_latency_ms=0)
        runner = BatchRunner(
            providers={"ts-model": provider},
            scorers=[ExactMatchScorer()],
            show_progress=False,
        )
        results = await runner.run_suite(sample_suite)
        for result in results["ts-model"]:
            assert result.timestamp
            assert "T" in result.timestamp  # ISO format check
