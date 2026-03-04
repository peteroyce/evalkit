"""Tests for JSON and SQLite storage backends."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from evalkit.core.types import (
    EvalCase,
    EvalResult,
    Judgment,
    ModelResponse,
    Score,
)
from evalkit.storage.backend import JSONFileBackend, SQLiteBackend


def _make_eval_result(case_id: str, score: float = 0.8) -> EvalResult:
    case = EvalCase(id=case_id, prompt="Test prompt?", expected="Expected answer.")
    response = ModelResponse(
        text="Model response text.",
        model="test-model",
        provider="mock",
        latency_ms=42.0,
        tokens_in=15,
        tokens_out=25,
        cost_usd=0.0001,
    )
    scores = [Score(value=score, scorer="exact_match", reasoning="Test reasoning.")]
    return EvalResult(
        case=case,
        response=response,
        scores=scores,
        aggregate_score=score,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _make_judgment(eval_id: str = "case_01") -> Judgment:
    return Judgment(
        eval_id=eval_id,
        preferred="model_a",
        models=["model_a", "model_b"],
        reason="Model A was more accurate.",
        judge="human",
    )


# ---------------------------------------------------------------------------
# JSONFileBackend tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestJSONFileBackend:
    async def test_save_and_get_run(self, tmp_storage: JSONFileBackend) -> None:
        results = [_make_eval_result("case_01"), _make_eval_result("case_02")]
        await tmp_storage.save_run(
            run_id="run-001",
            suite_name="test-suite",
            model="model-a",
            timestamp="2024-01-01T00:00:00+00:00",
            results=results,
            summary={"mean_score": 0.8},
        )
        data = await tmp_storage.get_run("run-001")
        assert data is not None
        assert data["id"] == "run-001"
        assert data["model"] == "model-a"
        assert data["suite_name"] == "test-suite"
        assert len(data["results"]) == 2

    async def test_get_run_not_found(self, tmp_storage: JSONFileBackend) -> None:
        data = await tmp_storage.get_run("nonexistent-run")
        assert data is None

    async def test_list_runs_empty(self, tmp_storage: JSONFileBackend) -> None:
        runs = await tmp_storage.list_runs()
        assert runs == []

    async def test_list_runs_returns_saved(self, tmp_storage: JSONFileBackend) -> None:
        results = [_make_eval_result("case_01")]
        await tmp_storage.save_run(
            run_id="run-list-001",
            suite_name="suite-x",
            model="m1",
            timestamp="2024-01-01T00:00:00+00:00",
            results=results,
        )
        await tmp_storage.save_run(
            run_id="run-list-002",
            suite_name="suite-y",
            model="m2",
            timestamp="2024-01-02T00:00:00+00:00",
            results=results,
        )
        runs = await tmp_storage.list_runs()
        assert len(runs) == 2

    async def test_list_runs_filter_by_suite(self, tmp_storage: JSONFileBackend) -> None:
        results = [_make_eval_result("c")]
        await tmp_storage.save_run("r1", "suite-a", "m1", "t", results)
        await tmp_storage.save_run("r2", "suite-b", "m2", "t", results)
        runs = await tmp_storage.list_runs(suite_name="suite-a")
        assert len(runs) == 1
        assert runs[0]["suite_name"] == "suite-a"

    async def test_list_runs_filter_by_model(self, tmp_storage: JSONFileBackend) -> None:
        results = [_make_eval_result("c")]
        await tmp_storage.save_run("r1", "suite", "alpha", "t", results)
        await tmp_storage.save_run("r2", "suite", "beta", "t", results)
        runs = await tmp_storage.list_runs(model="alpha")
        assert len(runs) == 1
        assert runs[0]["model"] == "alpha"

    async def test_delete_run(self, tmp_storage: JSONFileBackend) -> None:
        results = [_make_eval_result("c")]
        await tmp_storage.save_run("del-run", "s", "m", "t", results)
        assert await tmp_storage.get_run("del-run") is not None
        deleted = await tmp_storage.delete_run("del-run")
        assert deleted is True
        assert await tmp_storage.get_run("del-run") is None

    async def test_delete_nonexistent_run(self, tmp_storage: JSONFileBackend) -> None:
        deleted = await tmp_storage.delete_run("ghost-run")
        assert deleted is False

    async def test_save_and_get_judgment(self, tmp_storage: JSONFileBackend) -> None:
        j = _make_judgment("case_42")
        await tmp_storage.save_judgment(j)
        judgments = await tmp_storage.get_judgments(eval_id="case_42")
        assert len(judgments) == 1
        assert judgments[0].preferred == "model_a"
        assert judgments[0].eval_id == "case_42"

    async def test_get_judgments_filter_by_judge(self, tmp_storage: JSONFileBackend) -> None:
        j1 = Judgment(eval_id="c1", preferred="a", models=["a", "b"], judge="human")
        j2 = Judgment(eval_id="c2", preferred="b", models=["a", "b"], judge="gpt-4")
        await tmp_storage.save_judgment(j1)
        await tmp_storage.save_judgment(j2)
        human_judgments = await tmp_storage.get_judgments(judge="human")
        assert len(human_judgments) == 1
        assert human_judgments[0].judge == "human"

    async def test_judgment_roundtrip_preserves_fields(self, tmp_storage: JSONFileBackend) -> None:
        j = _make_judgment()
        await tmp_storage.save_judgment(j)
        judgments = await tmp_storage.get_judgments()
        assert len(judgments) == 1
        loaded = judgments[0]
        assert loaded.eval_id == j.eval_id
        assert loaded.preferred == j.preferred
        assert loaded.models == j.models
        assert loaded.reason == j.reason
        assert loaded.judge == j.judge

    async def test_results_preserved_in_json(self, tmp_storage: JSONFileBackend) -> None:
        result = _make_eval_result("case_99", score=0.75)
        await tmp_storage.save_run("r-json", "s", "m", "t", [result])
        data = await tmp_storage.get_run("r-json")
        assert len(data["results"]) == 1
        assert data["results"][0]["aggregate_score"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# SQLiteBackend tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSQLiteBackend:
    @pytest.fixture
    def sqlite_backend(self, tmp_path: Path) -> SQLiteBackend:
        return SQLiteBackend(tmp_path / "test.db")

    async def test_save_and_get_run(self, sqlite_backend: SQLiteBackend) -> None:
        results = [_make_eval_result("case_01")]
        await sqlite_backend.save_run(
            run_id="sqlite-run-1",
            suite_name="suite-sqlite",
            model="sqlite-model",
            timestamp="2024-01-01T00:00:00+00:00",
            results=results,
            summary={"mean_score": 0.8},
        )
        data = await sqlite_backend.get_run("sqlite-run-1")
        assert data is not None
        assert data["id"] == "sqlite-run-1"
        assert data["suite_name"] == "suite-sqlite"
        assert data["model"] == "sqlite-model"
        assert len(data["results"]) == 1

    async def test_get_run_not_found(self, sqlite_backend: SQLiteBackend) -> None:
        data = await sqlite_backend.get_run("does-not-exist")
        assert data is None

    async def test_list_runs(self, sqlite_backend: SQLiteBackend) -> None:
        results = [_make_eval_result("c")]
        await sqlite_backend.save_run("sr1", "suite", "m1", "t", results)
        await sqlite_backend.save_run("sr2", "suite", "m2", "t", results)
        runs = await sqlite_backend.list_runs()
        assert len(runs) == 2

    async def test_delete_run(self, sqlite_backend: SQLiteBackend) -> None:
        results = [_make_eval_result("c")]
        await sqlite_backend.save_run("to-delete", "s", "m", "t", results)
        deleted = await sqlite_backend.delete_run("to-delete")
        assert deleted is True
        assert await sqlite_backend.get_run("to-delete") is None

    async def test_save_judgment(self, sqlite_backend: SQLiteBackend) -> None:
        j = _make_judgment("case_sqlite_1")
        await sqlite_backend.save_judgment(j)
        judgments = await sqlite_backend.get_judgments(eval_id="case_sqlite_1")
        assert len(judgments) == 1
        assert judgments[0].preferred == "model_a"

    async def test_multiple_results_stored(self, sqlite_backend: SQLiteBackend) -> None:
        results = [_make_eval_result(f"case_{i}", score=i / 10.0) for i in range(5)]
        await sqlite_backend.save_run("multi-run", "s", "m", "t", results)
        data = await sqlite_backend.get_run("multi-run")
        assert len(data["results"]) == 5

    async def test_close_disposes_engine(self, sqlite_backend: SQLiteBackend) -> None:
        results = [_make_eval_result("c")]
        await sqlite_backend.save_run("close-test", "s", "m", "t", results)
        await sqlite_backend.close()  # Should not raise
