"""Tests for FastAPI routes using httpx async test client."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from evalkit.api.app import create_app
from evalkit.core.types import (
    EvalCase,
    EvalResult,
    Judgment,
    ModelResponse,
    Score,
)
from evalkit.storage.backend import JSONFileBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path) -> JSONFileBackend:
    return JSONFileBackend(tmp_path / "api_storage")


@pytest.fixture
def app(storage: JSONFileBackend):
    return create_app(storage=storage)


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


def _make_eval_result(case_id: str, score: float = 0.75) -> EvalResult:
    case = EvalCase(id=case_id, prompt="Test prompt?", expected="Expected answer.")
    response = ModelResponse(
        text="Model response.",
        model="test-model",
        provider="mock",
        latency_ms=30.0,
        tokens_in=10,
        tokens_out=15,
    )
    scores = [Score(value=score, scorer="exact_match")]
    return EvalResult(
        case=case,
        response=response,
        scores=scores,
        aggregate_score=score,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


async def _seed_run(
    storage: JSONFileBackend,
    run_id: str,
    model: str = "test-model",
    suite_name: str = "test-suite",
    n_cases: int = 2,
) -> list[EvalResult]:
    results = [_make_eval_result(f"case_{i:02d}") for i in range(n_cases)]
    await storage.save_run(
        run_id=run_id,
        suite_name=suite_name,
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat(),
        results=results,
        summary={"mean_score": 0.75},
    )
    return results


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_returns_ok(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    async def test_root_redirect_info(self, client: AsyncClient) -> None:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data


# ---------------------------------------------------------------------------
# Runs endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunsEndpoints:
    async def test_list_runs_empty(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/runs")
        assert response.status_code == 200
        assert response.json() == []

    async def test_list_runs_returns_saved(
        self, client: AsyncClient, storage: JSONFileBackend
    ) -> None:
        await _seed_run(storage, "run-001", model="gpt-4")
        await _seed_run(storage, "run-002", model="claude-3")
        response = await client.get("/api/v1/runs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    async def test_get_run_by_id(
        self, client: AsyncClient, storage: JSONFileBackend
    ) -> None:
        await _seed_run(storage, "get-run-001")
        response = await client.get("/api/v1/runs/get-run-001")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "get-run-001"
        assert len(data["results"]) == 2

    async def test_get_run_not_found(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/runs/nonexistent-run-id")
        assert response.status_code == 404

    async def test_list_runs_filter_by_model(
        self, client: AsyncClient, storage: JSONFileBackend
    ) -> None:
        await _seed_run(storage, "r1", model="alpha")
        await _seed_run(storage, "r2", model="beta")
        response = await client.get("/api/v1/runs", params={"model": "alpha"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["model"] == "alpha"

    async def test_delete_run(
        self, client: AsyncClient, storage: JSONFileBackend
    ) -> None:
        await _seed_run(storage, "del-run-001")
        response = await client.delete("/api/v1/runs/del-run-001")
        assert response.status_code == 200
        # Verify it's gone
        get_response = await client.get("/api/v1/runs/del-run-001")
        assert get_response.status_code == 404

    async def test_delete_nonexistent_run(self, client: AsyncClient) -> None:
        response = await client.delete("/api/v1/runs/ghost")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Compare endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCompareEndpoint:
    async def test_compare_two_runs(
        self, client: AsyncClient, storage: JSONFileBackend
    ) -> None:
        # Seed two runs with the same case IDs
        results_a = [_make_eval_result("case_00", 0.9), _make_eval_result("case_01", 0.8)]
        results_b = [_make_eval_result("case_00", 0.4), _make_eval_result("case_01", 0.5)]

        await storage.save_run("cmp-run-a", "suite", "model-a", "t", results_a)
        await storage.save_run("cmp-run-b", "suite", "model-b", "t", results_b)

        response = await client.get(
            "/api/v1/compare",
            params={"run_ids": ["cmp-run-a", "cmp-run-b"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "win_rates" in data
        assert "elo_ratings" in data

    async def test_compare_missing_run(self, client: AsyncClient) -> None:
        response = await client.get(
            "/api/v1/compare",
            params={"run_ids": ["ghost-1", "ghost-2"]},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Judge endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestJudgeEndpoint:
    async def test_submit_judgment(self, client: AsyncClient) -> None:
        payload = {
            "eval_id": "case_01",
            "preferred": "model-a",
            "models": ["model-a", "model-b"],
            "reason": "Model A was clearer.",
            "judge": "human",
        }
        response = await client.post("/api/v1/judge", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["eval_id"] == "case_01"
        assert data["preferred"] == "model-a"

    async def test_list_judgments_empty(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/judge")
        assert response.status_code == 200
        assert response.json() == []

    async def test_list_judgments_after_submit(self, client: AsyncClient) -> None:
        payload = {
            "eval_id": "case_02",
            "preferred": "model-b",
            "models": ["model-a", "model-b"],
            "judge": "human",
        }
        await client.post("/api/v1/judge", json=payload)
        response = await client.get("/api/v1/judge")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

    async def test_filter_judgments_by_eval_id(self, client: AsyncClient) -> None:
        for i in range(3):
            payload = {
                "eval_id": f"case_{i:02d}",
                "preferred": "model-a",
                "models": ["model-a", "model-b"],
                "judge": "human",
            }
            await client.post("/api/v1/judge", json=payload)

        response = await client.get("/api/v1/judge", params={"eval_id": "case_01"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["eval_id"] == "case_01"


# ---------------------------------------------------------------------------
# Evaluate endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEvaluateEndpoint:
    async def test_evaluate_with_mock_provider(
        self, client: AsyncClient, storage: JSONFileBackend, tmp_path: Path
    ) -> None:
        # Write a minimal YAML suite to disk
        suite_path = tmp_path / "test_suite.yaml"
        suite_path.write_text(
            "name: api-test\ndescription: API test\ncases:\n  - id: c1\n    prompt: 'Hello?'\n",
            encoding="utf-8",
        )

        payload = {
            "suite_path": str(suite_path),
            "providers": [
                {
                    "name": "mock-provider",
                    "type": "mock",
                    "model": "mock-v1",
                }
            ],
            "scorers": [{"type": "exact_match", "weight": 1.0}],
        }
        response = await client.post("/api/v1/evaluate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "run_ids" in data
        assert "mock-provider" in data["run_ids"]

    async def test_evaluate_missing_suite_returns_422(self, client: AsyncClient) -> None:
        payload = {
            "providers": [{"name": "m", "type": "mock", "model": "m"}],
        }
        response = await client.post("/api/v1/evaluate", json=payload)
        assert response.status_code == 422

    async def test_evaluate_builtin_dataset(self, client: AsyncClient, tmp_path: Path) -> None:
        payload = {
            "builtin_dataset": "reasoning",
            "providers": [
                {"name": "mock", "type": "mock", "model": "test-model"}
            ],
            "scorers": [{"type": "contains", "weight": 1.0}],
        }
        response = await client.post("/api/v1/evaluate", json=payload)
        # 200 if dataset files exist; 404 if not (CI without dataset files)
        assert response.status_code in {200, 404}
