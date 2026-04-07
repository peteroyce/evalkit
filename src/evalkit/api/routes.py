"""REST API routes for evalkit."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from evalkit import __version__
from evalkit.api.schemas import (
    CompareRequest,
    ComparisonResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    JudgmentRequest,
    JudgmentResponse,
    RunDetailResponse,
    RunSummaryResponse,
    EvalResultResponse,
    ModelResponseResponse,
    ScoreResponse,
)
from evalkit.comparison.analyzer import ComparisonAnalyzer
from evalkit.core.types import Judgment

logger = logging.getLogger(__name__)

router = APIRouter()


def _validate_suite_path(user_path: str) -> Path:
    """Validate a user-supplied suite file path to prevent directory traversal.

    Resolves the path and, when the ``EVALKIT_DATASETS_DIR`` environment
    variable is set, ensures the resolved path stays within that directory.
    """
    resolved = Path(user_path).resolve()

    base_dir = os.environ.get("EVALKIT_DATASETS_DIR")
    if base_dir:
        allowed = Path(base_dir).resolve()
        if not str(resolved).startswith(str(allowed) + os.sep) and resolved != allowed:
            raise ValueError(
                f"Suite path '{user_path}' is outside the allowed datasets directory"
            )

    # Even without a configured base dir, reject paths containing traversal
    # components in the raw input to block naive exploitation attempts.
    if ".." in Path(user_path).parts:
        raise ValueError(
            f"Suite path '{user_path}' contains disallowed '..' traversal"
        )

    return resolved


def _get_storage(request: Request) -> Any:
    return request.app.state.storage


def _get_runner_config(request: Request) -> Any:
    return request.app.state.runner_config


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Liveness probe endpoint."""
    return HealthResponse(status="ok", version=__version__)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


@router.post("/evaluate", response_model=EvaluateResponse, tags=["evaluation"])
async def evaluate(body: EvaluateRequest, request: Request) -> EvaluateResponse:
    """Run an evaluation suite against one or more models.

    Accepts an inline suite path or builtin dataset name, a list of provider
    configs, and scorer configs. Runs asynchronously and returns run IDs.
    """
    from evalkit.datasets.loader import DatasetLoader
    from evalkit.datasets.builtin import load_builtin_dataset
    from evalkit.providers import create_provider
    from evalkit.scorers import create_scorer
    from evalkit.runners.batch import BatchRunner

    storage = _get_storage(request)

    # Load the dataset
    if body.builtin_dataset:
        try:
            suite = load_builtin_dataset(body.builtin_dataset)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))
    elif body.suite_path:
        try:
            validated_path = _validate_suite_path(body.suite_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        try:
            loader = DatasetLoader(tag_filter=body.tag_filter or None)
            suite = loader.load(str(validated_path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
    else:
        raise HTTPException(
            status_code=422,
            detail="Either 'suite_path' or 'builtin_dataset' must be provided.",
        )

    if body.tag_filter and not body.builtin_dataset:
        suite = suite.filter_by_tags(body.tag_filter)

    # Build providers
    providers: dict[str, Any] = {}
    for pconf in body.providers:
        try:
            if pconf.type == "mock":
                kwargs: dict[str, Any] = {"model": pconf.model}
            else:
                kwargs = {
                    "model": pconf.model,
                    "temperature": pconf.temperature,
                    "max_tokens": pconf.max_tokens,
                }
                if pconf.api_key:
                    kwargs["api_key"] = pconf.api_key
                if pconf.base_url and pconf.type == "openai":
                    kwargs["base_url"] = pconf.base_url
            providers[pconf.name] = create_provider(pconf.type, **kwargs)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to create provider '{pconf.name}': {exc}",
            )

    if not providers:
        raise HTTPException(status_code=422, detail="At least one provider is required.")

    # Build scorers
    scorers = []
    for sconf in body.scorers:
        try:
            scorers.append(create_scorer(sconf.type, **sconf.params))
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to create scorer '{sconf.type}': {exc}",
            )

    if not scorers:
        from evalkit.scorers import ExactMatchScorer
        scorers = [ExactMatchScorer()]

    run_id = body.run_id or str(uuid.uuid4())

    runner = BatchRunner(
        providers=providers,
        scorers=scorers,
        concurrency=body.concurrency,
        storage=storage,
        show_progress=False,
    )

    try:
        all_results = await runner.run_suite(suite, run_id=run_id)
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")

    # Map model label to run_id
    run_ids = {model: f"{run_id}_{model}" for model in all_results}
    return EvaluateResponse(run_ids=run_ids)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@router.get("/runs", response_model=list[RunSummaryResponse], tags=["runs"])
async def list_runs(
    request: Request,
    suite_name: str | None = Query(default=None),
    model: str | None = Query(default=None),
    limit: int = Query(default=50, gt=0, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[RunSummaryResponse]:
    """List eval runs with optional filtering."""
    storage = _get_storage(request)
    if storage is None:
        raise HTTPException(status_code=503, detail="No storage backend configured.")

    runs = await storage.list_runs(suite_name=suite_name, model=model, limit=limit, offset=offset)
    return [
        RunSummaryResponse(
            run_id=r["id"],
            suite_name=r.get("suite_name"),
            model=r.get("model"),
            timestamp=r.get("timestamp"),
            n_cases=r.get("n_results", 0),
            mean_score=r.get("summary", {}).get("mean_score", 0.0),
        )
        for r in runs
    ]


@router.get("/runs/{run_id}", response_model=RunDetailResponse, tags=["runs"])
async def get_run(run_id: str, request: Request) -> RunDetailResponse:
    """Get detailed results for a specific eval run."""
    storage = _get_storage(request)
    if storage is None:
        raise HTTPException(status_code=503, detail="No storage backend configured.")

    data = await storage.get_run(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    result_responses = []
    for r in data.get("results", []):
        resp_data = r.get("response", {})
        scores_data = r.get("scores", [])
        result_responses.append(
            EvalResultResponse(
                case_id=r.get("case_id", ""),
                prompt=resp_data.get("text", ""),  # fallback
                expected=None,
                response=ModelResponseResponse(
                    text=resp_data.get("text", ""),
                    model=resp_data.get("model", ""),
                    provider=resp_data.get("provider", ""),
                    latency_ms=resp_data.get("latency_ms", 0.0),
                    tokens_in=resp_data.get("tokens_in", 0),
                    tokens_out=resp_data.get("tokens_out", 0),
                    cost_usd=resp_data.get("cost_usd"),
                ),
                scores=[
                    ScoreResponse(
                        value=s.get("value", 0.0),
                        scorer=s.get("scorer", ""),
                        reasoning=s.get("reasoning"),
                        metadata=s.get("metadata", {}),
                    )
                    for s in scores_data
                ],
                aggregate_score=r.get("aggregate_score", 0.0),
                timestamp=r.get("timestamp", ""),
            )
        )

    return RunDetailResponse(
        run_id=data["id"],
        suite_name=data.get("suite_name"),
        model=data.get("model"),
        timestamp=data.get("timestamp"),
        summary=data.get("summary", {}),
        results=result_responses,
    )


@router.delete("/runs/{run_id}", tags=["runs"])
async def delete_run(run_id: str, request: Request) -> dict[str, str]:
    """Delete an eval run."""
    storage = _get_storage(request)
    if storage is None:
        raise HTTPException(status_code=503, detail="No storage backend configured.")

    deleted = await storage.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return {"message": f"Run '{run_id}' deleted."}


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


@router.get("/compare", response_model=ComparisonResponse, tags=["comparison"])
async def compare_runs(
    request: Request,
    run_ids: list[str] = Query(..., min_length=2),
) -> ComparisonResponse:
    """Compare two or more eval runs.

    Loads results for each run ID from storage and runs ComparisonAnalyzer.
    """
    storage = _get_storage(request)
    if storage is None:
        raise HTTPException(status_code=503, detail="No storage backend configured.")

    from evalkit.core.types import EvalResult, EvalCase, ModelResponse, Score

    results: dict[str, list[EvalResult]] = {}

    for run_id in run_ids:
        data = await storage.get_run(run_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        model_label = data.get("model", run_id)
        run_results = []
        for r in data.get("results", []):
            case = EvalCase(id=r.get("case_id", ""), prompt="", metadata={})
            resp_data = r.get("response", {})
            response = ModelResponse(
                text=resp_data.get("text", ""),
                model=resp_data.get("model", ""),
                provider=resp_data.get("provider", ""),
                latency_ms=resp_data.get("latency_ms", 0.0),
                tokens_in=resp_data.get("tokens_in", 0),
                tokens_out=resp_data.get("tokens_out", 0),
            )
            scores = [
                Score(value=s["value"], scorer=s["scorer"], reasoning=s.get("reasoning"))
                for s in r.get("scores", [])
            ]
            run_results.append(
                EvalResult(
                    case=case,
                    response=response,
                    scores=scores,
                    aggregate_score=r.get("aggregate_score", 0.0),
                    timestamp=r.get("timestamp", ""),
                )
            )
        # Ensure unique model label if duplicate run models
        label = model_label
        if label in results:
            label = f"{label}_{run_id[:8]}"
        results[label] = run_results

    if len(results) < 2:
        raise HTTPException(
            status_code=422,
            detail="Need at least 2 distinct models to compare.",
        )

    try:
        analyzer = ComparisonAnalyzer(results=results)
        comparison = analyzer.analyze()
    except Exception as exc:
        logger.exception("Comparison failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}")

    return ComparisonResponse(
        models=comparison.models,
        win_rates=comparison.win_rates,
        elo_ratings=comparison.elo_ratings,
        score_summary=comparison.score_summary,
        head_to_head=comparison.head_to_head,
    )


# ---------------------------------------------------------------------------
# Judgments
# ---------------------------------------------------------------------------


@router.post("/judge", response_model=JudgmentResponse, tags=["judgment"])
async def submit_judgment(body: JudgmentRequest, request: Request) -> JudgmentResponse:
    """Submit a human or automated preference judgment."""
    storage = _get_storage(request)
    if storage is None:
        raise HTTPException(status_code=503, detail="No storage backend configured.")

    judgment = Judgment(
        eval_id=body.eval_id,
        preferred=body.preferred,
        models=body.models,
        reason=body.reason,
        judge=body.judge,
    )

    try:
        await storage.save_judgment(judgment)
    except Exception as exc:
        logger.exception("Failed to save judgment: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to save judgment: {exc}")

    return JudgmentResponse(eval_id=body.eval_id, preferred=body.preferred)


@router.get("/judge", tags=["judgment"])
async def list_judgments(
    request: Request,
    eval_id: str | None = Query(default=None),
    judge: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    """List stored judgments."""
    storage = _get_storage(request)
    if storage is None:
        raise HTTPException(status_code=503, detail="No storage backend configured.")

    judgments = await storage.get_judgments(eval_id=eval_id, judge=judge)
    return [j.to_dict() for j in judgments]
