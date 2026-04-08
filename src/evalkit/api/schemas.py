"""Pydantic request/response models for the evalkit REST API."""

from __future__ import annotations

from typing import Any, Literal

import re

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ProviderRequest(BaseModel):
    """Inline provider specification for API requests."""

    name: str
    type: Literal["openai", "anthropic", "mock"] = "mock"
    model: str = "mock-model"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)


class ScorerRequest(BaseModel):
    """Inline scorer specification for API requests."""

    type: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluateRequest(BaseModel):
    """Request body for POST /evaluate."""

    suite_path: str | None = Field(
        default=None,
        description="Path to a dataset YAML/JSON/CSV file on the server.",
    )
    builtin_dataset: str | None = Field(
        default=None,
        description="Name of a built-in dataset.",
    )
    providers: list[ProviderRequest] = Field(
        default_factory=list,
        description="Providers to evaluate against.",
    )
    scorers: list[ScorerRequest] = Field(
        default_factory=list,
        description="Scorers to apply.",
    )
    concurrency: int = Field(default=5, gt=0)
    tag_filter: list[str] = Field(default_factory=list)
    run_id: str | None = None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) > 128:
            raise ValueError("run_id must be 128 characters or fewer")
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("run_id must contain only alphanumeric characters, hyphens, and underscores")
        return v


class JudgmentRequest(BaseModel):
    """Request body for POST /judge."""

    eval_id: str
    preferred: str
    models: list[str]
    reason: str | None = None
    judge: str = "human"


class CompareRequest(BaseModel):
    """Query parameters / request for GET /compare."""

    run_ids: list[str] = Field(..., min_length=2)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ScoreResponse(BaseModel):
    value: float
    scorer: str
    reasoning: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponseResponse(BaseModel):
    text: str
    model: str
    provider: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float | None = None


class EvalResultResponse(BaseModel):
    case_id: str
    prompt: str
    expected: str | None = None
    response: ModelResponseResponse
    scores: list[ScoreResponse]
    aggregate_score: float
    timestamp: str


class RunSummaryResponse(BaseModel):
    run_id: str
    suite_name: str | None = None
    model: str | None = None
    timestamp: str | None = None
    n_cases: int = 0
    mean_score: float = 0.0


class RunDetailResponse(BaseModel):
    run_id: str
    suite_name: str | None = None
    model: str | None = None
    timestamp: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    results: list[EvalResultResponse] = Field(default_factory=list)


class ComparisonResponse(BaseModel):
    models: list[str]
    win_rates: dict[str, float]
    elo_ratings: dict[str, float]
    score_summary: dict[str, dict[str, float]]
    head_to_head: dict[str, dict[str, float]]


class EvaluateResponse(BaseModel):
    run_ids: dict[str, str]  # model_label -> run_id
    message: str = "Evaluation complete."


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class JudgmentResponse(BaseModel):
    message: str = "Judgment recorded."
    eval_id: str
    preferred: str
