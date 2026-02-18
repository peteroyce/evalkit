"""Core dataclasses for evalkit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    """A single evaluation case with a prompt and expected behavior."""

    id: str
    prompt: str
    system_prompt: str | None = None
    expected: str | None = None  # reference answer
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "expected": self.expected,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            system_prompt=data.get("system_prompt"),
            expected=data.get("expected"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )


@dataclass
class ModelResponse:
    """The response from a model provider."""

    text: str
    model: str
    provider: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelResponse":
        return cls(
            text=data["text"],
            model=data["model"],
            provider=data["provider"],
            latency_ms=data["latency_ms"],
            tokens_in=data["tokens_in"],
            tokens_out=data["tokens_out"],
            cost_usd=data.get("cost_usd"),
            raw=data.get("raw", {}),
        )


@dataclass
class Score:
    """A score assigned by a scorer to a model response."""

    value: float  # 0.0 to 1.0
    scorer: str
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Score value must be between 0.0 and 1.0, got {self.value}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "scorer": self.scorer,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Score":
        return cls(
            value=data["value"],
            scorer=data["scorer"],
            reasoning=data.get("reasoning"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvalResult:
    """The result of evaluating a single EvalCase against a single model."""

    case: EvalCase
    response: ModelResponse
    scores: list[Score]
    aggregate_score: float  # weighted average of scores
    timestamp: str  # ISO format

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "response": self.response.to_dict(),
            "scores": [s.to_dict() for s in self.scores],
            "aggregate_score": self.aggregate_score,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalResult":
        return cls(
            case=EvalCase.from_dict(data["case"]),
            response=ModelResponse.from_dict(data["response"]),
            scores=[Score.from_dict(s) for s in data["scores"]],
            aggregate_score=data["aggregate_score"],
            timestamp=data["timestamp"],
        )


@dataclass
class Judgment:
    """A pairwise preference judgment (human or model-based)."""

    eval_id: str
    preferred: str  # model name
    models: list[str]
    reason: str | None = None
    judge: str = "human"  # "human" or model name

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "preferred": self.preferred,
            "models": self.models,
            "reason": self.reason,
            "judge": self.judge,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Judgment":
        return cls(
            eval_id=data["eval_id"],
            preferred=data["preferred"],
            models=data["models"],
            reason=data.get("reason"),
            judge=data.get("judge", "human"),
        )


@dataclass
class EvalSuite:
    """A collection of evaluation cases with associated scorers."""

    name: str
    description: str
    cases: list[EvalCase]
    scorers: list[str]
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "cases": [c.to_dict() for c in self.cases],
            "scorers": self.scorers,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSuite":
        return cls(
            name=data["name"],
            description=data["description"],
            cases=[EvalCase.from_dict(c) for c in data.get("cases", [])],
            scorers=data.get("scorers", []),
            version=data.get("version", "1.0"),
        )

    def filter_by_tags(self, tags: list[str]) -> "EvalSuite":
        """Return a new EvalSuite containing only cases with any of the given tags."""
        filtered = [c for c in self.cases if any(t in c.tags for t in tags)]
        return EvalSuite(
            name=self.name,
            description=self.description,
            cases=filtered,
            scorers=self.scorers,
            version=self.version,
        )


@dataclass
class ComparisonResult:
    """The result of comparing multiple models on the same eval suite."""

    models: list[str]
    win_rates: dict[str, float]
    elo_ratings: dict[str, float]
    score_summary: dict[str, dict[str, float]]  # model -> {mean, std, median, p95}
    head_to_head: dict[str, dict[str, float]]  # model_a -> model_b -> win_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "win_rates": self.win_rates,
            "elo_ratings": self.elo_ratings,
            "score_summary": self.score_summary,
            "head_to_head": self.head_to_head,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComparisonResult":
        return cls(
            models=data["models"],
            win_rates=data["win_rates"],
            elo_ratings=data["elo_ratings"],
            score_summary=data["score_summary"],
            head_to_head=data["head_to_head"],
        )
