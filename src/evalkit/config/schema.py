"""Pydantic configuration schemas for evalkit."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    name: str = Field(..., description="Display name / label for this provider instance.")
    type: Literal["openai", "anthropic", "mock"] = Field(
        ..., description="Provider type — determines the class used."
    )
    model: str = Field(..., description="Model identifier (e.g., 'gpt-4o-mini').")
    api_key: str | None = Field(
        default=None,
        description="API key. Use '${ENV_VAR}' syntax to reference an environment variable.",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom base URL for OpenAI-compatible providers.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    timeout: float = Field(default=60.0, gt=0.0)
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters.",
    )

    model_config = {"extra": "allow"}


class ScorerConfig(BaseModel):
    """Configuration for a single scorer."""

    type: str = Field(..., description="Scorer type key (e.g., 'exact_match', 'llm_judge').")
    weight: float = Field(default=1.0, gt=0.0, description="Weight for composite scoring.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Scorer-specific parameters.",
    )

    model_config = {"extra": "allow"}


class StorageConfig(BaseModel):
    """Configuration for the storage backend."""

    backend: Literal["json", "sqlite"] = Field(
        default="json",
        description="Storage backend type.",
    )
    path: str = Field(
        default="./evalkit_storage",
        description="Directory or file path for storage.",
    )

    model_config = {"extra": "allow"}


class RunnerConfig(BaseModel):
    """Configuration for the eval runner."""

    concurrency: int = Field(
        default=5,
        gt=0,
        description="Maximum concurrent provider API calls.",
    )
    timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Per-call timeout in seconds.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Number of retries on transient failures.",
    )
    show_progress: bool = Field(
        default=True,
        description="Whether to display a rich progress bar during runs.",
    )

    model_config = {"extra": "allow"}


class EvalConfig(BaseModel):
    """Top-level configuration for an evalkit evaluation."""

    suite_path: str | None = Field(
        default=None,
        description="Path to the dataset YAML/JSON/CSV file.",
    )
    builtin_dataset: str | None = Field(
        default=None,
        description="Name of a built-in dataset (alternative to suite_path).",
    )
    providers: list[ProviderConfig] = Field(
        default_factory=list,
        description="List of provider configurations to evaluate against.",
    )
    scorers: list[ScorerConfig] = Field(
        default_factory=list,
        description="Ordered list of scorers to apply.",
    )
    runner: RunnerConfig = Field(
        default_factory=RunnerConfig,
        description="Runner/executor configuration.",
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Storage backend configuration.",
    )
    output_dir: str = Field(
        default="./evalkit_output",
        description="Directory for saving reports and charts.",
    )
    report_format: Literal["markdown", "html", "json"] = Field(
        default="markdown",
        description="Default report format.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Filter eval cases to only those with these tags.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional configuration.",
    )

    @model_validator(mode="after")
    def _check_suite_or_builtin(self) -> "EvalConfig":
        if self.suite_path is None and self.builtin_dataset is None:
            # Neither is required at schema validation time (can be set later)
            pass
        if self.suite_path is not None and self.builtin_dataset is not None:
            raise ValueError(
                "Specify either 'suite_path' or 'builtin_dataset', not both."
            )
        return self

    @field_validator("providers")
    @classmethod
    def _validate_provider_names_unique(
        cls, providers: list[ProviderConfig]
    ) -> list[ProviderConfig]:
        names = [p.name for p in providers]
        if len(names) != len(set(names)):
            raise ValueError("Provider names must be unique within a config.")
        return providers

    model_config = {"extra": "allow"}
