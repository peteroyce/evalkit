"""Core types and registry for evalkit."""

from evalkit.core.types import (
    EvalCase,
    EvalResult,
    ModelResponse,
    Score,
    Judgment,
    EvalSuite,
    ComparisonResult,
)
from evalkit.core.registry import Registry, global_registry

__all__ = [
    "EvalCase",
    "EvalResult",
    "ModelResponse",
    "Score",
    "Judgment",
    "EvalSuite",
    "ComparisonResult",
    "Registry",
    "global_registry",
]
