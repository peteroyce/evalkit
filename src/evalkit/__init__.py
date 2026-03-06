"""EvalKit — LLM evaluation and comparison framework."""

__version__ = "0.2.0"
__author__ = "EvalKit Contributors"

from evalkit.core.types import (
    EvalCase,
    EvalResult,
    ModelResponse,
    Score,
    Judgment,
    EvalSuite,
    ComparisonResult,
)

__all__ = [
    "__version__",
    "EvalCase",
    "EvalResult",
    "ModelResponse",
    "Score",
    "Judgment",
    "EvalSuite",
    "ComparisonResult",
]
