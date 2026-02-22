"""Scorers package — automated evaluation of model outputs."""

from evalkit.scorers.base import BaseScorer
from evalkit.scorers.exact_match import ExactMatchScorer
from evalkit.scorers.contains import ContainsScorer
from evalkit.scorers.regex import RegexScorer
from evalkit.scorers.similarity import SemanticSimilarityScorer
from evalkit.scorers.llm_judge import LLMJudgeScorer
from evalkit.scorers.composite import CompositeScorer
from evalkit.scorers.custom import CustomScorer


def create_scorer(scorer_type: str, **kwargs) -> BaseScorer:
    """Factory function to create a scorer by type name.

    Args:
        scorer_type: One of "exact_match", "contains", "regex",
                     "similarity", "llm_judge", "composite", "custom".
        **kwargs: Scorer-specific configuration kwargs.

    Returns:
        An instantiated BaseScorer subclass.
    """
    registry: dict[str, type[BaseScorer]] = {
        "exact_match": ExactMatchScorer,
        "contains": ContainsScorer,
        "regex": RegexScorer,
        "similarity": SemanticSimilarityScorer,
        "llm_judge": LLMJudgeScorer,
        "composite": CompositeScorer,
        "custom": CustomScorer,
    }
    if scorer_type not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown scorer type '{scorer_type}'. Available: [{available}]"
        )
    return registry[scorer_type](**kwargs)


__all__ = [
    "BaseScorer",
    "ExactMatchScorer",
    "ContainsScorer",
    "RegexScorer",
    "SemanticSimilarityScorer",
    "LLMJudgeScorer",
    "CompositeScorer",
    "CustomScorer",
    "create_scorer",
]
