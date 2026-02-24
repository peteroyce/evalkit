"""Comparison package — model analysis, Elo ratings, and human preferences."""

from evalkit.comparison.analyzer import ComparisonAnalyzer
from evalkit.comparison.elo import EloRating
from evalkit.comparison.human import HumanPreferenceCollector

__all__ = ["ComparisonAnalyzer", "EloRating", "HumanPreferenceCollector"]
