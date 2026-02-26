"""Datasets package — loading eval suites from files and built-in benchmarks."""

from evalkit.datasets.loader import DatasetLoader
from evalkit.datasets.builtin import list_builtin_datasets, load_builtin_dataset

__all__ = ["DatasetLoader", "list_builtin_datasets", "load_builtin_dataset"]
