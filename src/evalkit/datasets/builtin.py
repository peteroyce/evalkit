"""Built-in benchmark datasets for evalkit."""

from __future__ import annotations

import logging
from pathlib import Path

from evalkit.core.types import EvalSuite
from evalkit.datasets.loader import DatasetLoader

logger = logging.getLogger(__name__)

# Path to the datasets/ directory at the repository root
_PACKAGE_DIR = Path(__file__).parent
_DATASETS_DIR = _PACKAGE_DIR.parents[3] / "datasets"


_BUILTIN_REGISTRY: dict[str, str] = {
    "reasoning": "reasoning.yml",
    "summarization": "summarization.yml",
}


def list_builtin_datasets() -> list[str]:
    """Return a list of available built-in dataset names."""
    return sorted(_BUILTIN_REGISTRY.keys())


def load_builtin_dataset(name: str) -> EvalSuite:
    """Load a built-in benchmark dataset by name.

    Args:
        name: One of the names returned by ``list_builtin_datasets()``.

    Returns:
        An EvalSuite loaded from the bundled YAML file.

    Raises:
        KeyError: If the dataset name is not registered.
        FileNotFoundError: If the dataset file is missing.
    """
    if name not in _BUILTIN_REGISTRY:
        available = ", ".join(list_builtin_datasets())
        raise KeyError(
            f"Unknown built-in dataset '{name}'. Available: [{available}]"
        )

    filename = _BUILTIN_REGISTRY[name]
    path = _DATASETS_DIR / filename

    if not path.exists():
        # Try relative to the package install location (installed packages)
        alt_path = _PACKAGE_DIR.parents[2] / "datasets" / filename
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(
                f"Built-in dataset file '{filename}' not found. "
                f"Checked: '{path}' and '{alt_path}'"
            )

    loader = DatasetLoader()
    suite = loader.load(path)
    logger.info("Loaded built-in dataset '%s' from '%s'", name, path)
    return suite
