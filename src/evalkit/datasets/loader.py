"""DatasetLoader — load EvalSuites from YAML, JSON, and CSV files."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from evalkit.core.types import EvalCase, EvalSuite

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads EvalSuites from YAML, JSON, or CSV files.

    YAML/JSON format::

        name: my-suite
        description: "Description of the suite"
        version: "1.0"
        scorers: [exact_match, similarity]
        cases:
          - id: case_01
            prompt: "What is 2+2?"
            expected: "4"
            tags: [math]
            metadata:
              difficulty: easy

    CSV format (one case per row)::

        id,prompt,expected,tags,system_prompt
        case_01,"What is 2+2?","4","math",
        case_02,"Capital of France?","Paris","geography",

    Args:
        default_scorers: Scorers to use when the dataset file doesn't specify any.
        tag_filter: If provided, only include cases that have at least one of
                    these tags.
    """

    def __init__(
        self,
        default_scorers: list[str] | None = None,
        tag_filter: list[str] | None = None,
    ) -> None:
        self._default_scorers = default_scorers or ["exact_match"]
        self._tag_filter = tag_filter

    def load(self, path: str | Path) -> EvalSuite:
        """Load an EvalSuite from a file.

        Args:
            path: Path to a .yaml, .yml, .json, or .csv file.

        Returns:
            An EvalSuite object.

        Raises:
            ValueError: If the file format is unsupported or data is invalid.
            FileNotFoundError: If the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: '{p}'")

        suffix = p.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            suite = self._load_yaml(p)
        elif suffix == ".json":
            suite = self._load_json(p)
        elif suffix == ".csv":
            suite = self._load_csv(p)
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. "
                "Supported: .yaml, .yml, .json, .csv"
            )

        if self._tag_filter:
            original_count = len(suite.cases)
            suite = suite.filter_by_tags(self._tag_filter)
            logger.info(
                "DatasetLoader: tag filter applied — %d/%d cases retained",
                len(suite.cases),
                original_count,
            )

        logger.info(
            "DatasetLoader: loaded suite '%s' with %d cases from '%s'",
            suite.name,
            len(suite.cases),
            p,
        )
        return suite

    def load_many(self, paths: list[str | Path]) -> list[EvalSuite]:
        """Load multiple suites from a list of file paths."""
        return [self.load(p) for p in paths]

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_yaml(self, path: Path) -> EvalSuite:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML datasets. "
                "Install with: pip install pyyaml"
            ) from exc

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"YAML dataset file must contain a mapping at the top level: '{path}'")

        return self._parse_dict(data, path.stem)

    def _load_json(self, path: Path) -> EvalSuite:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"JSON dataset file must contain an object at the top level: '{path}'")

        return self._parse_dict(data, path.stem)

    def _parse_dict(self, data: dict[str, Any], default_name: str) -> EvalSuite:
        name = data.get("name", default_name)
        description = data.get("description", "")
        version = str(data.get("version", "1.0"))
        scorers = data.get("scorers", self._default_scorers)

        raw_cases = data.get("cases", [])
        if not isinstance(raw_cases, list):
            raise ValueError("'cases' must be a list.")

        cases: list[EvalCase] = []
        for i, raw in enumerate(raw_cases):
            if not isinstance(raw, dict):
                raise ValueError(f"Case at index {i} must be a mapping.")
            case_id = raw.get("id", f"case_{i:04d}")
            if not raw.get("prompt"):
                raise ValueError(f"Case '{case_id}' is missing a 'prompt' field.")
            cases.append(
                EvalCase(
                    id=str(case_id),
                    prompt=str(raw["prompt"]),
                    system_prompt=raw.get("system_prompt"),
                    expected=raw.get("expected"),
                    metadata=raw.get("metadata", {}),
                    tags=[str(t) for t in raw.get("tags", [])],
                )
            )

        return EvalSuite(
            name=name,
            description=description,
            cases=cases,
            scorers=scorers,
            version=version,
        )

    def _load_csv(self, path: Path) -> EvalSuite:
        cases: list[EvalCase] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file appears to be empty: '{path}'")

            for i, row in enumerate(reader):
                case_id = row.get("id", f"case_{i:04d}").strip()
                prompt = row.get("prompt", "").strip()
                if not prompt:
                    logger.warning("CSV row %d has no prompt; skipping.", i)
                    continue

                # Parse tags: either JSON list or comma-separated string
                raw_tags = row.get("tags", "").strip()
                if raw_tags.startswith("["):
                    try:
                        tags = json.loads(raw_tags)
                    except json.JSONDecodeError:
                        tags = [t.strip() for t in raw_tags.strip("[]").split(",") if t.strip()]
                elif raw_tags:
                    tags = [t.strip() for t in raw_tags.split(";") if t.strip()]
                else:
                    tags = []

                # Parse metadata if present
                raw_meta = row.get("metadata", "").strip()
                metadata: dict[str, Any] = {}
                if raw_meta:
                    try:
                        metadata = json.loads(raw_meta)
                    except json.JSONDecodeError:
                        pass

                cases.append(
                    EvalCase(
                        id=case_id,
                        prompt=prompt,
                        system_prompt=row.get("system_prompt") or None,
                        expected=row.get("expected") or None,
                        metadata=metadata,
                        tags=tags,
                    )
                )

        return EvalSuite(
            name=path.stem,
            description=f"Loaded from {path.name}",
            cases=cases,
            scorers=self._default_scorers,
            version="1.0",
        )
