"""Tests for DatasetLoader and built-in datasets."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from evalkit.datasets.loader import DatasetLoader
from evalkit.core.types import EvalCase, EvalSuite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str, filename: str = "suite.yaml") -> Path:
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def _write_json(tmp_path: Path, data: dict, filename: str = "suite.json") -> Path:
    p = tmp_path / filename
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_csv(tmp_path: Path, content: str, filename: str = "suite.csv") -> Path:
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# YAML loading tests
# ---------------------------------------------------------------------------


class TestYAMLLoader:
    def test_load_basic_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
            name: my-suite
            description: A test suite
            version: "1.0"
            scorers: [exact_match]
            cases:
              - id: case_01
                prompt: "What is 2+2?"
                expected: "4"
                tags: [math]
        """
        path = _write_yaml(tmp_path, yaml_content)
        loader = DatasetLoader()
        suite = loader.load(path)

        assert suite.name == "my-suite"
        assert suite.description == "A test suite"
        assert len(suite.cases) == 1
        assert suite.cases[0].id == "case_01"
        assert suite.cases[0].prompt == "What is 2+2?"
        assert suite.cases[0].expected == "4"
        assert "math" in suite.cases[0].tags

    def test_load_multiple_cases(self, tmp_path: Path) -> None:
        yaml_content = """
            name: multi
            description: Multiple cases
            cases:
              - id: c1
                prompt: "Q1?"
              - id: c2
                prompt: "Q2?"
              - id: c3
                prompt: "Q3?"
        """
        path = _write_yaml(tmp_path, yaml_content)
        suite = DatasetLoader().load(path)
        assert len(suite.cases) == 3

    def test_load_with_system_prompt(self, tmp_path: Path) -> None:
        yaml_content = """
            name: system-test
            description: System prompt test
            cases:
              - id: c1
                prompt: "Hello"
                system_prompt: "You are a helpful assistant."
        """
        path = _write_yaml(tmp_path, yaml_content)
        suite = DatasetLoader().load(path)
        assert suite.cases[0].system_prompt == "You are a helpful assistant."

    def test_load_with_metadata(self, tmp_path: Path) -> None:
        yaml_content = """
            name: meta-test
            description: Metadata test
            cases:
              - id: c1
                prompt: "Test?"
                metadata:
                  difficulty: hard
                  category: reasoning
        """
        path = _write_yaml(tmp_path, yaml_content)
        suite = DatasetLoader().load(path)
        assert suite.cases[0].metadata["difficulty"] == "hard"

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DatasetLoader().load(tmp_path / "nonexistent.yaml")

    def test_unsupported_extension_raises_error(self, tmp_path: Path) -> None:
        p = tmp_path / "suite.txt"
        p.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file format"):
            DatasetLoader().load(p)

    def test_tag_filter_applied(self, tmp_path: Path) -> None:
        yaml_content = """
            name: filter-test
            description: Tag filter
            cases:
              - id: c1
                prompt: "Math question?"
                tags: [math]
              - id: c2
                prompt: "Logic question?"
                tags: [logic]
              - id: c3
                prompt: "Math+logic?"
                tags: [math, logic]
        """
        path = _write_yaml(tmp_path, yaml_content)
        loader = DatasetLoader(tag_filter=["math"])
        suite = loader.load(path)
        assert len(suite.cases) == 2
        for case in suite.cases:
            assert "math" in case.tags

    def test_default_scorers_used_when_not_specified(self, tmp_path: Path) -> None:
        yaml_content = """
            name: no-scorers
            description: No scorers defined
            cases:
              - id: c1
                prompt: "Q?"
        """
        path = _write_yaml(tmp_path, yaml_content)
        loader = DatasetLoader(default_scorers=["similarity"])
        suite = loader.load(path)
        assert "similarity" in suite.scorers

    def test_case_without_expected_is_allowed(self, tmp_path: Path) -> None:
        yaml_content = """
            name: no-expected
            description: Cases without expected
            cases:
              - id: c1
                prompt: "Creative writing task."
        """
        path = _write_yaml(tmp_path, yaml_content)
        suite = DatasetLoader().load(path)
        assert suite.cases[0].expected is None


# ---------------------------------------------------------------------------
# JSON loading tests
# ---------------------------------------------------------------------------


class TestJSONLoader:
    def test_load_basic_json(self, tmp_path: Path) -> None:
        data = {
            "name": "json-suite",
            "description": "JSON test",
            "scorers": ["exact_match"],
            "cases": [
                {"id": "j1", "prompt": "JSON prompt?", "expected": "JSON answer."}
            ],
        }
        path = _write_json(tmp_path, data)
        suite = DatasetLoader().load(path)
        assert suite.name == "json-suite"
        assert len(suite.cases) == 1
        assert suite.cases[0].expected == "JSON answer."

    def test_auto_generates_case_ids(self, tmp_path: Path) -> None:
        data = {
            "name": "auto-id",
            "description": "Auto ID test",
            "cases": [{"prompt": "Q?"}, {"prompt": "Q2?"}],
        }
        path = _write_json(tmp_path, data)
        suite = DatasetLoader().load(path)
        assert all(c.id for c in suite.cases)


# ---------------------------------------------------------------------------
# CSV loading tests
# ---------------------------------------------------------------------------


class TestCSVLoader:
    def test_load_basic_csv(self, tmp_path: Path) -> None:
        csv_content = """\
            id,prompt,expected,tags
            csv_01,What is 2+2?,4,math
            csv_02,Capital of France?,Paris,geography
        """
        path = _write_csv(tmp_path, csv_content)
        suite = DatasetLoader().load(path)
        assert len(suite.cases) == 2
        assert suite.cases[0].id == "csv_01"
        assert suite.cases[0].expected == "4"

    def test_csv_with_system_prompt(self, tmp_path: Path) -> None:
        csv_content = """\
            id,prompt,expected,system_prompt
            c1,Q?,A,Be helpful.
        """
        path = _write_csv(tmp_path, csv_content)
        suite = DatasetLoader().load(path)
        assert suite.cases[0].system_prompt == "Be helpful."

    def test_csv_tags_semicolon_separated(self, tmp_path: Path) -> None:
        csv_content = """\
            id,prompt,tags
            c1,Q?,math;logic
        """
        path = _write_csv(tmp_path, csv_content)
        suite = DatasetLoader().load(path)
        assert "math" in suite.cases[0].tags
        assert "logic" in suite.cases[0].tags


# ---------------------------------------------------------------------------
# Built-in dataset tests
# ---------------------------------------------------------------------------


class TestBuiltinDatasets:
    def test_list_builtin_datasets(self) -> None:
        from evalkit.datasets.builtin import list_builtin_datasets
        names = list_builtin_datasets()
        assert "reasoning" in names
        assert "summarization" in names

    def test_load_reasoning_dataset(self) -> None:
        from evalkit.datasets.builtin import load_builtin_dataset
        try:
            suite = load_builtin_dataset("reasoning")
            assert suite.name == "reasoning"
            assert len(suite.cases) >= 5
            for case in suite.cases:
                assert case.id
                assert case.prompt
        except FileNotFoundError:
            pytest.skip("Built-in dataset files not found in this test environment.")

    def test_load_unknown_builtin_raises_keyerror(self) -> None:
        from evalkit.datasets.builtin import load_builtin_dataset
        with pytest.raises(KeyError, match="Unknown built-in dataset"):
            load_builtin_dataset("nonexistent-dataset")
