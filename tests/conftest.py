"""Shared pytest fixtures for evalkit tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
import pytest_asyncio

from evalkit.core.types import EvalCase, EvalSuite
from evalkit.providers.mock import MockProvider
from evalkit.storage.backend import JSONFileBackend


# ---------------------------------------------------------------------------
# Provider fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider() -> MockProvider:
    """A MockProvider configured with canned responses."""
    return MockProvider(
        model="mock-model-v1",
        responses=[
            "The answer is 42.",
            "Paris is the capital of France.",
            "The ball costs $0.05.",
            "No, we cannot conclude that some roses fade quickly.",
            "Water boils at 100 degrees Celsius at sea level.",
        ],
        mode="canned",
        simulated_latency_ms=5.0,
    )


@pytest.fixture
def echo_provider() -> MockProvider:
    """A MockProvider that echoes the prompt back."""
    return MockProvider(model="echo-model", mode="echo", simulated_latency_ms=1.0)


@pytest.fixture
def random_provider() -> MockProvider:
    """A MockProvider that returns random lorem-ipsum text."""
    return MockProvider(model="random-model", mode="random", simulated_latency_ms=1.0)


# ---------------------------------------------------------------------------
# Suite fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_cases() -> list[EvalCase]:
    """A small list of EvalCase objects for testing."""
    return [
        EvalCase(
            id="case_01",
            prompt="What is the meaning of life?",
            expected="The answer is 42.",
            tags=["philosophy", "trivial"],
            metadata={"difficulty": "easy"},
        ),
        EvalCase(
            id="case_02",
            prompt="What is the capital of France?",
            expected="Paris",
            tags=["geography"],
            metadata={"difficulty": "easy"},
        ),
        EvalCase(
            id="case_03",
            prompt="A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
            expected="$0.05",
            tags=["math", "cognitive-bias"],
            metadata={"difficulty": "hard"},
        ),
    ]


@pytest.fixture
def sample_suite(sample_cases: list[EvalCase]) -> EvalSuite:
    """A small EvalSuite for testing."""
    return EvalSuite(
        name="test-suite",
        description="A small test evaluation suite.",
        cases=sample_cases,
        scorers=["exact_match"],
        version="1.0",
    )


# ---------------------------------------------------------------------------
# Storage fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def tmp_storage(tmp_path: Path) -> JSONFileBackend:
    """A JSONFileBackend pointed at a temporary directory."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return JSONFileBackend(storage_dir)


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def event_loop_policy():
    """Use the default asyncio event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
