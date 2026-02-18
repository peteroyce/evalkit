# EvalKit Makefile
# ================

.PHONY: all install install-dev test test-cov lint format format-check clean serve docs help

PYTHON := python
PIP    := pip
PYTEST := pytest
RUFF   := ruff

# Default target
all: test

## install       Install the package in the current environment
install:
	$(PIP) install -e .

## install-dev   Install the package with development dependencies
install-dev:
	$(PIP) install -e ".[dev]"

## test           Run the full test suite
test:
	$(PYTEST) tests/ -v

## test-cov       Run tests with coverage report
test-cov:
	$(PYTEST) tests/ --cov=evalkit --cov-report=term-missing --cov-report=html -v

## test-fast      Run tests excluding slow tests
test-fast:
	$(PYTEST) tests/ -m "not slow" -v

## test-scorers   Run only scorer tests
test-scorers:
	$(PYTEST) tests/test_scorers/ -v

## test-runners   Run only runner tests
test-runners:
	$(PYTEST) tests/test_runners/ -v

## test-api       Run only API tests
test-api:
	$(PYTEST) tests/test_api/ -v

## lint           Run ruff linter
lint:
	$(RUFF) check src/ tests/

## lint-fix       Run ruff linter with auto-fix
lint-fix:
	$(RUFF) check --fix src/ tests/

## format         Format code with ruff formatter
format:
	$(RUFF) format src/ tests/

## format-check   Check formatting without making changes
format-check:
	$(RUFF) format --check src/ tests/

## check          Run lint + format check (CI-ready)
check: lint format-check

## serve          Start the API server (development mode)
serve:
	evalkit serve --port 8000 --reload

## serve-prod     Start the API server (production mode)
serve-prod:
	evalkit serve --port 8000 --workers 4

## run-demo       Run the built-in reasoning dataset with a mock provider
run-demo:
	evalkit run reasoning --provider mock --scorer exact_match --scorer contains

## datasets       List available built-in datasets
datasets:
	evalkit datasets --details

## clean          Remove build artifacts and caches
clean:
	rm -rf build/ dist/ .eggs/ *.egg-info
	rm -rf .pytest_cache/ .ruff_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

## build          Build distribution packages
build:
	$(PYTHON) -m build

## help           Show this help message
help:
	@echo "EvalKit Makefile targets:"
	@echo ""
	@grep -E '^## ' Makefile | sed 's/## /  /'
