# EvalKit

A production-grade LLM evaluation and comparison framework. Benchmark models across tasks with automated scoring, human preference tracking, and detailed analytics.

## Architecture

```
                         ┌─────────────┐
                         │  EvalSuite  │
                         │  (cases)    │
                         └──────┬──────┘
                                │
                    ┌───────────▼────────────┐
                    │      BatchRunner        │
                    │  (concurrency, retry)   │
                    └──┬──────────┬──────────┘
                       │          │
            ┌──────────▼──┐  ┌───▼──────────┐
            │  Provider A │  │  Provider B  │
            │  (OpenAI)   │  │  (Anthropic) │
            └──────────┬──┘  └───┬──────────┘
                       │          │
                    ┌──▼──────────▼──┐
                    │  EvalExecutor  │
                    │   (scoring)    │
                    └───────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────▼──┐   ┌──────▼──┐  ┌──────▼──┐
        │Scorer 1│   │Scorer 2 │  │Scorer N │
        │ExactM. │   │Contains │  │LLMJudge │
        └─────┬──┘   └──────┬──┘  └──────┬──┘
              │             │             │
              └─────────────┼─────────────┘
                            │
                    ┌───────▼───────┐
                    │ EvalResult    │
                    │ (scores,      │
                    │  latency,     │
                    │  tokens)      │
                    └───────┬───────┘
                            │
               ┌────────────┼────────────┐
               │            │            │
        ┌──────▼──┐  ┌──────▼──┐  ┌─────▼──────┐
        │ Storage │  │Analyzer │  │  Reports   │
        │JSON/SQL │  │Elo/H2H  │  │ MD/HTML/   │
        └─────────┘  └─────────┘  │ Charts     │
                                  └────────────┘
```

## Features

- **Multiple providers**: OpenAI, Anthropic, any OpenAI-compatible API (Ollama, Together AI, Groq), plus a `MockProvider` for offline testing
- **7 scorer types**: Exact match, contains, regex, semantic similarity (TF-IDF or sentence-transformers), LLM-as-judge, composite (weighted), and custom Python functions
- **Async runner**: Semaphore-based concurrency, per-call timeouts, automatic retries with exponential backoff
- **Comparison analytics**: Head-to-head win rates, Elo rating system, per-case score deltas, P95 statistics
- **Human preference collection**: Interactive CLI A/B comparison using rich panels
- **Dual storage backends**: JSON files (zero-dependency) or SQLite (SQLAlchemy async)
- **Report generation**: Markdown, HTML, and JSON reports; matplotlib charts (score distributions, radar plots, win-rate heatmaps, latency scatter)
- **REST API**: FastAPI with OpenAPI docs; endpoints for evaluate, runs, compare, judge
- **CLI**: Click-based CLI for all operations
- **Configuration**: Pydantic-validated YAML config with `${ENV_VAR}` interpolation

## Quick Start

### Installation

```bash
pip install evalkit
# With sentence-transformers for dense semantic similarity:
pip install "evalkit[sentence-transformers]"
```

### Run an evaluation

```bash
# Using the built-in reasoning dataset with a mock provider (no API key needed)
evalkit run reasoning --provider mock --scorer exact_match --scorer contains

# With OpenAI
OPENAI_API_KEY=sk-... evalkit run reasoning \
  --model gpt-4o-mini \
  --provider openai \
  --scorer exact_match \
  --scorer similarity
```

### Compare two runs

```bash
evalkit compare <run_id_1> <run_id_2> --format markdown
```

### Generate a report

```bash
evalkit report <run_id> --format html --charts
```

### Interactive human preference collection

```bash
evalkit judge <run_id_1> <run_id_2>
```

### List built-in datasets

```bash
evalkit datasets --details
```

### Start the API server

```bash
evalkit serve --port 8000
```

## Configuration

Create an `evalkit.yml` file:

```yaml
builtin_dataset: reasoning     # or suite_path: ./my-suite.yaml

providers:
  - name: gpt-4o-mini
    type: openai
    model: gpt-4o-mini
    api_key: ${OPENAI_API_KEY}
    temperature: 0.0
    max_tokens: 2048

scorers:
  - type: exact_match
    weight: 0.4
  - type: contains
    weight: 0.3
  - type: similarity
    weight: 0.3

runner:
  concurrency: 5
  timeout_seconds: 60.0
  max_retries: 2

storage:
  backend: json                 # or sqlite
  path: ${EVALKIT_STORAGE_PATH:-./evalkit_storage}

output_dir: ./evalkit_output
report_format: markdown
```

## Dataset Format

YAML dataset files follow this structure:

```yaml
name: my-suite
description: "Custom evaluation suite"
version: "1.0"
scorers: [exact_match, contains]
cases:
  - id: case_01
    prompt: "What is 2 + 2?"
    expected: "4"
    tags: [math, arithmetic]
    metadata:
      difficulty: easy
  - id: case_02
    prompt: "Summarize this text..."
    system_prompt: "You are a helpful assistant."
    expected: "The text is about..."
    tags: [summarization]
```

CSV format is also supported with columns: `id`, `prompt`, `expected`, `tags`, `system_prompt`, `metadata`.

## CLI Reference

| Command | Description |
|---|---|
| `evalkit run SUITE [--model M] [--provider P] [--scorer S]` | Run evaluation |
| `evalkit compare RUN_ID_1 RUN_ID_2 [--format md]` | Compare two runs |
| `evalkit report RUN_ID [--format html] [--charts]` | Generate report |
| `evalkit judge RUN_ID_1 RUN_ID_2` | Interactive A/B preference |
| `evalkit datasets [--details]` | List built-in datasets |
| `evalkit serve [--port 8000] [--reload]` | Start API server |

## API Reference

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/evaluate` | Run evaluation suite |
| `GET` | `/runs` | List runs (filterable) |
| `GET` | `/runs/{id}` | Get run details |
| `DELETE` | `/runs/{id}` | Delete a run |
| `GET` | `/compare?run_ids=A&run_ids=B` | Compare runs |
| `POST` | `/judge` | Submit preference judgment |
| `GET` | `/judge` | List judgments |

Interactive API docs: `http://localhost:8000/docs`

## Programmatic Usage

```python
import asyncio
from evalkit.providers import create_provider
from evalkit.scorers import create_scorer, ExactMatchScorer, ContainsScorer
from evalkit.scorers.composite import CompositeScorer
from evalkit.runners.batch import BatchRunner
from evalkit.datasets.builtin import load_builtin_dataset
from evalkit.comparison.analyzer import ComparisonAnalyzer
from evalkit.reports.formatter import ReportFormatter

async def main():
    # Load a built-in dataset
    suite = load_builtin_dataset("reasoning")

    # Set up providers
    providers = {
        "gpt-4o-mini": create_provider("openai", api_key="sk-...", model="gpt-4o-mini"),
        "claude-haiku": create_provider("anthropic", api_key="sk-ant-...", model="claude-3-haiku-20240307"),
    }

    # Configure a composite scorer
    scorer = CompositeScorer([
        (ExactMatchScorer(), 0.5),
        (ContainsScorer(), 0.5),
    ])

    # Run the evaluation
    runner = BatchRunner(providers=providers, scorers=[scorer], concurrency=5)
    results = await runner.run_suite(suite)

    # Compare models
    analyzer = ComparisonAnalyzer(results)
    comparison = analyzer.analyze()
    print(f"Win rates: {comparison.win_rates}")
    print(f"Elo ratings: {comparison.elo_ratings}")

    # Generate a report
    formatter = ReportFormatter(title="My Evaluation Report")
    report = formatter.format(results, comparison=comparison, fmt="markdown")
    print(report)

asyncio.run(main())
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
make test

# Run tests with coverage
make test-cov

# Lint and format
make lint
make format

# Demo run with mock provider
make run-demo
```

## License

MIT — see [LICENSE](LICENSE) for details.
