# EvalKit

An LLM evaluation and comparison framework: run a suite of prompts against several models,
score the outputs with pluggable scorers, and compare the models on win rates, Elo and latency.
It runs end to end offline with a mock provider, so evaluations of the harness itself cost
nothing and stay deterministic.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

## Features

- **Providers**: OpenAI chat-completions (any compatible endpoint via `base_url` — Azure,
  Together, Groq, Ollama, vLLM), Anthropic messages, and a `MockProvider` with canned, echo and
  random modes plus a `fail_every_n` switch for exercising retry paths.
- **Seven scorers**: exact match, contains, regex, semantic similarity, LLM-as-judge, composite
  (weighted, normalised internally) and custom Python callables. Similarity uses
  sentence-transformers when installed and falls back to a built-in TF-IDF cosine implementation,
  so nothing extra is required to get started.
- **Async execution**: semaphore-bounded concurrency, per-call timeouts and retries with
  exponential backoff. A scorer that raises is recorded as a 0.0 score carrying the error, so one
  broken scorer never discards a whole run.
- **Comparison analytics**: per-case pairwise head-to-head, overall win rates, Elo ratings
  (start 1500, K=32, ties scored 0.5) and mean/std/median/p95 per model. Elo is derived from
  case-by-case comparisons rather than mean scores, so a couple of outliers cannot dominate.
- **Human preference collection**: an interactive CLI that shows two outputs side by side and
  records A/B/tie judgments to storage.
- **Storage**: JSON files (no service required) or SQLite through SQLAlchemy async, behind one
  `StorageBackend` interface.
- **Reports and charts**: Markdown, HTML and JSON reports; matplotlib score distributions, radar
  charts, win-rate heatmaps, latency-vs-score scatter and Elo leaderboards.
- **Cost and token accounting**: per-model input/output price tables for the OpenAI and Claude
  families, and a tiktoken-backed token counter with truncation helpers.
- **REST API**: FastAPI with OpenAPI docs; run IDs and suite paths are validated to reject
  traversal and malformed identifiers before touching the filesystem.

## Architecture

```
   EvalSuite (YAML / JSON / CSV, or a built-in dataset)
        │  cases: prompt, expected, tags, metadata
        ▼
   BatchRunner ──────────────────────────────► one EvalExecutor per model
        │  semaphore(concurrency)                   │
        │                                           ├─ provider.generate()  (timeout, retries)
        ├── Provider A (openai)                     └─ every scorer → Score(0..1)
        ├── Provider B (anthropic)                        │
        └── Provider C (mock)                             ▼
                                              EvalResult: response, scores,
                                              weighted aggregate, latency, tokens
        ┌───────────────────────────┬──────────────────────┴───────────────┐
        ▼                           ▼                                      ▼
  StorageBackend            ComparisonAnalyzer                     ReportFormatter
  JSON files / SQLite       head-to-head, Elo, p95                 Markdown / HTML / JSON
        │                           │                              ChartGenerator (matplotlib)
        └── runs + judgments ───────┘
```

| Module | Responsibility |
|---|---|
| `core/types.py` | `EvalCase`, `EvalSuite`, `ModelResponse`, `Score`, `EvalResult`, `Judgment`, `ComparisonResult` |
| `providers/`, `scorers/` | HTTP adapters and scorer implementations, each behind a factory function |
| `runners/` | `EvalExecutor` (single case) and `BatchRunner` (suite across models) |
| `comparison/` | `ComparisonAnalyzer`, `EloRating`, `HumanPreferenceCollector` |
| `storage/`, `reports/` | JSON and SQLite backends; `ReportFormatter` and `ChartGenerator` |
| `api/`, `cli/`, `config/` | FastAPI app, Click command group, Pydantic config with `${ENV_VAR}` interpolation |

## Quickstart

```bash
git clone <repo-url> && cd evalkit
pip install -e ".[dev]"
# optional: dense embeddings for the similarity scorer
pip install -e ".[sentence-transformers]"
```

Run the bundled reasoning suite (20 cases) against the mock provider — no API key needed:

```bash
evalkit datasets --details          # reasoning (20 cases), summarization (16 cases)
evalkit run reasoning --provider mock --scorer exact_match --scorer contains
```

Against a real model:

```bash
export OPENAI_API_KEY=sk-...
evalkit run reasoning -p openai -m gpt-4o-mini -s exact_match -s similarity
```

Each run writes results to `./evalkit_storage` and a Markdown report to `./evalkit_output`.
The run ID for a model is `<run_id>_<model_label>`, which is what `compare`, `report` and
`judge` expect.

```bash
evalkit compare <run_id_1> <run_id_2> --format markdown
evalkit report <run_id> --format html --charts
evalkit judge <run_id_1> <run_id_2>      # interactive A/B preferences
evalkit serve --port 8000                # REST API + docs at /docs
```

Container image (multi-stage, non-root, SQLite storage, healthcheck on `/api/v1/health`):

```bash
docker build -f docker/Dockerfile -t evalkit .
docker run -p 8000:8000 evalkit
```

## Usage

```python
import asyncio
from evalkit.datasets.builtin import load_builtin_dataset
from evalkit.providers import create_provider
from evalkit.scorers import ExactMatchScorer, ContainsScorer, CompositeScorer
from evalkit.runners.batch import BatchRunner
from evalkit.comparison.analyzer import ComparisonAnalyzer
from evalkit.reports.formatter import ReportFormatter

async def main() -> None:
    suite = load_builtin_dataset("reasoning")

    providers = {
        "gpt-4o-mini": create_provider("openai", api_key="sk-...", model="gpt-4o-mini"),
        "claude-haiku": create_provider("anthropic", api_key="sk-ant-...",
                                        model="claude-3-5-haiku-20241022"),
    }
    scorer = CompositeScorer([(ExactMatchScorer(), 0.5), (ContainsScorer(), 0.5)])

    results = await BatchRunner(providers=providers, scorers=[scorer], concurrency=5) \
        .run_suite(suite)

    comparison = ComparisonAnalyzer(results).analyze()   # needs at least two models
    print(comparison.win_rates, comparison.elo_ratings, comparison.score_summary)

    print(ReportFormatter(title="Reasoning suite").format(
        results, comparison=comparison, fmt="markdown"))

asyncio.run(main())
```

### CLI

| Command | Description |
|---|---|
| `evalkit run SUITE [-m MODEL] [-p PROVIDER] [-s SCORER] [-t TAG]` | Run a suite (file path or built-in name) |
| `evalkit compare RUN_ID_1 RUN_ID_2 [--format markdown\|html\|json]` | Head-to-head comparison report |
| `evalkit report RUN_ID [--format html] [--charts]` | Report for a single run |
| `evalkit judge RUN_ID_1 RUN_ID_2 [--show-expected]` | Interactive human preferences |
| `evalkit datasets [--details]` | List built-in datasets |
| `evalkit serve [--port 8000] [--storage-backend sqlite]` | Start the REST API |

Every command takes `--storage` and `--storage-backend` (`json` or `sqlite`).

### REST API

Base URL `http://localhost:8000/api/v1`, interactive docs at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check with package version |
| POST | `/evaluate` | Run a suite against provider and scorer configs; returns run IDs |
| GET | `/runs` | List runs, filter by `suite_name`, `model`, `limit`, `offset` |
| GET | `/runs/{id}` | Full results for a run |
| DELETE | `/runs/{id}` | Delete a run |
| GET | `/compare?run_ids=A&run_ids=B` | Compare two or more runs |
| POST | `/judge` | Store a preference judgment |
| GET | `/judge` | List judgments, filter by `eval_id` or `judge` |

### Datasets and configuration

Suites load from YAML, JSON or CSV. `datasets/reasoning.yml` and `datasets/summarization.yml`
show the format:

```yaml
name: my-suite
description: "Custom evaluation suite"
version: "1.0"
scorers: [exact_match, contains]
cases:
  - id: case_01
    prompt: "A bat and a ball cost $1.10 in total..."
    expected: "$0.05"
    system_prompt: "You are a careful reasoner."
    tags: [math, arithmetic]
    metadata: {difficulty: easy}
```

`configs/default.yml` is the documented shape of an `EvalConfig`: providers, scorers, runner and
storage settings, with `${VAR}` and `${VAR:-default}` interpolation. Load and validate it with
`evalkit.config.loader.load_config(path)`; the CLI itself is driven by flags.

Environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `EVALKIT_API_KEY` for provider
credentials; `EVALKIT_DATASETS_DIR` to confine API-supplied suite paths to one directory;
`ALLOWED_ORIGINS` for API CORS; `EVALKIT_STORAGE_PATH` and `EVALKIT_OUTPUT_DIR` for config
interpolation and the container defaults.

## Tech stack

Python 3.11+, httpx, Pydantic v2, Click, Rich, FastAPI, Uvicorn, SQLAlchemy 2 with aiosqlite,
matplotlib, NumPy, scikit-learn, tiktoken, PyYAML. Packaged with hatchling; linted with ruff.

## Testing

```bash
make test        # pytest tests/ -v
make test-cov    # with coverage
make lint        # ruff check src/ tests/
```

Over 130 tests cover the scorers, executor and batch runner, dataset loading, storage backends,
comparison analytics and the API routes; they run offline against `MockProvider`. GitHub Actions
(`.github/workflows/ci.yml`) runs lint, the test matrix on Python 3.11 and 3.12 across Linux,
macOS and Windows, a package build and a Docker build.

## License

MIT — see [LICENSE](LICENSE).
