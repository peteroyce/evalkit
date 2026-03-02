"""Click CLI for evalkit: run, compare, report, judge, serve, datasets."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import click

# Configure basic logging before anything else
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("evalkit.cli")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("evalkit").setLevel(level)


def _load_config_or_exit(config_path: str | None) -> Any:
    """Load config from file or return None."""
    if config_path is None:
        # Try default locations
        for candidate in ["evalkit.yml", "evalkit.yaml", "configs/default.yml"]:
            if Path(candidate).exists():
                config_path = candidate
                break

    if config_path and Path(config_path).exists():
        from evalkit.config.loader import load_config
        return load_config(config_path)
    return None


def _get_storage(storage_path: str, backend: str = "json") -> Any:
    if backend == "sqlite":
        from evalkit.storage.backend import SQLiteBackend
        return SQLiteBackend(storage_path)
    from evalkit.storage.backend import JSONFileBackend
    return JSONFileBackend(storage_path)


@click.group()
@click.version_option(package_name="evalkit")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """EvalKit — LLM evaluation and comparison framework.

    Run evaluation suites against multiple models, compare outputs,
    generate reports, and collect human preferences.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("suite", metavar="SUITE_PATH_OR_BUILTIN")
@click.option("--model", "-m", multiple=True, help="Model name(s) to evaluate (e.g., gpt-4o).")
@click.option("--provider", "-p", default="mock", show_default=True,
              type=click.Choice(["openai", "anthropic", "mock"]),
              help="Provider type.")
@click.option("--api-key", envvar="EVALKIT_API_KEY", default=None, help="API key.")
@click.option("--base-url", default=None, help="Custom base URL for OpenAI-compatible providers.")
@click.option("--scorer", "-s", multiple=True, default=("exact_match",),
              help="Scorer(s) to use (can specify multiple).")
@click.option("--concurrency", default=5, show_default=True, help="Concurrent API calls.")
@click.option("--output-dir", default="./evalkit_output", show_default=True)
@click.option("--storage", "storage_path", default="./evalkit_storage", show_default=True)
@click.option("--storage-backend", default="json", type=click.Choice(["json", "sqlite"]))
@click.option("--tags", "-t", multiple=True, help="Filter cases by tag.")
@click.option("--config", "-c", default=None, help="Path to a YAML config file.")
@click.option("--run-id", default=None, help="Custom run ID.")
@click.pass_context
def run(
    ctx: click.Context,
    suite: str,
    model: tuple[str, ...],
    provider: str,
    api_key: str | None,
    base_url: str | None,
    scorer: tuple[str, ...],
    concurrency: int,
    output_dir: str,
    storage_path: str,
    storage_backend: str,
    tags: tuple[str, ...],
    config: str | None,
    run_id: str | None,
) -> None:
    """Run an evaluation suite against one or more models.

    SUITE_PATH_OR_BUILTIN can be a file path (.yaml/.json/.csv) or a
    built-in dataset name (e.g., "reasoning", "summarization").
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    async def _run() -> None:
        from evalkit.datasets.loader import DatasetLoader
        from evalkit.datasets.builtin import load_builtin_dataset, list_builtin_datasets
        from evalkit.providers import create_provider
        from evalkit.scorers import create_scorer
        from evalkit.runners.batch import BatchRunner
        from evalkit.reports.formatter import ReportFormatter

        # Load dataset
        builtin_names = list_builtin_datasets()
        if suite in builtin_names:
            eval_suite = load_builtin_dataset(suite)
        else:
            tag_list = list(tags) or None
            loader = DatasetLoader(tag_filter=tag_list)
            eval_suite = loader.load(suite)

        if tags:
            eval_suite = eval_suite.filter_by_tags(list(tags))

        console.print(
            f"[bold]Suite:[/bold] {eval_suite.name} — {len(eval_suite.cases)} cases"
        )

        # Build providers
        models_to_use = list(model) or ["default"]
        providers: dict[str, Any] = {}
        for m in models_to_use:
            kwargs: dict[str, Any] = {"model": m}
            if provider != "mock":
                key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
                kwargs["api_key"] = key
                if base_url:
                    kwargs["base_url"] = base_url
            providers[m] = create_provider(provider, **kwargs)

        # Build scorers
        scorers = [create_scorer(s) for s in scorer]

        # Storage
        storage = _get_storage(storage_path, storage_backend)

        runner = BatchRunner(
            providers=providers,
            scorers=scorers,
            concurrency=concurrency,
            storage=storage,
            show_progress=True,
        )

        all_results = await runner.run_suite(eval_suite, run_id=run_id)

        # Summary table
        table = Table(title="Evaluation Results", show_header=True)
        table.add_column("Model")
        table.add_column("Cases")
        table.add_column("Mean Score", justify="right")
        table.add_column("Avg Latency (ms)", justify="right")

        import statistics

        for m_label, res_list in all_results.items():
            if res_list:
                scores = [r.aggregate_score for r in res_list]
                latencies = [r.response.latency_ms for r in res_list]
                table.add_row(
                    m_label,
                    str(len(res_list)),
                    f"{statistics.mean(scores):.3f}",
                    f"{statistics.mean(latencies):.1f}",
                )

        console.print(table)

        # Generate report
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        formatter = ReportFormatter(title=f"EvalKit Report — {eval_suite.name}")
        report_text = formatter.format(all_results, fmt="markdown")
        report_path = out_dir / "report.md"
        report_path.write_text(report_text, encoding="utf-8")
        console.print(f"[green]Report saved to '{report_path}'[/green]")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("run_id_1")
@click.argument("run_id_2")
@click.option("--storage", "storage_path", default="./evalkit_storage", show_default=True)
@click.option("--storage-backend", default="json", type=click.Choice(["json", "sqlite"]))
@click.option("--output-dir", default="./evalkit_output", show_default=True)
@click.option("--format", "fmt", default="markdown",
              type=click.Choice(["markdown", "html", "json"]))
@click.pass_context
def compare(
    ctx: click.Context,
    run_id_1: str,
    run_id_2: str,
    storage_path: str,
    storage_backend: str,
    output_dir: str,
    fmt: str,
) -> None:
    """Compare two eval runs and generate a comparison report."""
    from rich.console import Console

    console = Console()

    async def _compare() -> None:
        from evalkit.core.types import EvalResult, EvalCase, ModelResponse, Score
        from evalkit.comparison.analyzer import ComparisonAnalyzer
        from evalkit.reports.formatter import ReportFormatter

        storage = _get_storage(storage_path, storage_backend)

        results: dict[str, list[EvalResult]] = {}
        for run_id in [run_id_1, run_id_2]:
            data = await storage.get_run(run_id)
            if data is None:
                console.print(f"[red]Run '{run_id}' not found in storage.[/red]")
                sys.exit(1)

            model_label = data.get("model", run_id)
            run_results = []
            for r in data.get("results", []):
                case = EvalCase(id=r.get("case_id", ""), prompt="", metadata={})
                resp_data = r.get("response", {})
                response = ModelResponse(
                    text=resp_data.get("text", ""),
                    model=resp_data.get("model", ""),
                    provider=resp_data.get("provider", ""),
                    latency_ms=resp_data.get("latency_ms", 0.0),
                    tokens_in=resp_data.get("tokens_in", 0),
                    tokens_out=resp_data.get("tokens_out", 0),
                )
                scores = [
                    Score(value=s["value"], scorer=s["scorer"])
                    for s in r.get("scores", [])
                ]
                run_results.append(
                    EvalResult(
                        case=case,
                        response=response,
                        scores=scores,
                        aggregate_score=r.get("aggregate_score", 0.0),
                        timestamp=r.get("timestamp", ""),
                    )
                )
            label = model_label
            if label in results:
                label = f"{label}_{run_id[:8]}"
            results[label] = run_results

        if len(results) < 2:
            console.print("[red]Need at least 2 distinct models to compare.[/red]")
            sys.exit(1)

        analyzer = ComparisonAnalyzer(results)
        comparison = analyzer.analyze()

        console.print(f"[bold]Win Rates:[/bold]")
        for model in sorted(comparison.models, key=lambda m: -comparison.win_rates.get(m, 0)):
            wr = comparison.win_rates[model]
            elo = comparison.elo_ratings[model]
            console.print(f"  {model}: win_rate={wr:.1%}, elo={elo:.1f}")

        # Save report
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        formatter = ReportFormatter(title="EvalKit Comparison Report")
        report = formatter.format(results, comparison=comparison, fmt=fmt)
        suffix = {"markdown": "md", "html": "html", "json": "json"}.get(fmt, "md")
        path = out_dir / f"comparison_report.{suffix}"
        path.write_text(report, encoding="utf-8")
        console.print(f"[green]Comparison report saved to '{path}'[/green]")

    asyncio.run(_compare())


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("run_id")
@click.option("--storage", "storage_path", default="./evalkit_storage", show_default=True)
@click.option("--storage-backend", default="json", type=click.Choice(["json", "sqlite"]))
@click.option("--format", "fmt", default="markdown",
              type=click.Choice(["markdown", "html", "json"]))
@click.option("--output-dir", default="./evalkit_output", show_default=True)
@click.option("--charts", is_flag=True, help="Also generate matplotlib charts.")
@click.pass_context
def report(
    ctx: click.Context,
    run_id: str,
    storage_path: str,
    storage_backend: str,
    fmt: str,
    output_dir: str,
    charts: bool,
) -> None:
    """Generate a report for an eval run."""
    from rich.console import Console

    console = Console()

    async def _report() -> None:
        from evalkit.core.types import EvalResult, EvalCase, ModelResponse, Score
        from evalkit.reports.formatter import ReportFormatter

        storage = _get_storage(storage_path, storage_backend)
        data = await storage.get_run(run_id)
        if data is None:
            console.print(f"[red]Run '{run_id}' not found.[/red]")
            sys.exit(1)

        model_label = data.get("model", run_id)
        run_results = []
        for r in data.get("results", []):
            case = EvalCase(id=r.get("case_id", ""), prompt="", metadata={})
            resp_data = r.get("response", {})
            response = ModelResponse(
                text=resp_data.get("text", ""),
                model=resp_data.get("model", ""),
                provider=resp_data.get("provider", ""),
                latency_ms=resp_data.get("latency_ms", 0.0),
                tokens_in=resp_data.get("tokens_in", 0),
                tokens_out=resp_data.get("tokens_out", 0),
            )
            scores = [
                Score(value=s["value"], scorer=s["scorer"])
                for s in r.get("scores", [])
            ]
            run_results.append(
                EvalResult(
                    case=case,
                    response=response,
                    scores=scores,
                    aggregate_score=r.get("aggregate_score", 0.0),
                    timestamp=r.get("timestamp", ""),
                )
            )

        results = {model_label: run_results}
        formatter = ReportFormatter(title=f"EvalKit Report — {run_id}")
        report_text = formatter.format(results, fmt=fmt)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = {"markdown": "md", "html": "html", "json": "json"}.get(fmt, "md")
        path = out_dir / f"report_{run_id[:12]}.{suffix}"
        path.write_text(report_text, encoding="utf-8")
        console.print(f"[green]Report saved to '{path}'[/green]")

        if charts:
            try:
                from evalkit.reports.charts import ChartGenerator
                cg = ChartGenerator()
                fig = cg.score_distribution(results)
                cg.save(fig, out_dir / "score_distribution.png")
                console.print(f"[green]Charts saved to '{out_dir}'[/green]")
            except ImportError as exc:
                console.print(f"[yellow]Could not generate charts: {exc}[/yellow]")

    asyncio.run(_report())


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("run_id_1")
@click.argument("run_id_2")
@click.option("--storage", "storage_path", default="./evalkit_storage", show_default=True)
@click.option("--storage-backend", default="json", type=click.Choice(["json", "sqlite"]))
@click.option("--judge-id", default="human", show_default=True)
@click.option("--show-expected", is_flag=True)
@click.pass_context
def judge(
    ctx: click.Context,
    run_id_1: str,
    run_id_2: str,
    storage_path: str,
    storage_backend: str,
    judge_id: str,
    show_expected: bool,
) -> None:
    """Run an interactive human preference collection session.

    Compare model outputs from two eval runs case-by-case and record
    your A/B preferences.
    """
    from rich.console import Console

    console = Console()

    async def _judge() -> None:
        from evalkit.core.types import EvalResult, EvalCase, ModelResponse, Score
        from evalkit.comparison.human import HumanPreferenceCollector

        storage = _get_storage(storage_path, storage_backend)

        all_results: dict[str, list[EvalResult]] = {}
        for run_id in [run_id_1, run_id_2]:
            data = await storage.get_run(run_id)
            if data is None:
                console.print(f"[red]Run '{run_id}' not found.[/red]")
                sys.exit(1)

            model_label = data.get("model", run_id)
            run_results = []
            for r in data.get("results", []):
                case = EvalCase(
                    id=r.get("case_id", ""),
                    prompt=r.get("response", {}).get("text", ""),
                    expected=r.get("expected"),
                    metadata={},
                )
                resp_data = r.get("response", {})
                response = ModelResponse(
                    text=resp_data.get("text", ""),
                    model=resp_data.get("model", ""),
                    provider=resp_data.get("provider", ""),
                    latency_ms=resp_data.get("latency_ms", 0.0),
                    tokens_in=resp_data.get("tokens_in", 0),
                    tokens_out=resp_data.get("tokens_out", 0),
                )
                scores = [
                    Score(value=s["value"], scorer=s["scorer"])
                    for s in r.get("scores", [])
                ]
                run_results.append(
                    EvalResult(
                        case=case,
                        response=response,
                        scores=scores,
                        aggregate_score=r.get("aggregate_score", 0.0),
                        timestamp=r.get("timestamp", ""),
                    )
                )
            all_results[model_label] = run_results

        models = list(all_results.keys())
        if len(models) < 2:
            console.print("[red]Need results from 2 different models.[/red]")
            sys.exit(1)

        label_a, label_b = models[0], models[1]
        collector = HumanPreferenceCollector(
            show_prompt=True,
            show_expected=show_expected,
            judge_id=judge_id,
        )
        judgments = collector.collect(
            all_results[label_a], all_results[label_b], label_a, label_b
        )

        # Persist judgments
        for j in judgments:
            await storage.save_judgment(j)

        console.print(
            f"[bold green]{len(judgments)} judgments saved to storage.[/bold green]"
        )
        summary = collector.summary()
        console.print(f"Preference counts: {summary['preferred_counts']}")

    asyncio.run(_judge())


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--details", is_flag=True, help="Show case counts and descriptions.")
def datasets(details: bool) -> None:
    """List available built-in datasets."""
    from rich.console import Console
    from rich.table import Table
    from evalkit.datasets.builtin import list_builtin_datasets, load_builtin_dataset

    console = Console()
    names = list_builtin_datasets()

    if details:
        table = Table(title="Built-in Datasets")
        table.add_column("Name")
        table.add_column("Cases", justify="right")
        table.add_column("Description")
        for name in names:
            try:
                suite = load_builtin_dataset(name)
                table.add_row(name, str(len(suite.cases)), suite.description)
            except Exception as exc:
                table.add_row(name, "?", f"[red]{exc}[/red]")
        console.print(table)
    else:
        for name in names:
            click.echo(name)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--storage", "storage_path", default="./evalkit_storage", show_default=True)
@click.option("--storage-backend", default="json", type=click.Choice(["json", "sqlite"]))
@click.option("--reload", is_flag=True, help="Enable hot-reload (development mode).")
@click.option("--workers", default=1, show_default=True)
@click.pass_context
def serve(
    ctx: click.Context,
    host: str,
    port: int,
    storage_path: str,
    storage_backend: str,
    reload: bool,
    workers: int,
) -> None:
    """Start the EvalKit REST API server.

    The API is available at http://HOST:PORT/api/v1/
    Interactive docs at http://HOST:PORT/docs
    """
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn is required to run the API server. Install with: pip install uvicorn")
        sys.exit(1)

    from rich.console import Console
    console = Console()

    storage = _get_storage(storage_path, storage_backend)
    from evalkit.api.app import create_app

    app = create_app(storage=storage)

    console.print(
        f"[bold green]EvalKit API server starting[/bold green]\n"
        f"  URL: http://{host}:{port}\n"
        f"  Docs: http://{host}:{port}/docs\n"
        f"  Storage: {storage_backend} @ '{storage_path}'"
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )


def main() -> None:
    """Entry point for the evalkit CLI."""
    cli()


if __name__ == "__main__":
    main()
