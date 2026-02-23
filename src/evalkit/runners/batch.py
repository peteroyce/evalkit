"""BatchRunner — runs full EvalSuites with progress tracking and storage."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from evalkit.core.types import EvalResult, EvalSuite
from evalkit.providers.base import BaseProvider
from evalkit.runners.executor import EvalExecutor
from evalkit.scorers.base import BaseScorer

logger = logging.getLogger(__name__)


class BatchRunner:
    """Runs an entire EvalSuite against one or more providers.

    Uses an asyncio.Semaphore to bound concurrency, tracks progress with
    rich (if available), and optionally persists results to a storage backend.

    Args:
        providers: Mapping of model label to BaseProvider instance.
        scorers: List of scorers to apply to every result.
        concurrency: Maximum concurrent provider calls per model.
        timeout_seconds: Per-call timeout forwarded to EvalExecutor.
        max_retries: Per-call retries forwarded to EvalExecutor.
        scorer_weights: Optional scorer weight overrides.
        storage: Optional StorageBackend to persist results.
        show_progress: Whether to display a rich progress bar.
    """

    def __init__(
        self,
        providers: dict[str, BaseProvider],
        scorers: list[BaseScorer],
        concurrency: int = 5,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        scorer_weights: dict[str, float] | None = None,
        storage: Any | None = None,
        show_progress: bool = True,
    ) -> None:
        self._providers = providers
        self._scorers = scorers
        self._concurrency = concurrency
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._scorer_weights = scorer_weights
        self._storage = storage
        self._show_progress = show_progress

    async def run_suite(
        self,
        suite: EvalSuite,
        generate_kwargs: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> dict[str, list[EvalResult]]:
        """Run the entire suite against all configured providers.

        Args:
            suite: The EvalSuite containing cases and scorer config.
            generate_kwargs: Extra kwargs forwarded to provider.generate().
            run_id: Optional run ID (auto-generated if not supplied).

        Returns:
            Dict mapping model label to list of EvalResult (one per case).
        """
        run_id = run_id or str(uuid.uuid4())
        generate_kwargs = generate_kwargs or {}
        all_results: dict[str, list[EvalResult]] = {}

        logger.info(
            "BatchRunner: starting run '%s' — suite='%s', %d cases, %d models",
            run_id,
            suite.name,
            len(suite.cases),
            len(self._providers),
        )

        try:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                BarColumn,
                TextColumn,
                TimeElapsedColumn,
                MofNCompleteColumn,
            )
            use_rich = self._show_progress
        except ImportError:
            use_rich = False

        total_tasks = len(suite.cases) * len(self._providers)

        if use_rich:
            from rich.progress import (
                Progress,
                SpinnerColumn,
                BarColumn,
                TextColumn,
                TimeElapsedColumn,
                MofNCompleteColumn,
            )
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            )
            ctx = progress
        else:
            ctx = _NullContext()

        with ctx as prog:
            if use_rich:
                task_id = prog.add_task(
                    f"Running {suite.name}", total=total_tasks
                )

            for model_label, provider in self._providers.items():
                executor = EvalExecutor(
                    provider=provider,
                    scorers=self._scorers,
                    timeout_seconds=self._timeout,
                    max_retries=self._max_retries,
                    scorer_weights=self._scorer_weights,
                )
                semaphore = asyncio.Semaphore(self._concurrency)
                results: list[EvalResult | None] = [None] * len(suite.cases)

                async def _run_case(idx: int, case: Any, _executor: EvalExecutor) -> None:
                    async with semaphore:
                        try:
                            result = await _executor.run(case, generate_kwargs)
                            results[idx] = result
                            logger.info(
                                "BatchRunner: [%s] case '%s' -> aggregate=%.3f",
                                model_label,
                                case.id,
                                result.aggregate_score,
                            )
                        except Exception as exc:
                            logger.error(
                                "BatchRunner: [%s] case '%s' failed: %s",
                                model_label,
                                case.id,
                                exc,
                            )
                        finally:
                            if use_rich:
                                prog.advance(task_id)

                coros = [
                    _run_case(i, case, executor)
                    for i, case in enumerate(suite.cases)
                ]
                await asyncio.gather(*coros)

                model_results = [r for r in results if r is not None]
                all_results[model_label] = model_results

                logger.info(
                    "BatchRunner: model '%s' complete — %d/%d cases succeeded",
                    model_label,
                    len(model_results),
                    len(suite.cases),
                )

        # Persist to storage if configured
        if self._storage is not None:
            await self._persist_results(run_id, suite, all_results)

        return all_results

    async def _persist_results(
        self,
        run_id: str,
        suite: EvalSuite,
        all_results: dict[str, list[EvalResult]],
    ) -> None:
        for model_label, results in all_results.items():
            try:
                model_run_id = f"{run_id}_{model_label}"
                avg_score = (
                    sum(r.aggregate_score for r in results) / len(results)
                    if results else 0.0
                )
                await self._storage.save_run(
                    run_id=model_run_id,
                    suite_name=suite.name,
                    model=model_label,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    results=results,
                    summary={"mean_score": avg_score, "n_cases": len(results)},
                )
                logger.info("BatchRunner: persisted run '%s' to storage", model_run_id)
            except Exception as exc:
                logger.error(
                    "BatchRunner: failed to persist run for model '%s': %s",
                    model_label,
                    exc,
                )


class _NullContext:
    """Minimal no-op context manager for when rich is unavailable."""

    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *args: Any) -> None:
        pass
