"""StorageBackend ABC and implementations: JSONFileBackend, SQLiteBackend."""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evalkit.core.types import EvalResult, Judgment

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for eval run persistence backends."""

    @abstractmethod
    async def save_run(
        self,
        run_id: str,
        suite_name: str,
        model: str,
        timestamp: str,
        results: list[EvalResult],
        summary: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Persist an eval run and its results."""
        ...

    @abstractmethod
    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve a stored run by ID. Returns None if not found."""
        ...

    @abstractmethod
    async def list_runs(
        self,
        suite_name: str | None = None,
        model: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List stored runs with optional filtering."""
        ...

    @abstractmethod
    async def save_judgment(self, judgment: Judgment) -> None:
        """Persist a human or automated judgment."""
        ...

    @abstractmethod
    async def get_judgments(
        self,
        eval_id: str | None = None,
        judge: str | None = None,
    ) -> list[Judgment]:
        """Retrieve judgments with optional filtering."""
        ...

    @abstractmethod
    async def delete_run(self, run_id: str) -> bool:
        """Delete a run and its associated results. Returns True if found."""
        ...


# ---------------------------------------------------------------------------
# JSON File Backend
# ---------------------------------------------------------------------------


class JSONFileBackend(StorageBackend):
    """Stores eval runs as individual JSON files in a directory.

    Structure::

        storage_dir/
            runs/
                {run_id}.json       # full run record with results embedded
            judgments/
                {judgment_id}.json  # individual judgment files

    Args:
        storage_dir: Root directory for storing JSON files.
    """

    def __init__(self, storage_dir: str | Path) -> None:
        self._root = Path(storage_dir)
        self._runs_dir = self._root / "runs"
        self._judgments_dir = self._root / "judgments"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._judgments_dir.mkdir(parents=True, exist_ok=True)
        logger.info("JSONFileBackend initialised at '%s'", self._root)

    def _run_path(self, run_id: str) -> Path:
        return self._runs_dir / f"{run_id}.json"

    async def save_run(
        self,
        run_id: str,
        suite_name: str,
        model: str,
        timestamp: str,
        results: list[EvalResult],
        summary: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "id": run_id,
            "suite_name": suite_name,
            "model": model,
            "timestamp": timestamp,
            "summary": summary or {},
            "config": config or {},
            "results": [r.to_dict() for r in results],
        }
        path = self._run_path(run_id)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug("JSONFileBackend: saved run '%s' to '%s'", run_id, path)

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        path = self._run_path(run_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.debug("JSONFileBackend: loaded run '%s'", run_id)
        return data

    async def list_runs(
        self,
        suite_name: str | None = None,
        model: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        runs = []
        for path in sorted(self._runs_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("JSONFileBackend: failed to read '%s': %s", path, exc)
                continue
            if suite_name and data.get("suite_name") != suite_name:
                continue
            if model and data.get("model") != model:
                continue
            # Return a summary without the full result list
            runs.append({
                "id": data["id"],
                "suite_name": data.get("suite_name"),
                "model": data.get("model"),
                "timestamp": data.get("timestamp"),
                "summary": data.get("summary", {}),
                "n_results": len(data.get("results", [])),
            })
        return runs[offset: offset + limit]

    async def save_judgment(self, judgment: Judgment) -> None:
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **judgment.to_dict(),
        }
        path = self._judgments_dir / f"{record['id']}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug("JSONFileBackend: saved judgment '%s'", record["id"])

    async def get_judgments(
        self,
        eval_id: str | None = None,
        judge: str | None = None,
    ) -> list[Judgment]:
        judgments = []
        for path in sorted(self._judgments_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("JSONFileBackend: failed to read judgment '%s': %s", path, exc)
                continue
            if eval_id and data.get("eval_id") != eval_id:
                continue
            if judge and data.get("judge") != judge:
                continue
            judgments.append(Judgment.from_dict(data))
        return judgments

    async def delete_run(self, run_id: str) -> bool:
        path = self._run_path(run_id)
        if not path.exists():
            return False
        path.unlink()
        logger.debug("JSONFileBackend: deleted run '%s'", run_id)
        return True


# ---------------------------------------------------------------------------
# SQLite Backend
# ---------------------------------------------------------------------------


class SQLiteBackend(StorageBackend):
    """Stores eval runs using SQLAlchemy + SQLite (async via aiosqlite).

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._engine: Any = None
        self._session_factory: Any = None
        logger.info("SQLiteBackend initialised with db='%s'", db_path)

    async def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy.orm import sessionmaker
            from evalkit.storage.models import Base
        except ImportError as exc:
            raise ImportError(
                "SQLiteBackend requires sqlalchemy and aiosqlite. "
                "Install with: pip install sqlalchemy aiosqlite"
            ) from exc

        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{self._db_path}",
            echo=False,
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._session_factory = sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.debug("SQLiteBackend: engine and tables initialised")
        return self._engine

    async def save_run(
        self,
        run_id: str,
        suite_name: str,
        model: str,
        timestamp: str,
        results: list[EvalResult],
        summary: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        await self._get_engine()
        from evalkit.storage.models import EvalRun, EvalResultRow

        async with self._session_factory() as session:
            run = EvalRun(
                id=run_id,
                suite_name=suite_name,
                model=model,
                timestamp=timestamp,
                config_json=json.dumps(config or {}),
                summary_json=json.dumps(summary or {}),
            )
            session.add(run)

            for result in results:
                row = EvalResultRow(
                    run_id=run_id,
                    case_id=result.case.id,
                    response_json=json.dumps(result.response.to_dict()),
                    scores_json=json.dumps([s.to_dict() for s in result.scores]),
                    aggregate_score=result.aggregate_score,
                    timestamp=result.timestamp,
                )
                session.add(row)

            await session.commit()
        logger.debug("SQLiteBackend: saved run '%s' with %d results", run_id, len(results))

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        await self._get_engine()
        from sqlalchemy import select
        from evalkit.storage.models import EvalRun, EvalResultRow
        from evalkit.core.types import ModelResponse, Score, EvalCase

        async with self._session_factory() as session:
            run = await session.get(EvalRun, run_id)
            if run is None:
                return None

            stmt = select(EvalResultRow).where(EvalResultRow.run_id == run_id)
            result_rows = (await session.execute(stmt)).scalars().all()

            results = []
            for row in result_rows:
                results.append({
                    "case_id": row.case_id,
                    "response": json.loads(row.response_json),
                    "scores": json.loads(row.scores_json),
                    "aggregate_score": row.aggregate_score,
                    "timestamp": row.timestamp,
                })

            return {
                "id": run.id,
                "suite_name": run.suite_name,
                "model": run.model,
                "timestamp": run.timestamp,
                "config": json.loads(run.config_json or "{}"),
                "summary": json.loads(run.summary_json or "{}"),
                "results": results,
            }

    async def list_runs(
        self,
        suite_name: str | None = None,
        model: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        await self._get_engine()
        from sqlalchemy import select, func
        from evalkit.storage.models import EvalRun, EvalResultRow

        async with self._session_factory() as session:
            stmt = select(EvalRun)
            if suite_name:
                stmt = stmt.where(EvalRun.suite_name == suite_name)
            if model:
                stmt = stmt.where(EvalRun.model == model)
            stmt = stmt.order_by(EvalRun.timestamp.desc()).offset(offset).limit(limit)
            runs = (await session.execute(stmt)).scalars().all()

            result = []
            for run in runs:
                count_stmt = select(func.count()).where(EvalResultRow.run_id == run.id)
                n_results = (await session.execute(count_stmt)).scalar() or 0
                result.append({
                    "id": run.id,
                    "suite_name": run.suite_name,
                    "model": run.model,
                    "timestamp": run.timestamp,
                    "summary": json.loads(run.summary_json or "{}"),
                    "n_results": n_results,
                })
            return result

    async def save_judgment(self, judgment: Judgment) -> None:
        await self._get_engine()
        from evalkit.storage.models import HumanJudgment

        async with self._session_factory() as session:
            row = HumanJudgment(
                eval_id=judgment.eval_id,
                preferred=judgment.preferred,
                models_json=json.dumps(judgment.models),
                reason=judgment.reason,
                judge=judgment.judge,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            session.add(row)
            await session.commit()
        logger.debug("SQLiteBackend: saved judgment for eval_id='%s'", judgment.eval_id)

    async def get_judgments(
        self,
        eval_id: str | None = None,
        judge: str | None = None,
    ) -> list[Judgment]:
        await self._get_engine()
        from sqlalchemy import select
        from evalkit.storage.models import HumanJudgment

        async with self._session_factory() as session:
            stmt = select(HumanJudgment)
            if eval_id:
                stmt = stmt.where(HumanJudgment.eval_id == eval_id)
            if judge:
                stmt = stmt.where(HumanJudgment.judge == judge)
            rows = (await session.execute(stmt)).scalars().all()

            return [
                Judgment(
                    eval_id=row.eval_id,
                    preferred=row.preferred,
                    models=json.loads(row.models_json),
                    reason=row.reason,
                    judge=row.judge,
                )
                for row in rows
            ]

    async def delete_run(self, run_id: str) -> bool:
        await self._get_engine()
        from sqlalchemy import delete as sa_delete
        from evalkit.storage.models import EvalRun, EvalResultRow

        async with self._session_factory() as session:
            run = await session.get(EvalRun, run_id)
            if run is None:
                return False
            await session.delete(run)
            await session.commit()
        logger.debug("SQLiteBackend: deleted run '%s'", run_id)
        return True

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            logger.debug("SQLiteBackend: engine disposed")
