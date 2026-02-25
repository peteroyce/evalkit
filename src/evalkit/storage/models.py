"""SQLAlchemy ORM models for evalkit storage."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class EvalRun(Base):
    """Top-level record for a single model eval run against a suite."""

    __tablename__ = "eval_runs"

    id = Column(String(128), primary_key=True)
    suite_name = Column(String(256), nullable=False, index=True)
    model = Column(String(256), nullable=False, index=True)
    timestamp = Column(String(64), nullable=False, index=True)
    config_json = Column(Text, nullable=True)
    summary_json = Column(Text, nullable=True)

    results = relationship(
        "EvalResultRow",
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="select",
    )

    __table_args__ = (
        Index("ix_eval_runs_suite_model", "suite_name", "model"),
    )

    def __repr__(self) -> str:
        return f"<EvalRun id={self.id!r} model={self.model!r} suite={self.suite_name!r}>"


class EvalResultRow(Base):
    """Stores a single EvalResult (response + scores) for a run."""

    __tablename__ = "eval_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        String(128),
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    case_id = Column(String(256), nullable=False, index=True)
    response_json = Column(Text, nullable=False)
    scores_json = Column(Text, nullable=False)
    aggregate_score = Column(Float, nullable=False)
    timestamp = Column(String(64), nullable=False)

    run = relationship("EvalRun", back_populates="results")

    __table_args__ = (
        Index("ix_eval_results_run_case", "run_id", "case_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<EvalResultRow id={self.id} run_id={self.run_id!r} "
            f"case_id={self.case_id!r} score={self.aggregate_score:.3f}>"
        )


class HumanJudgment(Base):
    """Stores a single human pairwise preference judgment."""

    __tablename__ = "human_judgments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    eval_id = Column(String(256), nullable=False, index=True)  # case id
    preferred = Column(String(256), nullable=False)
    models_json = Column(Text, nullable=False)  # JSON list of model names
    reason = Column(Text, nullable=True)
    judge = Column(String(256), nullable=False, default="human")
    timestamp = Column(String(64), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<HumanJudgment id={self.id} eval_id={self.eval_id!r} "
            f"preferred={self.preferred!r} judge={self.judge!r}>"
        )
