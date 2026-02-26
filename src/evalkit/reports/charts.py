"""ChartGenerator — matplotlib charts for score distributions and comparisons."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from evalkit.core.types import ComparisonResult, EvalResult

logger = logging.getLogger(__name__)


def _require_matplotlib() -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for chart generation. "
            "Install with: pip install matplotlib"
        ) from exc


class ChartGenerator:
    """Generates matplotlib charts from eval results.

    All chart methods return a ``matplotlib.figure.Figure`` object.
    Use ``save(fig, path)`` to write to disk, or ``show(fig)`` to display
    interactively.

    Args:
        dpi: Dots per inch for saved figures.
        figsize: Default figure size (width, height) in inches.
        style: Matplotlib style name (e.g., "seaborn-v0_8", "ggplot").
    """

    def __init__(
        self,
        dpi: int = 150,
        figsize: tuple[float, float] = (10, 6),
        style: str = "default",
    ) -> None:
        self._dpi = dpi
        self._figsize = figsize
        self._style = style

    def _apply_style(self, plt: Any) -> None:
        try:
            plt.style.use(self._style)
        except Exception:
            pass  # Fall back to default if style not available

    def score_distribution(
        self,
        results: dict[str, list[EvalResult]],
        bins: int = 20,
        title: str = "Score Distribution",
    ) -> Any:
        """Histogram of aggregate scores for each model.

        Args:
            results: Dict mapping model label to list of EvalResult.
            bins: Number of histogram bins.
            title: Chart title.

        Returns:
            matplotlib Figure.
        """
        plt = _require_matplotlib()
        self._apply_style(plt)

        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)

        colors = plt.cm.tab10.colors  # type: ignore
        for idx, (model, res_list) in enumerate(results.items()):
            scores = [r.aggregate_score for r in res_list]
            ax.hist(
                scores,
                bins=bins,
                alpha=0.6,
                label=model,
                color=colors[idx % len(colors)],
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xlabel("Aggregate Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.legend(framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def radar_chart(
        self,
        results: dict[str, list[EvalResult]],
        title: str = "Multi-Dimension Comparison",
    ) -> Any:
        """Radar/spider plot of mean scores per scorer across models.

        Args:
            results: Dict mapping model label to list of EvalResult.
            title: Chart title.

        Returns:
            matplotlib Figure.
        """
        plt = _require_matplotlib()
        import numpy as np

        # Collect scorer names
        scorer_names: list[str] = []
        for res_list in results.values():
            for res in res_list:
                for s in res.scores:
                    if s.scorer not in scorer_names:
                        scorer_names.append(s.scorer)
        if not scorer_names:
            logger.warning("No scorer data available for radar chart.")
            fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)
            ax.text(0.5, 0.5, "No scorer data", ha="center", va="center", fontsize=14)
            return fig

        # Compute mean score per scorer per model
        model_scores: dict[str, list[float]] = {}
        for model, res_list in results.items():
            scorer_totals: dict[str, list[float]] = {name: [] for name in scorer_names}
            for res in res_list:
                for s in res.scores:
                    if s.scorer in scorer_totals:
                        scorer_totals[s.scorer].append(s.value)
            model_scores[model] = [
                (sum(scorer_totals[sn]) / len(scorer_totals[sn])) if scorer_totals[sn] else 0.0
                for sn in scorer_names
            ]

        N = len(scorer_names)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        self._apply_style(plt)
        fig, ax = plt.subplots(
            figsize=self._figsize,
            dpi=self._dpi,
            subplot_kw={"projection": "polar"},
        )

        colors = plt.cm.tab10.colors  # type: ignore
        for idx, (model, values) in enumerate(model_scores.items()):
            values_plot = values + values[:1]
            ax.plot(
                angles, values_plot,
                linewidth=2,
                linestyle="solid",
                label=model,
                color=colors[idx % len(colors)],
            )
            ax.fill(angles, values_plot, alpha=0.1, color=colors[idx % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(scorer_names, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()
        return fig

    def win_rate_heatmap(
        self,
        comparison: ComparisonResult,
        title: str = "Head-to-Head Win Rates",
    ) -> Any:
        """Heatmap of head-to-head win rates between models.

        Args:
            comparison: A ComparisonResult with head_to_head data.
            title: Chart title.

        Returns:
            matplotlib Figure.
        """
        plt = _require_matplotlib()
        import numpy as np

        models = comparison.models
        n = len(models)
        matrix = np.zeros((n, n))
        matrix[:] = float("nan")

        for i, ma in enumerate(models):
            for j, mb in enumerate(models):
                if ma != mb:
                    matrix[i][j] = comparison.head_to_head.get(ma, {}).get(mb, 0.0)

        self._apply_style(plt)
        fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.2)), dpi=self._dpi)

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Win Rate")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(models, fontsize=10)
        ax.set_xlabel("Opponent (Model B)", fontsize=11)
        ax.set_ylabel("Model A", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Annotate cells
        for i in range(n):
            for j in range(n):
                if not math.isnan(matrix[i][j]):
                    text = ax.text(
                        j, i,
                        f"{matrix[i][j]:.0%}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                    )

        fig.tight_layout()
        return fig

    def latency_vs_score(
        self,
        results: dict[str, list[EvalResult]],
        title: str = "Latency vs Score",
    ) -> Any:
        """Scatter plot of per-case latency against aggregate score.

        Args:
            results: Dict mapping model label to list of EvalResult.
            title: Chart title.

        Returns:
            matplotlib Figure.
        """
        plt = _require_matplotlib()
        self._apply_style(plt)

        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)
        colors = plt.cm.tab10.colors  # type: ignore

        for idx, (model, res_list) in enumerate(results.items()):
            latencies = [r.response.latency_ms for r in res_list]
            scores = [r.aggregate_score for r in res_list]
            ax.scatter(
                latencies,
                scores,
                alpha=0.7,
                label=model,
                color=colors[idx % len(colors)],
                s=60,
                edgecolors="white",
                linewidths=0.5,
            )

        ax.set_xlabel("Latency (ms)", fontsize=12)
        ax.set_ylabel("Aggregate Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(framealpha=0.9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def elo_leaderboard(
        self,
        comparison: ComparisonResult,
        title: str = "Elo Rating Leaderboard",
    ) -> Any:
        """Horizontal bar chart of Elo ratings.

        Args:
            comparison: A ComparisonResult with elo_ratings data.
            title: Chart title.

        Returns:
            matplotlib Figure.
        """
        plt = _require_matplotlib()
        self._apply_style(plt)

        models = sorted(comparison.elo_ratings, key=lambda m: comparison.elo_ratings[m])
        ratings = [comparison.elo_ratings[m] for m in models]

        fig, ax = plt.subplots(figsize=(8, max(3, len(models) * 0.8)), dpi=self._dpi)
        bars = ax.barh(models, ratings, color="#4a90e2", edgecolor="white", linewidth=0.5)

        # Annotate bars
        for bar, rating in zip(bars, ratings):
            ax.text(
                rating + 5,
                bar.get_y() + bar.get_height() / 2,
                f"{rating:.1f}",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Elo Rating", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.1)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    def save(self, fig: Any, path: str | Path, fmt: str = "png") -> Path:
        """Save a figure to disk.

        Args:
            fig: matplotlib Figure to save.
            path: Output file path.
            fmt: File format (png, svg, pdf).

        Returns:
            Absolute path to the saved file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), format=fmt, bbox_inches="tight", dpi=self._dpi)
        logger.info("ChartGenerator: saved chart to '%s'", out)
        return out.resolve()

    def show(self, fig: Any) -> None:
        """Display a figure interactively (requires a display)."""
        plt = _require_matplotlib()
        plt.figure(fig.number)
        plt.show()
