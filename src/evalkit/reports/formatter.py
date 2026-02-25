"""ReportFormatter — generate markdown, HTML, and JSON reports."""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from typing import Any

from evalkit.core.types import ComparisonResult, EvalResult

logger = logging.getLogger(__name__)

_HTML_STYLE = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }
h1 { color: #1a1a2e; border-bottom: 3px solid #4a90e2; padding-bottom: 10px; }
h2 { color: #16213e; margin-top: 30px; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th { background: #4a90e2; color: white; padding: 10px 12px; text-align: left; }
td { padding: 8px 12px; border-bottom: 1px solid #eee; }
tr:hover { background: #f5f9ff; }
.score-high { color: #27ae60; font-weight: bold; }
.score-mid { color: #f39c12; }
.score-low { color: #e74c3c; }
.summary-box { background: #f0f4ff; border-left: 4px solid #4a90e2; padding: 15px; margin: 10px 0; border-radius: 4px; }
.metadata { color: #888; font-size: 0.9em; }
</style>
"""


def _score_class(value: float) -> str:
    if value >= 0.7:
        return "score-high"
    elif value >= 0.4:
        return "score-mid"
    return "score-low"


def _fmt(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


class ReportFormatter:
    """Generates formatted reports from eval results.

    Supports markdown, HTML, and JSON output formats.

    Args:
        title: Report title (defaults to auto-generated).
    """

    def __init__(self, title: str = "EvalKit Report") -> None:
        self._title = title

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format(
        self,
        results: dict[str, list[EvalResult]],
        comparison: ComparisonResult | None = None,
        fmt: str = "markdown",
    ) -> str:
        """Generate a report in the specified format.

        Args:
            results: Dict mapping model label to list of EvalResult.
            comparison: Optional pre-computed comparison result.
            fmt: One of "markdown", "html", "json".

        Returns:
            Report as a string.
        """
        fmt = fmt.lower()
        if fmt == "markdown":
            return self._markdown(results, comparison)
        elif fmt == "html":
            return self._html(results, comparison)
        elif fmt == "json":
            return self._json(results, comparison)
        else:
            raise ValueError(f"Unknown format '{fmt}'. Choose from: markdown, html, json.")

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _markdown(
        self,
        results: dict[str, list[EvalResult]],
        comparison: ComparisonResult | None,
    ) -> str:
        lines: list[str] = []
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines += [
            f"# {self._title}",
            f"",
            f"*Generated: {ts}*",
            f"",
        ]

        # Summary table
        lines += ["## Summary", "", "| Model | Cases | Mean Score | Median | P95 | Avg Latency (ms) |", "| --- | --- | --- | --- | --- | --- |"]
        for model, res_list in results.items():
            scores = [r.aggregate_score for r in res_list]
            latencies = [r.response.latency_ms for r in res_list]
            if scores:
                mean = statistics.mean(scores)
                median = statistics.median(scores)
                p95 = sorted(scores)[int(0.95 * len(scores))]
                avg_lat = statistics.mean(latencies)
            else:
                mean = median = p95 = avg_lat = 0.0
            lines.append(
                f"| {model} | {len(res_list)} | {_fmt(mean)} | {_fmt(median)} | {_fmt(p95)} | {avg_lat:.1f} |"
            )
        lines.append("")

        # Comparison section
        if comparison:
            lines += self._markdown_comparison(comparison)

        # Per-model breakdown
        for model, res_list in results.items():
            lines += [f"## Model: {model}", ""]
            lines += [
                "| Case ID | Score | Tags |",
                "| --- | --- | --- |",
            ]
            for r in sorted(res_list, key=lambda x: x.aggregate_score):
                tags = ", ".join(r.case.tags) if r.case.tags else ""
                lines.append(
                    f"| {r.case.id} | {_fmt(r.aggregate_score)} | {tags} |"
                )
            lines.append("")

            # Per-case details
            lines.append("<details><summary>Per-case details</summary>")
            lines.append("")
            for r in res_list:
                lines += [
                    f"### {r.case.id}",
                    "",
                    f"**Prompt:** {r.case.prompt[:300]}",
                    "",
                    f"**Response:** {r.response.text[:500]}",
                    "",
                ]
                if r.case.expected:
                    lines.append(f"**Expected:** {r.case.expected[:300]}")
                    lines.append("")
                lines += [
                    f"**Aggregate Score:** {_fmt(r.aggregate_score)}",
                    "",
                    "**Scorer Breakdown:**",
                ]
                for s in r.scores:
                    lines.append(
                        f"- `{s.scorer}`: {_fmt(s.value)}"
                        + (f" — {s.reasoning}" if s.reasoning else "")
                    )
                lines.append("")
            lines.append("</details>")
            lines.append("")

        return "\n".join(lines)

    def _markdown_comparison(self, comparison: ComparisonResult) -> list[str]:
        lines: list[str] = ["## Model Comparison", ""]

        # Win rates
        lines += ["### Win Rates", "", "| Model | Win Rate | Elo Rating |", "| --- | --- | --- |"]
        for model in sorted(comparison.models, key=lambda m: -comparison.win_rates.get(m, 0)):
            wr = comparison.win_rates.get(model, 0)
            elo = comparison.elo_ratings.get(model, 1500)
            lines.append(f"| {model} | {wr:.1%} | {elo:.1f} |")
        lines.append("")

        # Head-to-head
        if len(comparison.models) >= 2:
            lines += ["### Head-to-Head Win Rates", ""]
            header = "| Model A \\ Model B |" + "".join(f" {m} |" for m in comparison.models)
            separator = "| --- |" + " --- |" * len(comparison.models)
            lines += [header, separator]
            for ma in comparison.models:
                row = f"| **{ma}** |"
                for mb in comparison.models:
                    if ma == mb:
                        row += " — |"
                    else:
                        wr = comparison.head_to_head.get(ma, {}).get(mb, 0.0)
                        row += f" {wr:.1%} |"
                lines.append(row)
            lines.append("")

        # Score summaries
        lines += ["### Score Statistics", "", "| Model | Mean | Std | Median | P95 |", "| --- | --- | --- | --- | --- |"]
        for model, stats in comparison.score_summary.items():
            lines.append(
                f"| {model} | {_fmt(stats.get('mean', 0))} | {_fmt(stats.get('std', 0))} | "
                f"{_fmt(stats.get('median', 0))} | {_fmt(stats.get('p95', 0))} |"
            )
        lines.append("")
        return lines

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _html(
        self,
        results: dict[str, list[EvalResult]],
        comparison: ComparisonResult | None,
    ) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        body_parts: list[str] = []

        body_parts.append(f"<h1>{self._title}</h1>")
        body_parts.append(f'<p class="metadata">Generated: {ts}</p>')

        # Summary table
        body_parts.append("<h2>Summary</h2>")
        body_parts.append(
            "<table><tr><th>Model</th><th>Cases</th><th>Mean Score</th>"
            "<th>Median</th><th>P95</th><th>Avg Latency (ms)</th></tr>"
        )
        for model, res_list in results.items():
            scores = [r.aggregate_score for r in res_list]
            latencies = [r.response.latency_ms for r in res_list]
            if scores:
                mean = statistics.mean(scores)
                median = statistics.median(scores)
                p95 = sorted(scores)[int(0.95 * len(scores))]
                avg_lat = statistics.mean(latencies)
            else:
                mean = median = p95 = avg_lat = 0.0
            cls = _score_class(mean)
            body_parts.append(
                f"<tr><td>{model}</td><td>{len(res_list)}</td>"
                f'<td class="{cls}">{_fmt(mean)}</td>'
                f"<td>{_fmt(median)}</td><td>{_fmt(p95)}</td>"
                f"<td>{avg_lat:.1f}</td></tr>"
            )
        body_parts.append("</table>")

        if comparison:
            body_parts.append(self._html_comparison(comparison))

        # Per-model details
        for model, res_list in results.items():
            body_parts.append(f"<h2>Model: {model}</h2>")
            body_parts.append(
                "<table><tr><th>Case ID</th><th>Score</th><th>Tags</th></tr>"
            )
            for r in sorted(res_list, key=lambda x: x.aggregate_score):
                tags = ", ".join(r.case.tags) if r.case.tags else ""
                cls = _score_class(r.aggregate_score)
                body_parts.append(
                    f"<tr><td>{r.case.id}</td>"
                    f'<td class="{cls}">{_fmt(r.aggregate_score)}</td>'
                    f"<td>{tags}</td></tr>"
                )
            body_parts.append("</table>")

        body = "\n".join(body_parts)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self._title}</title>
{_HTML_STYLE}
</head>
<body>
{body}
</body>
</html>
"""

    def _html_comparison(self, comparison: ComparisonResult) -> str:
        rows = ""
        for model in sorted(comparison.models, key=lambda m: -comparison.win_rates.get(m, 0)):
            wr = comparison.win_rates.get(model, 0)
            elo = comparison.elo_ratings.get(model, 1500)
            rows += f"<tr><td>{model}</td><td>{wr:.1%}</td><td>{elo:.1f}</td></tr>"
        return (
            "<h2>Model Comparison</h2>"
            "<table><tr><th>Model</th><th>Win Rate</th><th>Elo Rating</th></tr>"
            f"{rows}</table>"
        )

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _json(
        self,
        results: dict[str, list[EvalResult]],
        comparison: ComparisonResult | None,
    ) -> str:
        payload: dict[str, Any] = {
            "title": self._title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "models": {},
        }

        for model, res_list in results.items():
            scores = [r.aggregate_score for r in res_list]
            payload["models"][model] = {
                "n_cases": len(res_list),
                "mean_score": statistics.mean(scores) if scores else 0.0,
                "results": [r.to_dict() for r in res_list],
            }

        if comparison:
            payload["comparison"] = comparison.to_dict()

        return json.dumps(payload, indent=2, ensure_ascii=False)
