"""HumanPreferenceCollector — interactive CLI A/B preference collection."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from evalkit.core.types import EvalResult, Judgment

logger = logging.getLogger(__name__)


class HumanPreferenceCollector:
    """Interactive CLI tool for collecting human A/B preferences.

    Displays two model outputs side-by-side using rich panels and prompts
    the user to choose A, B, or Tie. Records Judgment objects that can be
    used to update Elo ratings.

    Args:
        show_prompt: Whether to show the original prompt in the comparison.
        show_expected: Whether to show the expected answer (if available).
        judge_id: Identifier for the human judge (default: "human").
    """

    def __init__(
        self,
        show_prompt: bool = True,
        show_expected: bool = False,
        judge_id: str = "human",
    ) -> None:
        self._show_prompt = show_prompt
        self._show_expected = show_expected
        self._judge_id = judge_id
        self._judgments: list[Judgment] = []

    def _try_rich(self) -> bool:
        try:
            import rich  # noqa: F401
            return True
        except ImportError:
            return False

    def _display_rich(
        self,
        result_a: EvalResult,
        result_b: EvalResult,
        label_a: str,
        label_b: str,
        case_num: int,
        total: int,
    ) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.text import Text

        console = Console()
        console.print(
            f"\n[bold]Case {case_num}/{total}[/bold] — [cyan]{result_a.case.id}[/cyan]",
            justify="center",
        )

        if self._show_prompt:
            console.print(
                Panel(result_a.case.prompt, title="Prompt", border_style="dim"),
            )

        if self._show_expected and result_a.case.expected:
            console.print(
                Panel(
                    result_a.case.expected,
                    title="Reference Answer",
                    border_style="dim yellow",
                )
            )

        panel_a = Panel(
            result_a.response.text,
            title=f"[bold blue]A — {label_a}[/bold blue]",
            border_style="blue",
        )
        panel_b = Panel(
            result_b.response.text,
            title=f"[bold green]B — {label_b}[/bold green]",
            border_style="green",
        )
        console.print(Columns([panel_a, panel_b]))

        # Show scores if available
        score_a = result_a.aggregate_score
        score_b = result_b.aggregate_score
        console.print(
            f"[dim]Automated scores — A: {score_a:.3f}  B: {score_b:.3f}[/dim]"
        )

    def _display_plain(
        self,
        result_a: EvalResult,
        result_b: EvalResult,
        label_a: str,
        label_b: str,
        case_num: int,
        total: int,
    ) -> None:
        print(f"\n=== Case {case_num}/{total}: {result_a.case.id} ===")
        if self._show_prompt:
            print(f"Prompt: {result_a.case.prompt[:200]}")
        print(f"\n--- A: {label_a} ---")
        print(result_a.response.text[:500])
        print(f"\n--- B: {label_b} ---")
        print(result_b.response.text[:500])
        print(f"\nScores — A: {result_a.aggregate_score:.3f}  B: {result_b.aggregate_score:.3f}")

    def _prompt_choice(self, use_rich: bool = True) -> str:
        """Prompt user for A/B/T/skip choice. Returns 'A', 'B', 'T', or 'S'."""
        if use_rich:
            try:
                from rich.prompt import Prompt

                choice = Prompt.ask(
                    "Your preference",
                    choices=["A", "B", "T", "S", "a", "b", "t", "s"],
                    default="S",
                )
                return choice.upper()
            except Exception:
                pass

        while True:
            raw = input("Your preference [A/B/T=tie/S=skip]: ").strip().upper()
            if raw in {"A", "B", "T", "S", ""}:
                return raw or "S"
            print("Please enter A, B, T, or S.")

    def collect(
        self,
        results_a: list[EvalResult],
        results_b: list[EvalResult],
        label_a: str,
        label_b: str,
    ) -> list[Judgment]:
        """Run interactive preference collection session.

        Args:
            results_a: EvalResults for model A (must be same suite as results_b).
            results_b: EvalResults for model B.
            label_a: Display name for model A.
            label_b: Display name for model B.

        Returns:
            List of Judgment objects collected during the session.
        """
        use_rich = self._try_rich()

        # Build case-id index for alignment
        index_b = {r.case.id: r for r in results_b}
        paired = [(r, index_b[r.case.id]) for r in results_a if r.case.id in index_b]

        if not paired:
            logger.warning("HumanPreferenceCollector: no shared cases between the two result sets.")
            return []

        session_judgments: list[Judgment] = []
        total = len(paired)

        if use_rich:
            from rich.console import Console
            Console().print(
                f"[bold]Starting preference collection — {total} cases[/bold]\n"
                f"Models: [blue]{label_a}[/blue] (A) vs [green]{label_b}[/green] (B)\n"
                "[dim]Commands: A=prefer A, B=prefer B, T=tie, S=skip[/dim]"
            )
        else:
            print(f"\nStarting preference collection — {total} cases")
            print(f"A: {label_a}  |  B: {label_b}  |  Commands: A/B/T/S")

        for idx, (res_a, res_b) in enumerate(paired, 1):
            if use_rich:
                self._display_rich(res_a, res_b, label_a, label_b, idx, total)
            else:
                self._display_plain(res_a, res_b, label_a, label_b, idx, total)

            try:
                choice = self._prompt_choice(use_rich)
            except (KeyboardInterrupt, EOFError):
                print("\nSession interrupted.")
                break

            if choice == "S":
                logger.debug("HumanPreferenceCollector: case %s skipped", res_a.case.id)
                continue

            preferred = label_a if choice == "A" else (label_b if choice == "B" else "tie")
            judgment = Judgment(
                eval_id=res_a.case.id,
                preferred=preferred,
                models=[label_a, label_b],
                reason=None,
                judge=self._judge_id,
            )
            session_judgments.append(judgment)
            self._judgments.append(judgment)

            logger.debug(
                "HumanPreferenceCollector: case '%s' — preferred='%s'",
                res_a.case.id,
                preferred,
            )

        if use_rich:
            from rich.console import Console
            Console().print(
                f"\n[bold green]Session complete![/bold green] "
                f"{len(session_judgments)}/{total} cases judged."
            )
        else:
            print(f"\nSession complete: {len(session_judgments)}/{total} cases judged.")

        return session_judgments

    @property
    def all_judgments(self) -> list[Judgment]:
        """All judgments collected across all sessions."""
        return list(self._judgments)

    def summary(self) -> dict[str, Any]:
        """Return a summary of collected preferences."""
        from collections import Counter
        preferred_counts = Counter(j.preferred for j in self._judgments)
        return {
            "total_judgments": len(self._judgments),
            "preferred_counts": dict(preferred_counts),
        }
