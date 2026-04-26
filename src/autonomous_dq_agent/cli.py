"""Typer CLI for the autonomous DQ agent."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autonomous_dq_agent import __version__
from autonomous_dq_agent.core.profiler import DataProfiler
from autonomous_dq_agent.core.reporter import QualityReporter
from autonomous_dq_agent.core.validator import DataValidator
from autonomous_dq_agent.services.ai_agent import ClaudeAIAgent

app = typer.Typer(
    name="dq-agent",
    help="Autonomous Data Quality Agent — profile, validate, and fix your data.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _load_dataframe(input_file: Path) -> pd.DataFrame:
    """Load a DataFrame from CSV, JSON, or Parquet based on file extension."""
    suffix = input_file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_file)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(input_file)
    if suffix == ".json":
        return pd.read_json(input_file)
    if suffix == ".jsonl":
        return pd.read_json(input_file, lines=True)
    raise ValueError(
        f"Unsupported file format: {suffix}. Use CSV, JSON, JSONL, or Parquet."
    )


@app.command()
def version() -> None:
    """Print the dq-agent version."""
    console.print(f"dq-agent [bold cyan]v{__version__}[/bold cyan]")


@app.command()
def profile(
    input_file: Path = typer.Argument(..., help="Input data file (CSV/JSON/Parquet)"),
    dataset_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Dataset logical name"
    ),
    output_html: Optional[Path] = typer.Option(
        None, "--html", help="Save HTML report to path"
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--json", help="Save JSON report to path"
    ),
    show_issues: bool = typer.Option(
        True, "--issues/--no-issues", help="Print quality issues"
    ),
) -> None:
    """Profile a data file and display a quality report."""
    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    name = dataset_name or input_file.stem
    with console.status(f"[bold green]Profiling {input_file.name}..."):
        df = _load_dataframe(input_file)
        profiler = DataProfiler()
        result = profiler.profile(df, dataset_name=name)

    score = result.overall_quality_score
    score_color = "green" if score >= 80 else "yellow" if score >= 50 else "red"
    console.print(
        Panel(
            f"[bold]Dataset:[/bold] {result.dataset_name}\n"
            f"[bold]Rows:[/bold] {result.row_count:,}  [bold]Columns:[/bold] {result.column_count}  "
            f"[bold]Memory:[/bold] {result.memory_mb:.2f} MB\n"
            f"[bold]Duplicates:[/bold] {result.duplicate_row_count:,} ({result.duplicate_row_rate:.2%})\n"
            f"[bold]Quality Score:[/bold] [{score_color}]{score:.1f}/100[/{score_color}]",
            title="[bold cyan]Data Profile Summary[/bold cyan]",
            border_style="cyan",
        )
    )

    col_table = Table(title="Column Profiles", box=box.ROUNDED, show_lines=False)
    col_table.add_column("Column", style="bold")
    col_table.add_column("Type")
    col_table.add_column("Null %", justify="right")
    col_table.add_column("Distinct", justify="right")
    col_table.add_column("Stats")
    for col_name, col in result.columns.items():
        null_str = f"{col.null_rate:.1%}"
        null_style = (
            "red"
            if col.null_rate > 0.2
            else "yellow" if col.null_rate > 0.05 else "green"
        )
        stats_str = ""
        if col.numeric_stats:
            ns = col.numeric_stats
            stats_str = f"mean={ns.mean:.2f} std={ns.std:.2f}"
        elif col.categorical_stats:
            cs = col.categorical_stats
            stats_str = f"top={cs.most_frequent!r}"
        col_table.add_row(
            col_name,
            col.column_type.value,
            f"[{null_style}]{null_str}[/{null_style}]",
            str(col.distinct_count),
            stats_str,
        )
    console.print(col_table)

    if show_issues and result.quality_issues:
        issue_table = Table(title="Quality Issues", box=box.ROUNDED, show_lines=True)
        issue_table.add_column("Sev", width=8)
        issue_table.add_column("Column")
        issue_table.add_column("Description")
        issue_table.add_column("Recommendation")
        sev_colors = {
            "critical": "red",
            "high": "orange3",
            "medium": "yellow",
            "low": "green",
            "info": "blue",
        }
        for issue in result.quality_issues:
            sev = issue.severity.value
            color = sev_colors.get(sev, "white")
            col_label = issue.column or "[dataset]"
            issue_table.add_row(
                f"[{color}]{sev.upper()}[/{color}]",
                col_label,
                issue.description,
                issue.recommendation,
            )
        console.print(issue_table)

    reporter = QualityReporter()
    if output_html:
        reporter.save_html_report(result, filename=output_html.name)
        console.print(f"[green]HTML report saved: {output_html}[/green]")
    if output_json:
        reporter.save_json_report(result, filename=output_json.name)
        console.print(f"[green]JSON report saved: {output_json}[/green]")


@app.command()
def suggest(
    input_file: Path = typer.Argument(..., help="Input data file"),
    dataset_name: Optional[str] = typer.Option(None, "--name", "-n"),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save suite JSON"
    ),
) -> None:
    """Generate an ExpectationSuite from a data file using AI or heuristics."""
    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    name = dataset_name or input_file.stem
    with console.status("[bold green]Profiling and generating expectations..."):
        df = _load_dataframe(input_file)
        profiler = DataProfiler()
        result = profiler.profile(df, dataset_name=name)
        agent = ClaudeAIAgent()
        suite = agent.suggest_expectations(result)

    mode = "Claude AI" if agent.is_ai_enabled else "Heuristic"
    console.print(
        Panel(
            f"[bold]Suite:[/bold] {suite.suite_name}\n"
            f"[bold]Mode:[/bold] {mode}\n"
            f"[bold]Expectations generated:[/bold] [cyan]{len(suite.expectations)}[/cyan]",
            title="[bold cyan]Expectation Suite[/bold cyan]",
            border_style="cyan",
        )
    )

    exp_table = Table(title="Generated Expectations", box=box.ROUNDED)
    exp_table.add_column("Type")
    exp_table.add_column("Column")
    exp_table.add_column("Parameters")
    for exp in suite.expectations:
        exp_table.add_row(
            exp.expectation_type.value,
            exp.column or "[table]",
            json.dumps(exp.kwargs, default=str)[:80],
        )
    console.print(exp_table)

    if output_file:
        output_file.write_text(suite.model_dump_json(indent=2))
        console.print(f"[green]Suite saved: {output_file}[/green]")


@app.command()
def validate(
    input_file: Path = typer.Argument(..., help="Input data file"),
    suite_file: Path = typer.Argument(..., help="Expectation suite JSON file"),
    dataset_name: Optional[str] = typer.Option(None, "--name", "-n"),
    fail_on_error: bool = typer.Option(
        False, "--fail/--no-fail", help="Exit 1 if validation fails"
    ),
) -> None:
    """Validate a data file against an ExpectationSuite."""
    from autonomous_dq_agent.models.validation import ExpectationSuite as Suite

    for path in [input_file, suite_file]:
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            raise typer.Exit(1)

    with console.status("[bold green]Validating..."):
        df = _load_dataframe(input_file)
        suite_data = json.loads(suite_file.read_text())
        suite = Suite.model_validate(suite_data)
        validator = DataValidator()
        result = validator.validate(df, suite)

    status_color = "green" if result.success else "red"
    status_label = "PASS" if result.success else "FAIL"
    console.print(
        Panel(
            f"[bold]Result:[/bold] [{status_color}]{status_label}[/{status_color}]\n"
            f"[bold]Passed:[/bold] {result.successful_expectations}/{result.evaluated_expectations} "
            f"({result.success_percent:.1f}%)",
            title="[bold cyan]Validation Result[/bold cyan]",
            border_style="cyan",
        )
    )

    if not result.success:
        fail_table = Table(title="Failed Expectations", box=box.ROUNDED)
        fail_table.add_column("Column")
        fail_table.add_column("Expectation")
        fail_table.add_column("Observed")
        for r in result.failed_results():
            fail_table.add_row(
                r.expectation.column or "[table]",
                r.expectation.expectation_type.value,
                str(r.observed_value or r.error_message or "—"),
            )
        console.print(fail_table)

    if fail_on_error and not result.success:
        raise typer.Exit(1)


@app.command(name="run-all")
def run_all(
    input_file: Path = typer.Argument(..., help="Input data file"),
    dataset_name: Optional[str] = typer.Option(None, "--name", "-n"),
    output_dir: Path = typer.Option(Path("reports"), "--output-dir", "-o"),
    fail_on_error: bool = typer.Option(False, "--fail/--no-fail"),
) -> None:
    """Profile then suggest expectations then validate then save reports (full pipeline)."""
    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    name = dataset_name or input_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[bold cyan]NEXUS Autonomous DQ Pipeline[/bold cyan] — {input_file.name}\n"
    )

    with console.status("[1/4] Profiling dataset..."):
        df = _load_dataframe(input_file)
        profiler = DataProfiler()
        profile_result = profiler.profile(df, dataset_name=name)

    console.print(
        f"[green]v[/green] Profile complete — "
        f"{profile_result.row_count:,} rows, score={profile_result.overall_quality_score:.1f}/100"
    )

    with console.status("[2/4] Generating expectation suite..."):
        agent = ClaudeAIAgent()
        suite = agent.suggest_expectations(profile_result)

    console.print(
        f"[green]v[/green] Suite generated — {len(suite.expectations)} expectations"
    )

    with console.status("[3/4] Validating against suite..."):
        validator = DataValidator()
        val_result = validator.validate(df, suite)

    status = "[green]PASS[/green]" if val_result.success else "[red]FAIL[/red]"
    console.print(
        f"[green]v[/green] Validation {status} — "
        f"{val_result.successful_expectations}/{val_result.evaluated_expectations} passed"
    )

    with console.status("[4/4] Saving reports..."):
        reporter = QualityReporter(output_dir=str(output_dir))
        html_path = reporter.save_html_report(profile_result)
        json_path = reporter.save_json_report(profile_result)
        suite_path = output_dir / f"{name}_suite.json"
        suite_path.write_text(suite.model_dump_json(indent=2))

    console.print("[green]v[/green] Reports saved:")
    console.print(f"   HTML:  {html_path}")
    console.print(f"   JSON:  {json_path}")
    console.print(f"   Suite: {suite_path}")

    if fail_on_error and not val_result.success:
        raise typer.Exit(1)


def main() -> None:
    """Entry point for the dq-agent CLI."""
    app()


if __name__ == "__main__":
    main()
