"""CLI entry point for auto-swe-bench."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="auto-swe-bench",
    help="Automated SWE-bench Verified runs against locally-hosted LLMs.",
    add_completion=False,
)
console = Console()


def _load(config_path: Path):
    from .config import load_config
    try:
        return load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to YAML config file", exists=True),
    skip_download: bool = typer.Option(False, "--skip-download", help="Skip model download step"),
    skip_eval: bool = typer.Option(False, "--skip-eval", help="Skip evaluation harness after inference"),
):
    """Run the full pipeline: download → start server → run agent → evaluate."""
    from .runner import run_pipeline

    cfg = _load(config)
    if skip_eval:
        cfg.evaluation.run_evaluation = False

    run_pipeline(cfg)


@app.command()
def download(
    config: Path = typer.Argument(..., help="Path to YAML config file", exists=True),
):
    """Download model(s) defined in the config without running any inference."""
    from .config import expand_sweep
    from .runner import make_backend

    cfg = _load(config)
    runs = expand_sweep(cfg)

    for run_cfg in runs:
        backend = make_backend(run_cfg)
        console.print(f"\n[cyan]Downloading model for:[/cyan] {run_cfg.name}")
        path = backend.download()
        console.print(f"[green]Ready:[/green] {path}")


@app.command()
def evaluate(
    config: Path = typer.Argument(..., help="Path to YAML config file", exists=True),
    predictions: Path = typer.Argument(..., help="Path to predictions.jsonl", exists=True),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run ID for output naming"),
):
    """Run only the SWE-bench evaluation harness on an existing predictions file."""
    from .evaluator import parse_results, run_evaluation

    cfg = _load(config)
    rid = run_id or predictions.stem
    results = run_evaluation(
        predictions_path=predictions,
        run_id=rid,
        config=cfg.evaluation,
        dataset=cfg.dataset,
        output_dir=predictions.parent,
    )
    resolved, total, pct = parse_results(results)
    console.print(f"\n[bold green]Result:[/bold green] {resolved}/{total} resolved ({pct:.1f}%)")


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to YAML config file", exists=True),
):
    """Validate a YAML config file without running anything."""
    from .config import expand_sweep

    cfg = _load(config)
    runs = expand_sweep(cfg)
    console.print(f"[green]Config valid.[/green] {'Sweep: ' + str(len(runs)) + ' runs' if len(runs) > 1 else 'Single run: ' + cfg.name}")
    for r in runs:
        console.print(f"  • {r.name}  backend={r.backend.type}  model={r.model.effective_name()}")


def main():
    app()


if __name__ == "__main__":
    main()
