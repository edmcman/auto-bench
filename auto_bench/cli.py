"""CLI entry point for auto-bench."""
from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console

app = typer.Typer(
    name="auto-bench",
    help="Automated SWE-bench Verified runs against locally-hosted LLMs.",
    add_completion=False,
)
console = Console()


def _load_yaml(path: Path, config_class, label: str):
    try:
        return config_class.model_validate(yaml.safe_load(path.read_text()))
    except Exception as exc:
        console.print(f"[red]{label}:[/red] {exc}")
        raise typer.Exit(1)


def _load_local(local_path: Path | None) -> LocalConfig:
    from .config import LocalConfig

    if local_path is not None:
        return _load_yaml(local_path, LocalConfig, f"Local config error in {local_path}")

    default = Path.home() / ".config" / "auto-bench" / "local.yaml"
    if default.exists():
        return _load_yaml(default, LocalConfig, f"Local config error in {default}")

    return LocalConfig()


def _load_configs(config_path: Path, local_path: Path | None = None) -> list[RunConfig]:
    from .config import ExperimentConfig, load_experiment_configs, merge_configs

    try:
        experiments = load_experiment_configs(config_path)
    except ValueError as exc:
        console.print(f"[red]Experiment config error:[/red] {exc}")
        raise typer.Exit(1)

    local = _load_local(local_path)
    return [merge_configs(exp, local) for exp in experiments]


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to experiment config jsonnet", exists=True),
    local: Path = typer.Option(
        None, "--local", "-l",
        help="Path to local config (default: ~/.config/auto-bench/local.yaml)",
    ),
    skip_download: bool = typer.Option(False, "--skip-download", help="Skip model download step"),
    skip_eval: bool = typer.Option(False, "--skip-eval", help="Skip evaluation harness after inference"),
    parallel: int = typer.Option(None, "--parallel", "-p", help="Override parallelism (agent trials + backend slots)"),
    resume_from: Path = typer.Option(
        None, "--resume-from",
        help="Resume a partially-completed sweep from this directory",
        exists=True, file_okay=False, dir_okay=True,
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume the most recent sweep in the output directory"),
):
    """Run the full pipeline: download -> start server -> run agent -> evaluate."""
    from .config import RunConfig
    from .runner import _find_latest_sweep_dir, run_pipeline

    if resume_from and resume:
        console.print("[red]--resume-from and --resume are mutually exclusive[/red]")
        raise typer.Exit(1)

    runs = _load_configs(config, local)
    if skip_eval:
        for r in runs:
            r.evaluation.run_evaluation = False

    if parallel is not None:
        for r in runs:
            r.agent.trials = parallel
            r.backend_options.parallel = parallel

    sweep_name = config.stem

    if resume:
        resume_from = _find_latest_sweep_dir(Path(runs[0].output_dir), sweep_name)
        if resume_from is None:
            console.print(f"[red]No sweep directories found in {runs[0].output_dir}[/red]")
            raise typer.Exit(1)
        console.print(f"[dim]Resuming latest sweep: {resume_from}[/dim]")

    run_pipeline(runs, sweep_name=sweep_name, resume_from=resume_from)


@app.command()
def download(
    config: Path = typer.Argument(..., help="Path to experiment config jsonnet", exists=True),
    local: Path = typer.Option(
        None, "--local", "-l",
        help="Path to local config (default: ~/.config/auto-bench/local.yaml)",
    ),
):
    """Download model(s) defined in the config without running any inference."""
    from .runner import make_backend

    runs = _load_configs(config, local)
    for run_cfg in runs:
        backend = make_backend(run_cfg)
        console.print(f"\n[cyan]Downloading model for:[/cyan] {run_cfg.name}")
        path = backend.download()
        console.print(f"[green]Ready:[/green] {path}")


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to experiment config jsonnet", exists=True),
    local: Path = typer.Option(
        None, "--local", "-l",
        help="Path to local config (default: ~/.config/auto-bench/local.yaml)",
    ),
):
    """Validate a config file without running anything."""
    runs = _load_configs(config, local)
    n = len(runs)
    label = f"Sweep: {n} runs" if n > 1 else f"Single run: {runs[0].name}"
    console.print(f"[green]Config valid.[/green] {label}")
    for r in runs:
        console.print(f"  * {r.name}  backend={r.backend.type}  model={r.model.effective_name()}")


@app.command()
def serve(
    config: Path = typer.Argument(..., help="Path to experiment config jsonnet", exists=True),
    local: Path = typer.Option(
        None, "--local", "-l",
        help="Path to local config (default: ~/.config/auto-bench/local.yaml)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Print the backend command and exit without running it"),
):
    """Download model and start the backend server. Runs until Ctrl+C."""
    from .runner import serve_model

    runs = _load_configs(config, local)

    if len(runs) > 1:
        console.print(
            f"[yellow]Sweep config with {len(runs)} entries detected. "
            f"Serving only the first: [bold]{runs[0].name}[/bold][/yellow]"
        )

    serve_model(runs[0], dry_run=dry_run)


def main():
    app()


if __name__ == "__main__":
    main()