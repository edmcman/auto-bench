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


def _load_configs(config_path: Path, local_path: Path | None = None):
    from .config import ExperimentConfig, merge_configs

    experiment = _load_yaml(config_path, ExperimentConfig, "Experiment config error")
    return merge_configs(experiment, _load_local(local_path))


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to experiment config YAML", exists=True),
    local: Path = typer.Option(
        None, "--local", "-l",
        help="Path to local config (default: ~/.config/auto-bench/local.yaml)",
    ),
    skip_download: bool = typer.Option(False, "--skip-download", help="Skip model download step"),
    skip_eval: bool = typer.Option(False, "--skip-eval", help="Skip evaluation harness after inference"),
    resume_from: Path = typer.Option(
        None, "--resume-from",
        help="Resume a partially-completed sweep from this directory",
        exists=True, file_okay=False, dir_okay=True,
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume the most recent sweep in the output directory"),
):
    """Run the full pipeline: download -> start server -> run agent -> evaluate."""
    from .runner import _find_latest_sweep_dir, run_pipeline

    if resume_from and resume:
        console.print("[red]--resume-from and --resume are mutually exclusive[/red]")
        raise typer.Exit(1)

    cfg = _load_configs(config, local)
    if skip_eval:
        cfg.evaluation.run_evaluation = False

    if resume:
        resume_from = _find_latest_sweep_dir(Path(cfg.output_dir), cfg.name)
        if resume_from is None:
            console.print(f"[red]No sweep directories found in {cfg.output_dir}[/red]")
            raise typer.Exit(1)
        console.print(f"[dim]Resuming latest sweep: {resume_from}[/dim]")

    run_pipeline(cfg, resume_from=resume_from)


@app.command()
def download(
    config: Path = typer.Argument(..., help="Path to experiment config YAML", exists=True),
    local: Path = typer.Option(
        None, "--local", "-l",
        help="Path to local config (default: ~/.config/auto-bench/local.yaml)",
    ),
):
    """Download model(s) defined in the config without running any inference."""
    from .config import expand_sweep
    from .runner import make_backend

    cfg = _load_configs(config, local)
    runs = expand_sweep(cfg)

    for run_cfg in runs:
        backend = make_backend(run_cfg)
        console.print(f"\n[cyan]Downloading model for:[/cyan] {run_cfg.name}")
        path = backend.download()
        console.print(f"[green]Ready:[/green] {path}")


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to experiment config YAML", exists=True),
    local: Path = typer.Option(
        None, "--local", "-l",
        help="Path to local config (default: ~/.config/auto-bench/local.yaml)",
    ),
):
    """Validate a YAML config file without running anything."""
    from .config import expand_sweep

    cfg = _load_configs(config, local)
    runs = expand_sweep(cfg)
    console.print(f"[green]Config valid.[/green] {'Sweep: ' + str(len(runs)) + ' runs' if len(runs) > 1 else 'Single run: ' + cfg.name}")
    for r in runs:
        console.print(f"  * {r.name}  backend={r.backend.type}  model={r.model.effective_name()}")


@app.command()
def serve(
    config: Path = typer.Argument(..., help="Path to experiment config YAML", exists=True),
    local: Path = typer.Option(
        None, "--local", "-l",
        help="Path to local config (default: ~/.config/auto-bench/local.yaml)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Print the backend command and exit without running it"),
):
    """Download model and start the backend server. Runs until Ctrl+C."""
    from .config import expand_sweep
    from .runner import serve_model

    cfg = _load_configs(config, local)
    runs = expand_sweep(cfg)

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
