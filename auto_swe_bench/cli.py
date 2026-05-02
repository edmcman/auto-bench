"""CLI entry point for auto-swe-bench."""
from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console

app = typer.Typer(
    name="auto-swe-bench",
    help="Automated SWE-bench Verified runs against locally-hosted LLMs.",
    add_completion=False,
)
console = Console()


def _load_local(local_path: Path | None) -> LocalConfig:
    """Load local config: explicit --local, default file, or built-in defaults."""
    from .config import LocalConfig

    if local_path is not None:
        try:
            data = yaml.safe_load(local_path.read_text())
            return LocalConfig.model_validate(data)
        except Exception as exc:
            console.print(f"[red]Local config error in {local_path}:[/red] {exc}")
            raise typer.Exit(1)

    default = Path.home() / ".config" / "auto-swe-bench" / "local.yaml"
    if default.exists():
        try:
            data = yaml.safe_load(default.read_text())
            return LocalConfig.model_validate(data)
        except Exception as exc:
            console.print(f"[red]Local config error in {default}:[/red] {exc}")
            raise typer.Exit(1)

    return LocalConfig()


def _load_configs(config_path: Path, local_path: Path | None = None):
    """Load experiment + local configs and merge into a RunConfig."""
    from .config import ExperimentConfig, merge_configs

    data = yaml.safe_load(config_path.read_text())
    try:
        experiment = ExperimentConfig.model_validate(data)
    except Exception as exc:
        console.print(f"[red]Experiment config error:[/red] {exc}")
        raise typer.Exit(1)

    return merge_configs(experiment, _load_local(local_path))


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to experiment config YAML", exists=True),
    local: Path = typer.Option(
        None, "--local", "-l",
        help="Path to local config (default: ~/.config/auto-swe-bench/local.yaml)",
    ),
    skip_download: bool = typer.Option(False, "--skip-download", help="Skip model download step"),
    skip_eval: bool = typer.Option(False, "--skip-eval", help="Skip evaluation harness after inference"),
):
    """Run the full pipeline: download -> start server -> run agent -> evaluate."""
    from .runner import run_pipeline

    cfg = _load_configs(config, local)
    if skip_eval:
        cfg.evaluation.run_evaluation = False

    run_pipeline(cfg)


@app.command()
def download(
    config: Path = typer.Argument(..., help="Path to experiment config YAML", exists=True),
    local: Path = typer.Option(
        None, "--local", "-l",
        help="Path to local config (default: ~/.config/auto-swe-bench/local.yaml)",
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
        help="Path to local config (default: ~/.config/auto-swe-bench/local.yaml)",
    ),
):
    """Validate a YAML config file without running anything."""
    from .config import expand_sweep

    cfg = _load_configs(config, local)
    runs = expand_sweep(cfg)
    console.print(f"[green]Config valid.[/green] {'Sweep: ' + str(len(runs)) + ' runs' if len(runs) > 1 else 'Single run: ' + cfg.name}")
    for r in runs:
        console.print(f"  * {r.name}  backend={r.backend.type}  model={r.model.effective_name()}")


def main():
    app()


if __name__ == "__main__":
    main()
