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


def _print_sweep_entries(runs: list[RunConfig]) -> None:
    console.print("[yellow]Sweep config detected. Use --entry to select one:[/yellow]")
    for i, r in enumerate(runs):
        console.print(f"  [{i}] {r.name}")


def _select_entry(runs: list[RunConfig], entry: str) -> RunConfig:
    try:
        idx = int(entry)
        if 0 <= idx < len(runs):
            return runs[idx]
        console.print(f"[red]--entry index {idx} out of range (0–{len(runs) - 1})[/red]")
        raise typer.Exit(1)
    except ValueError:
        pass
    matches = [r for r in runs if r.name == entry]
    if not matches:
        matches = [r for r in runs if entry in r.name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        console.print(f"[red]--entry '{entry}' matched no sweep entries[/red]")
    else:
        names = ", ".join(r.name for r in matches)
        console.print(f"[red]--entry '{entry}' matched multiple entries: {names}[/red]")
    raise typer.Exit(1)


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
    keep_jobs: bool = typer.Option(False, "--keep-jobs", help="Keep the jobs/ directory after evaluation (default: compress to jobs.tar.zst)"),
    delete_jobs: bool = typer.Option(False, "--delete-jobs", help="Delete jobs/ after evaluation instead of compressing"),
):
    """Run the full pipeline: download -> start server -> run agent -> evaluate."""
    from .config import RunConfig
    from .runner import _find_latest_sweep_dir, run_pipeline

    if resume_from and resume:
        console.print("[red]--resume-from and --resume are mutually exclusive[/red]")
        raise typer.Exit(1)

    if keep_jobs and delete_jobs:
        console.print("[red]--keep-jobs and --delete-jobs are mutually exclusive[/red]")
        raise typer.Exit(1)

    runs = _load_configs(config, local)
    if skip_eval:
        for r in runs:
            r.evaluation.run_evaluation = False

    if keep_jobs:
        for r in runs:
            r.jobs_cleanup = "none"
    elif delete_jobs:
        for r in runs:
            r.jobs_cleanup = "delete"

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
    entry: str = typer.Option(None, "--entry", "-e", help="Sweep entry to download: 0-based index or name substring"),
):
    """Download model(s) defined in the config without running any inference."""
    from .runner import make_backend

    runs = _load_configs(config, local)

    if len(runs) > 1:
        if entry is None:
            _print_sweep_entries(runs)
            raise typer.Exit(1)
        runs = [_select_entry(runs, entry)]

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
    entry: str = typer.Option(None, "--entry", "-e", help="Sweep entry to serve: 0-based index or name substring"),
):
    """Download model and start the backend server. Runs until Ctrl+C."""
    from .runner import serve_model

    runs = _load_configs(config, local)

    if len(runs) > 1:
        if entry is None:
            _print_sweep_entries(runs)
            raise typer.Exit(1)
        run_cfg = _select_entry(runs, entry)
    else:
        run_cfg = runs[0]

    serve_model(run_cfg, dry_run=dry_run)


def main():
    app()


if __name__ == "__main__":
    main()