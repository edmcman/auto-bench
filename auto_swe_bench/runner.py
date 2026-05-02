"""Top-level orchestration: download → start → agent → stop → evaluate."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .agent import run_agent
from .backends.base import Backend
from .backends.llamacpp import LlamaCppBackend
from .backends.openai_backend import OpenAIBackend
from .backends.vllm import VllmBackend
from .config import RunConfig, expand_sweep
from .evaluator import collect_harbor_results, parse_results

console = Console()


def make_backend(config: RunConfig) -> Backend:
    if config.backend.type == "llamacpp":
        return LlamaCppBackend(config.model, config.backend, config.sampling, config.backend_options)
    elif config.backend.type == "vllm":
        return VllmBackend(config.model, config.backend, config.sampling, config.backend_options)
    elif config.backend.type == "openai":
        return OpenAIBackend(config.model, config.backend, config.sampling, config.backend_options)
    else:
        raise ValueError(f"Unknown backend type: {config.backend.type}")


def serve_model(config: RunConfig, dry_run: bool = False) -> None:
    """Download model, start backend server, print URL, and block until Ctrl+C.

    If dry_run is True, print the backend command and exit without starting.
    """
    if config.backend.type == "openai":
        console.print(
            "[yellow]Backend type is 'openai' -- this is an external server. "
            "Nothing to start locally.[/yellow]"
        )
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{config.name}_{timestamp}"
    output_dir = Path(config.output_dir) / run_id

    console.rule(f"[bold blue]Serve: {run_id}")
    backend = make_backend(config)

    console.print("\n[bold]Step 1/3:[/bold] Downloading model...")
    t0 = time.monotonic()
    model_path = backend.download()
    console.print(f"[dim]Download done in {time.monotonic() - t0:.1f}s[/dim]")

    cmd = backend.build_start_command(model_path)
    if dry_run:
        console.print(f"\n[cyan]Would run:[/cyan] {' '.join(cmd)}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold]Step 2/3:[/bold] Starting backend server...")
    t0 = time.monotonic()
    backend.start(model_path, output_dir=output_dir)

    try:
        backend.wait_ready()
        console.print(
            f"[green]Backend ready[/green] in {time.monotonic() - t0:.1f}s — {backend.base_url}"
        )
        console.print(
            f"\n[bold green]Server is running.[/bold green] Press Ctrl+C to stop.\n"
        )
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Received interrupt, shutting down...[/yellow]")
    finally:
        backend.stop()
        console.print("[green]Server stopped.[/green]")


def run_single(config: RunConfig) -> dict:
    """
    Run a single (non-sweep) pipeline: download → start → agent → stop → evaluate.
    Returns result summary dict.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{config.name}_{timestamp}"
    output_dir = Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold blue]Run: {run_id}")

    backend = make_backend(config)

    # 1. Download model
    console.print("\n[bold]Step 1/4:[/bold] Downloading model...")
    t0 = time.monotonic()
    model_path = backend.download()
    console.print(f"[dim]Download done in {time.monotonic()-t0:.1f}s[/dim]")

    # 2. Start backend server
    console.print("\n[bold]Step 2/4:[/bold] Starting backend server...")
    t0 = time.monotonic()
    backend.start(model_path, output_dir=output_dir)
    backend.wait_ready()
    console.print(f"[green]Backend ready[/green] in {time.monotonic()-t0:.1f}s — {backend.base_url}")

    agent_output: Path | None = None
    results: dict = {}

    try:
        # 3. Run Harbor
        console.print("\n[bold]Step 3/4:[/bold] Running Harbor agent...")
        t0 = time.monotonic()
        agent_output = run_agent(config, backend, output_dir)
        console.print(f"[dim]Agent done in {time.monotonic()-t0:.1f}s[/dim]")

    finally:
        # 4. Stop backend (always, even on failure)
        console.print("\n[bold]Step 4/4:[/bold] Stopping backend server...")
        backend.stop()

    # 5. Evaluate
    if config.evaluation.run_evaluation and agent_output:
        console.print("\n[bold]Collecting Harbor results...[/bold]")
        results = collect_harbor_results(agent_output)
        resolved, total, pct = parse_results(results)
        console.print(
            f"\n[bold green]Result:[/bold green] {resolved}/{total} resolved ({pct:.1f}%)"
        )
    elif not config.evaluation.run_evaluation:
        console.print(f"\n[yellow]Evaluation skipped.[/yellow] Jobs: {agent_output}")

    return {
        "run_id": run_id,
        "name": config.name,
        "agent_output": str(agent_output) if agent_output else None,
        "results": results,
        "output_dir": str(output_dir),
    }


def run_pipeline(config: RunConfig) -> list[dict]:
    """
    Entry point for running a config (potentially a sweep).
    Returns list of result dicts (one per sweep entry, or one for a plain run).
    """
    runs = expand_sweep(config)
    all_results: list[dict] = []

    if len(runs) > 1:
        sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = Path(config.output_dir) / f"sweep_{sweep_timestamp}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for run_config in runs:
            run_config.output_dir = str(sweep_dir)

    for i, run_config in enumerate(runs):
        if len(runs) > 1:
            console.rule(f"[bold magenta]Sweep {i+1}/{len(runs)}: {run_config.name}")
        result = run_single(run_config)
        all_results.append(result)

    if len(runs) > 1:
        _print_sweep_summary(all_results)

    return all_results


def _print_sweep_summary(results: list[dict]) -> None:
    console.rule("[bold]Sweep Summary")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Run", style="dim")
    table.add_column("Resolved", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("% Resolved", justify="right")

    for r in results:
        resolved, total, pct = parse_results(r.get("results", {}))
        table.add_row(
            r["name"],
            str(resolved),
            str(total),
            f"{pct:.1f}%",
        )

    console.print(table)

    # Save summary markdown
    if results:
        summary_path = Path(results[0]["output_dir"]).parent / "summary.md"
        lines = [
            "# Sweep Summary\n",
            "| Run | Resolved | Total | % Resolved |",
            "|-----|----------|-------|------------|",
        ]
        for r in results:
            resolved, total, pct = parse_results(r.get("results", {}))
            lines.append(f"| {r['name']} | {resolved} | {total} | {pct:.1f}% |")
        summary_path.write_text("\n".join(lines) + "\n")
        console.print(f"[dim]Summary saved to {summary_path}[/dim]")
