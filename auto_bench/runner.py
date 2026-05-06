"""Top-level orchestration: download → start → agent → stop → evaluate."""
from __future__ import annotations

import json
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
from .downloader import is_gguf_cached, is_snapshot_cached, remove_from_cache
from .evaluator import collect_harbor_results, parse_results

console = Console()


def _download(backend: Backend) -> str:
    t0 = time.monotonic()
    model_path = backend.download()
    console.print(f"[dim]Download done in {time.monotonic() - t0:.1f}s[/dim]")
    return model_path


def _start_backend(backend: Backend, model_path: str, output_dir: Path) -> None:
    t0 = time.monotonic()
    backend.start(model_path, output_dir=output_dir)
    backend.wait_ready()
    time.sleep(5)
    backend.check_alive()
    console.print(f"[green]Backend ready[/green] in {time.monotonic() - t0:.1f}s — {backend.base_url}")


def make_backend(config: RunConfig) -> Backend:
    if config.backend.type == "llamacpp":
        return LlamaCppBackend(config.model, config.backend, config.sampling, config.backend_options)
    elif config.backend.type == "vllm":
        return VllmBackend(config.model, config.backend, config.sampling, config.backend_options)
    elif config.backend.type == "openai":
        return OpenAIBackend(config.model, config.backend, config.sampling, config.backend_options)
    else:
        raise ValueError(f"Unknown backend type: {config.backend.type}")


def _is_model_cached(config: RunConfig) -> bool:
    """Check whether the model in *config* is already in the local HF cache."""
    model = config.model
    if model.source == "local":
        return True  # local models are never removed
    if config.backend.type == "llamacpp":
        return is_gguf_cached(model.repo_id, model.filename, model.revision)
    elif config.backend.type == "vllm":
        return is_snapshot_cached(model.repo_id, model.revision)
    return True  # openai: no local model


def _is_entry_complete(entry_dir: Path) -> bool:
    """Return True if *entry_dir* has Harbor evaluation results."""
    jobs_dir = entry_dir / "jobs"
    return jobs_dir.is_dir() and bool(list(jobs_dir.rglob("verifier/reward.txt")))


def _find_completed_entries(sweep_dir: Path) -> set[str]:
    """Scan *sweep_dir* for completed entry directories and return their names."""
    completed: set[str] = set()
    if not sweep_dir.is_dir():
        return completed
    for child in sweep_dir.iterdir():
        if not child.is_dir():
            continue
        if _is_entry_complete(child):
            # Directory names are {name}_{YYYYMMDD_HHMMSS} — strip trailing timestamp
            name = "_".join(child.name.rsplit("_", 2)[:-2])
            completed.add(name)
    return completed


def _collect_previous_results(sweep_dir: Path) -> list[dict]:
    """Re-read evaluation results from already-completed entry directories."""
    results: list[dict] = []
    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir() or not _is_entry_complete(child):
            continue
        name = "_".join(child.name.rsplit("_", 2)[:-2])
        meta_file = child / "run_meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        results.append({
            "name": name,
            "results": collect_harbor_results(child / "jobs"),
            "output_dir": str(child),
            **meta,
        })
    return results


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

    was_cached = _is_model_cached(config) if config.remove_downloaded_models else True

    console.print("\n[bold]Step 1/3:[/bold] Downloading model...")
    model_path = _download(backend)

    cmd = backend.build_start_command(model_path)
    if dry_run:
        console.print(f"\n[cyan]Would run:[/cyan] {' '.join(cmd)}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold]Step 2/3:[/bold] Starting backend server...")
    try:
        _start_backend(backend, model_path, output_dir)
        console.print(
            f"\n[bold green]Server is running.[/bold green] Press Ctrl+C to stop.\n"
        )
        while True:
            backend.check_alive()
            time.sleep(2)
    except KeyboardInterrupt:
        console.print("\n[yellow]Received interrupt, shutting down...[/yellow]")
    finally:
        backend.stop()
        console.print("[green]Server stopped.[/green]")
        if config.remove_downloaded_models and not was_cached:
            remove_from_cache(model_path)


def run_single(config: RunConfig) -> dict:
    """
    Run a single (non-sweep) pipeline: download → start → agent → stop → evaluate.
    Returns result summary dict.
    """
    t_start = time.monotonic()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{config.name}_{timestamp}"
    output_dir = Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold blue]Run: {run_id}")

    backend = make_backend(config)

    was_cached = _is_model_cached(config) if config.remove_downloaded_models else True

    # 1. Download model
    console.print("\n[bold]Step 1/4:[/bold] Downloading model...")
    model_path = _download(backend)

    # 2. Start backend server
    console.print("\n[bold]Step 2/4:[/bold] Starting backend server...")
    _start_backend(backend, model_path, output_dir)

    # 2.5 Perplexity (optional)
    ppl: float | None = None
    if config.evaluation.perplexity.enabled:
        from .perplexity import compute_perplexity

        console.print("\n[bold]Computing perplexity...[/bold]")
        t0 = time.monotonic()
        ppl = compute_perplexity(backend, config.evaluation.perplexity)
        console.print(f"[green]Perplexity:[/green] {ppl:.2f} ({time.monotonic() - t0:.1f}s)")

    agent_output: Path | None = None
    results: dict = {}

    try:
        # 3. Run Harbor
        console.print("\n[bold]Step 3/4:[/bold] Running Harbor agent...")
        t0 = time.monotonic()
        agent_output = run_agent(config, backend, output_dir)
        console.print(f"[dim]Agent done in {time.monotonic()-t0:.1f}s[/dim]")
        backend.check_alive()

    finally:
        # 4. Stop backend (always, even on failure)
        console.print("\n[bold]Step 4/4:[/bold] Stopping backend server...")
        backend.stop()
        if config.remove_downloaded_models and not was_cached:
            remove_from_cache(model_path)

    # 5. Evaluate
    if config.evaluation.run_evaluation and agent_output:
        console.print("\n[bold]Collecting Harbor results...[/bold]")
        results = collect_harbor_results(agent_output)
        resolved, total, pct = parse_results(results)
        console.print(
            f"\n[bold green]Result:[/bold green] {resolved}/{total} resolved ({pct:.1f}%)"
        )
        if n_incomplete := results.get("n_incomplete", 0):
            console.print(
                f"\n[bold yellow]Warning:[/bold yellow] Harbor job is incomplete — "
                f"{n_incomplete} trial(s) never ran (job may have crashed)."
            )
        if exc_stats := results.get("exception_stats"):
            console.print("\n[bold]Exceptions:[/bold]")
            for exc_type, ids in sorted(exc_stats.items()):
                console.print(f"  [red]{exc_type}[/red]: {len(ids)} instance(s)")
                for iid in ids:
                    console.print(f"    [dim]- {iid}[/dim]")
    elif not config.evaluation.run_evaluation:
        console.print(f"\n[yellow]Evaluation skipped.[/yellow] Jobs: {agent_output}")

    total_runtime = time.monotonic() - t_start
    (output_dir / "run_meta.json").write_text(
        json.dumps({"perplexity": ppl, "total_runtime": total_runtime})
    )
    return {
        "run_id": run_id,
        "name": config.name,
        "agent_output": str(agent_output) if agent_output else None,
        "results": results,
        "output_dir": str(output_dir),
        "total_runtime": total_runtime,
        "perplexity": ppl,
    }


def run_pipeline(config: RunConfig, *, resume_from: Path | None = None) -> list[dict]:
    runs = expand_sweep(config)
    all_results: list[dict] = []
    is_sweep = len(runs) > 1

    if resume_from is not None:
        if not is_sweep:
            console.print("[yellow]--resume given but config is not a sweep; running as fresh.[/yellow]")
            sweep_dir = Path(config.output_dir)
        else:
            sweep_dir = resume_from
            completed = _find_completed_entries(sweep_dir)
            if completed:
                console.print(
                    f"[dim]Found {len(completed)} already-completed entries; skipping them.[/dim]"
                )
            all_results = _collect_previous_results(sweep_dir)
        runs_to_do = [r for r in runs if r.name not in {rr["name"] for rr in all_results}]
    else:
        if is_sweep:
            sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sweep_dir = Path(config.output_dir) / f"sweep_{config.name}_{sweep_timestamp}"
        else:
            sweep_dir = Path(config.output_dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        runs_to_do = runs

    for run_config in runs_to_do:
        run_config.output_dir = str(sweep_dir)

    total = len(runs_to_do) + len(all_results)
    for i, run_config in enumerate(runs_to_do):
        if is_sweep:
            done = len(all_results) + i
            console.rule(f"[bold magenta]Sweep {done + 1}/{total}: {run_config.name}")

        try:
            result = run_single(run_config)
            all_results.append(result)
        except Exception as exc:
            console.print(f"[red]Entry '{run_config.name}' failed: {exc}[/red]")
            raise SystemExit(1)

        if is_sweep:
            _write_sweep_summary_md(all_results, sweep_dir)

    if is_sweep:
        _print_sweep_summary(all_results)

    return all_results


def _fmt_runtime(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def _fmt_exceptions(results: dict) -> str:
    exc_stats = results.get("exception_stats", {})
    n_incomplete = results.get("n_incomplete", 0)
    parts = [f"{t}({len(ids)})" for t, ids in sorted(exc_stats.items())]
    if n_incomplete:
        parts.append(f"Incomplete({n_incomplete})")
    return ", ".join(parts)


def _write_sweep_summary_md(results: list[dict], sweep_dir: Path) -> None:
    """Write summary.md to *sweep_dir* from the current results list."""
    lines = [
        "# Sweep Summary\n",
        "| Run | Resolved | Total | % Resolved | PPL | Runtime | Exceptions | Error |",
        "|-----|----------|-------|------------|-----|---------|------------|-------|",
    ]
    for r in results:
        resolved, total, pct = parse_results(r.get("results", {}))
        runtime = _fmt_runtime(r.get("total_runtime"))
        exceptions = _fmt_exceptions(r.get("results", {}))
        error = r.get("error", "")
        ppl = f"{r['perplexity']:.2f}" if r.get("perplexity") else "—"
        lines.append(
            f"| {r['name']} | {resolved} | {total} | {pct:.1f}% | {ppl} | {runtime} | {exceptions} | {error} |"
        )
    (sweep_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _print_sweep_summary(results: list[dict]) -> None:
    console.rule("[bold]Sweep Summary")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Run", style="dim")
    table.add_column("Resolved", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("% Resolved", justify="right")
    table.add_column("PPL", justify="right")
    table.add_column("Exceptions", style="red")
    table.add_column("Runtime", justify="right")
    table.add_column("Error", style="red")

    for r in results:
        resolved, total, pct = parse_results(r.get("results", {}))
        name = r["name"]
        error = r.get("error", "")
        ppl = f"{r['perplexity']:.2f}" if r.get("perplexity") else "—"
        if error:
            name += " [red](failed)[/red]"
        table.add_row(
            name, str(resolved), str(total), f"{pct:.1f}%", ppl,
            _fmt_exceptions(r.get("results", {})),
            _fmt_runtime(r.get("total_runtime")), error,
        )

    console.print(table)

    if results:
        sweep_dir = Path(results[0]["output_dir"]).parent
        _write_sweep_summary_md(results, sweep_dir)
        console.print(f"[dim]Summary saved to {sweep_dir / 'summary.md'}[/dim]")
