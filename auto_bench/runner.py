"""Top-level orchestration: download → start → agent → stop → evaluate."""
from __future__ import annotations

import csv
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
from .config import RunConfig
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
    """Return True if *entry_dir* has a run_meta.json (run_single finished successfully)."""
    return (entry_dir / "run_meta.json").exists()


def _parse_entry_name(dir_name: str) -> str | None:
    """Strip _YYYYMMDD_HHMMSS suffix from directory name. Returns None if unparseable."""
    parts = dir_name.rsplit("_", 2)
    if len(parts) < 3:
        return None
    return parts[0]


def _find_completed_entries(sweep_dir: Path) -> set[str]:
    """Scan *sweep_dir* for completed entry directories and return their names."""
    completed: set[str] = set()
    if not sweep_dir.is_dir():
        return completed
    for child in sweep_dir.iterdir():
        if not child.is_dir():
            continue
        if _is_entry_complete(child):
            name = _parse_entry_name(child.name)
            if name:
                completed.add(name)
    return completed


def _collect_previous_results(sweep_dir: Path) -> list[dict]:
    """Re-read evaluation results from already-completed entry directories."""
    results: list[dict] = []
    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir() or not _is_entry_complete(child):
            continue
        name = _parse_entry_name(child.name)
        if name is None:
            continue
        meta_file = child / "run_meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        results.append({
            "name": name,
            "results": collect_harbor_results(child / "jobs"),
            "output_dir": str(child),
            **meta,
        })
    return results


def _find_latest_sweep_dir(output_dir: Path, name: str | None = None) -> Path | None:
    """Return the most recent sweep_* directory under output_dir, or None."""
    prefix = f"sweep_{name}_" if name else "sweep_"
    dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    return max(dirs, key=lambda d: d.name) if dirs else None


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


def run_single(
    config: RunConfig,
    logits_save: Path | None = None,
    logits_base: Path | None = None,
) -> dict:
    """
    Run a single (non-sweep) pipeline: download → start → agent → stop → ppl → evaluate.
    logits_save: path to save reference logits (llamacpp sweep reference entry).
    logits_base: path to load reference logits for KL computation.
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

    # 2. Perplexity + KL divergence (llamacpp only; before server start to avoid VRAM conflict)
    ppl: float | None = None
    kl: float | None = None

    ppl_cfg = config.evaluation.perplexity
    kl_cfg = config.evaluation.kl_divergence
    wants_ppl = ppl_cfg.enabled and config.backend.type == "llamacpp"
    wants_kl = kl_cfg.enabled and config.backend.type == "llamacpp" and (logits_save or logits_base)

    if wants_ppl or wants_kl:
        from .perplexity import _load_text, run_llama_perplexity

        console.print("\n[bold]Step 2/4:[/bold] Computing perplexity...")
        t0 = time.monotonic()
        text = _load_text(ppl_cfg)
        ppl, kl = run_llama_perplexity(
            cmd_template=config.backend.llamacpp.perplexity_cmd_template,
            model_path=model_path,
            text=text,
            n_gpu_layers=config.backend.llamacpp.n_gpu_layers,
            logits_save=logits_save,
            logits_base=logits_base,
        )
        console.print(f"[green]Perplexity:[/green] {ppl:.4f} ({time.monotonic() - t0:.1f}s)")
        if kl is not None:
            console.print(f"[green]KL divergence:[/green] {kl:.6f}")

    # 3. Start backend server
    console.print("\n[bold]Step 3/4:[/bold] Starting backend server...")
    _start_backend(backend, model_path, output_dir)

    agent_output: Path | None = None
    results: dict = {}

    try:
        # 4. Run Harbor
        console.print("\n[bold]Step 4/4:[/bold] Running Harbor agent...")
        t0 = time.monotonic()
        agent_output = run_agent(config, backend, output_dir)
        console.print(f"[dim]Agent done in {time.monotonic()-t0:.1f}s[/dim]")
        backend.check_alive()

    finally:
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
    version = backend.get_version()
    (output_dir / "run_meta.json").write_text(
        json.dumps({"perplexity": ppl, "kl_divergence": kl, "total_runtime": total_runtime, "version": version})
    )
    return {
        "run_id": run_id,
        "name": config.name,
        "agent_output": str(agent_output) if agent_output else None,
        "results": results,
        "output_dir": str(output_dir),
        "total_runtime": total_runtime,
        "perplexity": ppl,
        "kl_divergence": kl,
        "version": version,
    }


def run_pipeline(runs: list[RunConfig], *, sweep_name: str | None = None, resume_from: Path | None = None) -> list[dict]:
    all_results: list[dict] = []
    is_sweep = len(runs) > 1

    if resume_from is not None:
        if not is_sweep:
            console.print("[yellow]--resume-from given but config is not a sweep; running as fresh.[/yellow]")
            sweep_dir = Path(runs[0].output_dir)
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
            name = sweep_name or runs[0].name
            sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sweep_dir = Path(runs[0].output_dir) / f"sweep_{name}_{sweep_timestamp}"
        else:
            sweep_dir = Path(runs[0].output_dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        runs_to_do = runs

    for run_config in runs_to_do:
        run_config.output_dir = str(sweep_dir)

    # Logits file for KL divergence (llamacpp sweep only)
    ref_logits: Path | None = None
    if is_sweep and runs[0].backend.type == "llamacpp" and runs[0].evaluation.kl_divergence.enabled:
        ref_logits = sweep_dir / "reference_logits.bin"

    total = len(runs_to_do) + len(all_results)
    for i, run_config in enumerate(runs_to_do):
        if is_sweep:
            done = len(all_results)
            console.rule(f"[bold magenta]Sweep {done + 1}/{total}: {run_config.name}")

        actual_idx = len(all_results)
        is_reference = is_sweep and actual_idx == 0
        try:
            result = run_single(
                run_config,
                logits_save=ref_logits if is_reference else None,
                logits_base=ref_logits if (ref_logits and ref_logits.exists() and not is_reference) else None,
            )
            all_results.append(result)
        except Exception as exc:
            console.print(f"[red]Entry '{run_config.name}' failed: {exc}[/red]")
            raise SystemExit(1)

        if is_sweep:
            _write_sweep_summary_md(all_results, sweep_dir)
            _write_sweep_csv(all_results, sweep_dir)

    if is_sweep:
        _print_sweep_summary(all_results)
        if ref_logits and ref_logits.exists():
            ref_logits.unlink()

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


def _fmt_kl(kl: float | None) -> str:
    return f"{kl:.4f}" if kl is not None else "—"


def _write_sweep_summary_md(results: list[dict], sweep_dir: Path) -> None:
    """Write summary.md to *sweep_dir* from the current results list."""
    lines = [
        "# Sweep Summary\n",
        "| Run | Resolved | Total | % Resolved | PPL | KL | Runtime | Version | Exceptions | Error |",
        "|-----|----------|-------|------------|-----|----|---------|---------|------------|-------|",
    ]
    for r in results:
        resolved, total, pct = parse_results(r.get("results", {}))
        runtime = _fmt_runtime(r.get("total_runtime"))
        exceptions = _fmt_exceptions(r.get("results", {}))
        error = r.get("error", "")
        ppl = f"{r['perplexity']:.2f}" if r.get("perplexity") else "—"
        kl = _fmt_kl(r.get("kl_divergence"))
        version = r.get("version") or "—"
        lines.append(
            f"| {r['name']} | {resolved} | {total} | {pct:.1f}% | {ppl} | {kl} | {runtime} | {version} | {exceptions} | {error} |"
        )
    (sweep_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _write_sweep_csv(results: list[dict], sweep_dir: Path) -> None:
    fields = ["name", "resolved", "total", "pct_resolved", "perplexity",
              "kl_divergence", "runtime_seconds", "version", "exceptions", "error", "output_dir"]
    with (sweep_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            resolved, total, pct = parse_results(r.get("results", {}))
            w.writerow({
                "name": r["name"],
                "resolved": resolved,
                "total": total,
                "pct_resolved": round(pct, 4),
                "perplexity": r.get("perplexity") or "",
                "kl_divergence": r.get("kl_divergence") or "",
                "runtime_seconds": r.get("total_runtime") or "",
                "version": r.get("version") or "",
                "exceptions": _fmt_exceptions(r.get("results", {})),
                "error": r.get("error", ""),
                "output_dir": r.get("output_dir", ""),
            })


def _print_sweep_summary(results: list[dict]) -> None:
    console.rule("[bold]Sweep Summary")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Run", style="dim")
    table.add_column("Resolved", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("% Resolved", justify="right")
    table.add_column("PPL", justify="right")
    table.add_column("KL", justify="right")
    table.add_column("Version", justify="right")
    table.add_column("Exceptions", style="red")
    table.add_column("Runtime", justify="right")
    table.add_column("Error", style="red")

    for r in results:
        resolved, total, pct = parse_results(r.get("results", {}))
        name = r["name"]
        error = r.get("error", "")
        ppl = f"{r['perplexity']:.2f}" if r.get("perplexity") else "—"
        kl = _fmt_kl(r.get("kl_divergence"))
        version = r.get("version") or "—"
        if error:
            name += " [red](failed)[/red]"
        table.add_row(
            name, str(resolved), str(total), f"{pct:.1f}%", ppl, kl,
            version,
            _fmt_exceptions(r.get("results", {})),
            _fmt_runtime(r.get("total_runtime")), error,
        )

    console.print(table)

    if results:
        sweep_dir = Path(results[0]["output_dir"]).parent
        _write_sweep_summary_md(results, sweep_dir)
        _write_sweep_csv(results, sweep_dir)
        console.print(f"[dim]Summary saved to {sweep_dir / 'summary.md'}[/dim]")
        try:
            from .chart import plot_kl_vs_resolves
            chart_path = sweep_dir / "kl_vs_resolves.png"
            plot_kl_vs_resolves(results, chart_path)
            console.print(f"[dim]Chart saved to {chart_path}[/dim]")
        except ValueError:
            pass
