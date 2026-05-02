"""Evaluation result collection: Harbor verifier output and SWE-bench harness."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from .config import EvaluationConfig

console = Console()


def collect_harbor_results(jobs_dir: Path) -> dict:
    """
    Read Harbor's verifier/reward.txt files to compute resolve rate.
    Returns a dict with resolved, total, resolved_ids, failed_ids.
    """
    resolved_ids: list[str] = []
    failed_ids: list[str] = []

    for reward_file in sorted(jobs_dir.rglob("verifier/reward.txt")):
        # Harbor names trial directories after the instance ID
        instance_id = reward_file.parent.parent.name
        reward = reward_file.read_text().strip()
        if reward == "1":
            resolved_ids.append(instance_id)
        else:
            failed_ids.append(instance_id)

    total = len(resolved_ids) + len(failed_ids)
    return {
        "resolved": len(resolved_ids),
        "total": total,
        "resolved_ids": resolved_ids,
        "failed_ids": failed_ids,
    }


def run_evaluation(
    predictions_path: Path,
    run_id: str,
    config: EvaluationConfig,
    dataset: str = "SWE-bench/SWE-bench_Verified",
) -> dict:
    """
    Run the SWE-bench evaluation harness on a predictions JSONL file.
    Returns a dict with the result summary.
    """
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--predictions_path", str(predictions_path),
        "--max_workers", str(config.max_workers),
        "--run_id", run_id,
        "--cache_level", config.cache_level,
    ]

    if config.clean:
        cmd.append("--clean")

    if config.needs_local_build():
        # ARM/Apple Silicon: build Docker images locally instead of pulling x86 from DockerHub
        console.print("[yellow]ARM detected: using --namespace '' for local Docker builds[/yellow]")
        cmd += ["--namespace", ""]

    console.print(f"[cyan]Running SWE-bench evaluation harness...[/cyan]")
    console.print(f"[dim]{' '.join(cmd)}[/dim]")

    result = subprocess.run(cmd, capture_output=False, check=False)

    if result.returncode != 0:
        console.print(f"[red]Evaluation harness exited with code {result.returncode}[/red]")

    # Parse results from the output JSON that swebench writes alongside the predictions
    results_dir = predictions_path.parent
    result_files = list(results_dir.rglob(f"{run_id}*.json"))
    if result_files:
        with open(result_files[-1]) as f:
            return json.load(f)

    return {}


def parse_results(results: dict) -> tuple[int, int, float]:
    """
    Extract (resolved, total, pct) from a swebench results dict.
    Returns (0, 0, 0.0) if parsing fails.
    """
    try:
        resolved = results.get("resolved", 0)
        total = results.get("total", 0)
        pct = (resolved / total * 100) if total > 0 else 0.0
        return resolved, total, pct
    except Exception:
        return 0, 0, 0.0
