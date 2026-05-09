"""Evaluation result collection from Harbor verifier output."""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

console = Console()


def collect_harbor_results(jobs_dir: Path) -> dict:
    resolved_ids: list[str] = []
    failed_ids: list[str] = []
    exception_stats: dict[str, list[str]] = {}

    # Instance result.json files are at jobs/TIMESTAMP/INSTANCE/result.json
    for result_json in sorted(jobs_dir.glob("*/*/result.json")):
        instance_dir = result_json.parent
        instance_id = instance_dir.name
        reward_file = instance_dir / "verifier" / "reward.txt"

        if reward_file.exists():
            if reward_file.read_text().strip() == "1":
                resolved_ids.append(instance_id)
            else:
                failed_ids.append(instance_id)
        else:
            failed_ids.append(instance_id)

        try:
            data = json.loads(result_json.read_text())
            exc = data.get("exception_info")
            if exc:
                exc_type = exc["exception_type"]
                exception_stats.setdefault(exc_type, []).append(instance_id)
        except (json.JSONDecodeError, KeyError):
            pass

    total = len(resolved_ids) + len(failed_ids)
    n_incomplete = 0

    for job_result in sorted(jobs_dir.glob("*/result.json")):
        try:
            data = json.loads(job_result.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("finished_at") is None:
            stats = data.get("stats", {})
            n_total = data.get("n_total_trials", 0)
            n_done = stats.get("n_completed_trials", 0)
            n_incomplete += n_total - n_done

    return {
        "resolved": len(resolved_ids),
        "total": total,
        "resolved_ids": resolved_ids,
        "failed_ids": failed_ids,
        "exception_stats": exception_stats,
        "n_incomplete": n_incomplete,
    }


def parse_results(results: dict) -> tuple[int, int, float]:
    try:
        resolved = results.get("resolved", 0)
        total = results.get("total", 0)
        pct = (resolved / total * 100) if total > 0 else 0.0
        return resolved, total, pct
    except Exception:
        return 0, 0, 0.0
