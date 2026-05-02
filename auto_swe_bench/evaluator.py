"""Evaluation result collection from Harbor verifier output."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()


def collect_harbor_results(jobs_dir: Path) -> dict:
    resolved_ids: list[str] = []
    failed_ids: list[str] = []

    for reward_file in sorted(jobs_dir.rglob("verifier/reward.txt")):
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


def parse_results(results: dict) -> tuple[int, int, float]:
    try:
        resolved = results.get("resolved", 0)
        total = results.get("total", 0)
        pct = (resolved / total * 100) if total > 0 else 0.0
        return resolved, total, pct
    except Exception:
        return 0, 0, 0.0
