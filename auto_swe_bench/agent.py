"""mini-SWE-agent invocation wrapper."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from .backends.base import Backend
from .config import RunConfig

console = Console()

# Map HuggingFace dataset names to mini-swe-agent --subset shorthand
_SUBSET_MAP: dict[str, str] = {
    "SWE-bench/SWE-bench_Verified": "verified",
    "SWE-bench/SWE-bench_Lite": "lite",
    "SWE-bench/SWE-bench": "full",
    "princeton-nlp/SWE-bench_Verified": "verified",
    "princeton-nlp/SWE-bench_Lite": "lite",
    "princeton-nlp/SWE-bench": "full",
}


def _dataset_to_subset(dataset: str) -> str:
    return _SUBSET_MAP.get(dataset, dataset)


def run_agent(config: RunConfig, backend: Backend, output_dir: Path) -> Path:
    """
    Run mini-SWE-agent in SWE-bench batch mode against the dataset.
    Returns the path to the predictions file (preds.json).
    """
    predictions_path = output_dir / "preds.json"

    mini_extra = shutil.which("mini-extra")
    if mini_extra is None:
        raise FileNotFoundError(
            "mini-extra not found in PATH. Install with: pip install mini-swe-agent"
        )

    # litellm model string for a local OpenAI-compatible server
    litellm_model = f"openai/{backend.model_name}"

    subset = _dataset_to_subset(config.dataset)

    cmd: list[str] = [
        mini_extra, "swebench",
        "--model", litellm_model,
        "--api-base", backend.base_url,
        "--subset", subset,
        "--output", str(output_dir),
        "--workers", str(config.agent.workers),
        "--max-steps", str(config.agent.max_steps),
        "--temperature", str(config.sampling.temperature),
        "--top-p", str(config.sampling.top_p),
        "--max-tokens", str(config.sampling.max_tokens),
    ]

    # API key (litellm requires something even for local servers)
    api_key = (
        config.backend.llamacpp.api_key
        if config.backend.type == "llamacpp"
        else config.backend.vllm.api_key
    ) or "EMPTY"
    cmd += ["--api-key", api_key]

    # Instance filter: convert list of IDs to regex
    if config.instance_ids:
        pattern = "^(" + "|".join(re.escape(i) for i in config.instance_ids) + ")$"
        cmd += ["--filter", pattern]

    cmd.extend(config.agent.extra_args)

    console.print(f"[cyan]Running mini-SWE-agent:[/cyan] predictions → {predictions_path}")
    console.print(f"[dim]{' '.join(cmd)}[/dim]")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        console.print(f"[red]mini-swe-agent exited with code {result.returncode}[/red]")

    if not predictions_path.exists():
        # Search for preds.json in subdirs (mini-extra may create timestamped dirs)
        candidates = list(output_dir.rglob("preds.json"))
        if candidates:
            predictions_path = candidates[-1]
            console.print(f"[dim]Found predictions at {predictions_path}[/dim]")
        else:
            raise FileNotFoundError(
                f"No preds.json found under {output_dir}. "
                "mini-swe-agent may have failed — check output above."
            )

    return predictions_path
