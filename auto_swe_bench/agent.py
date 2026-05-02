"""Harbor invocation wrapper for SWE-bench evaluations."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from rich.console import Console

from .backends.base import Backend
from .config import RunConfig

console = Console()

# Map HuggingFace dataset names to Harbor's expected format
_HARBOR_DATASET_MAP: dict[str, str] = {
    "SWE-bench/SWE-bench_Verified": "swe-bench/swe-bench-verified",
    "SWE-bench/SWE-bench_Lite": "swe-bench/swe-bench-lite",
    "SWE-bench/SWE-bench": "swe-bench/swe-bench",
    "princeton-nlp/SWE-bench_Verified": "swe-bench/swe-bench-verified",
    "princeton-nlp/SWE-bench_Lite": "swe-bench/swe-bench-lite",
    "princeton-nlp/SWE-bench": "swe-bench/swe-bench",
}

# Default --ak kwargs injected for OpenHands (workaround for Harbor bug)
_OPENHANDS_DEFAULT_KWARGS = [
    'version="0.57.0"',
    'python_version="3.12"',
]


def _normalize_dataset(dataset: str) -> str:
    return _HARBOR_DATASET_MAP.get(dataset, dataset.lower())


def _docker_base_url(base_url: str, gateway: str) -> str:
    """Rewrite 127.0.0.1/localhost to the Docker gateway IP."""
    return base_url.replace("127.0.0.1", gateway).replace("localhost", gateway)


def _get_api_key(config: RunConfig) -> str:
    key = None
    if config.backend.type == "llamacpp":
        key = config.backend.llamacpp.api_key
    elif config.backend.type == "vllm":
        key = config.backend.vllm.api_key
    elif config.backend.type == "openai":
        key = config.backend.openai.api_key
    return key or "EMPTY"


def run_agent(config: RunConfig, backend: Backend, output_dir: Path) -> Path:
    """
    Run Harbor against the dataset.
    Returns the path to the Harbor jobs/ directory under output_dir.
    """
    uvx = shutil.which("uvx")
    if uvx is None:
        raise FileNotFoundError(
            "uvx not found in PATH. Install uv: https://docs.astral.sh/uv/"
        )

    docker_url = _docker_base_url(backend.base_url, config.backend.docker_gateway)
    model_string = f"openai/{backend.model_name}"
    dataset = _normalize_dataset(config.dataset)
    agent_cfg = config.agent

    cmd: list[str] = [
        uvx, "harbor", "run",
        "--dataset", dataset,
        "--agent", agent_cfg.agent,
        "--model", model_string,
        "--env", agent_cfg.env,
        "-k", str(agent_cfg.attempts),
        "-n", str(agent_cfg.trials),
        "--agent-setup-multiplier", str(agent_cfg.setup_multiplier),
    ]

    if agent_cfg.limit is not None:
        cmd += ["-l", str(agent_cfg.limit)]

    for instance_id in config.instance_ids:
        cmd += ["-i", instance_id]

    cmd += [
        "--ae", f"OPENAI_BASE_URL={docker_url}",
        "--ae", f"OPENAI_API_KEY={_get_api_key(config)}",
    ]

    # Auto-inject OpenHands --ak defaults (workaround for Harbor bug)
    agent_kwargs = list(agent_cfg.agent_kwargs)
    if agent_cfg.agent == "openhands":
        existing_keys = {kv.split("=")[0] for kv in agent_kwargs}
        for default_kv in _OPENHANDS_DEFAULT_KWARGS:
            key = default_kv.split("=")[0]
            if key not in existing_keys:
                agent_kwargs.insert(0, default_kv)

    for kv in agent_kwargs:
        cmd += ["--ak", kv]

    for kv in agent_cfg.agent_env:
        cmd += ["--ae", kv]

    cmd.extend(agent_cfg.extra_args)

    console.print(f"[cyan]Running Harbor:[/cyan] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, cwd=str(output_dir))
    if result.returncode != 0:
        console.print(f"[red]Harbor exited with code {result.returncode}[/red]")

    jobs_dir = output_dir / "jobs"
    if not jobs_dir.exists():
        raise FileNotFoundError(
            f"Harbor did not produce a jobs/ directory under {output_dir}. "
            "Harbor may have failed — check output above."
        )

    return jobs_dir
