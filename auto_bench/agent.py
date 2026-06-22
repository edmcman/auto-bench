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


def _docker_base_url(gateway: str, port: int) -> str:
    """Build the OpenAI-compatible URL for Docker containers using the gateway IP."""
    return f"http://{gateway}:{port}/v1"


def _get_api_key(config: RunConfig) -> str:
    if config.backend.type == "vllm":
        return config.backend.vllm.api_key or "EMPTY"
    if config.backend.type == "openai":
        return config.backend.openai.api_key or "EMPTY"
    return "EMPTY"


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

    if config.backend.type == "openai":
        # Rewrite localhost in user-provided URL to docker gateway
        docker_url = config.backend.openai.base_url.replace("127.0.0.1", config.backend.docker_gateway).replace("localhost", config.backend.docker_gateway)
    else:
        docker_url = _docker_base_url(config.backend.docker_gateway, config.backend.effective_port())
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
        "--agent-setup-timeout-multiplier", str(agent_cfg.setup_multiplier),
        "--agent-timeout-multiplier", str(agent_cfg.agent_timeout_multiplier),
        "--max-retries", str(agent_cfg.max_retries),
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
        if agent_cfg.max_iterations is not None and "max_iterations" not in existing_keys:
            agent_kwargs.append(f"max_iterations={agent_cfg.max_iterations}")
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
        raise RuntimeError(f"Harbor exited with code {result.returncode}")

    jobs_dir = output_dir / "jobs"
    if not jobs_dir.exists():
        raise FileNotFoundError(
            f"Harbor did not produce a jobs/ directory under {output_dir}. "
            "Harbor may have failed — check output above."
        )

    return jobs_dir
