"""Pydantic config schema for auto-swe-bench YAML configs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

class SweepEntry(BaseModel):
    """One entry in a quantization / parameter sweep."""
    label: str
    filename: str | None = None
    # Allow overriding any sampling or backend fields per sweep entry
    sampling: SamplingConfig | None = None
    backend: BackendConfig | None = None


class ModelConfig(BaseModel):
    """Model source and identity config."""
    # Logical name written into the predictions JSONL
    name: str | None = None
    # HuggingFace source (default)
    source: Literal["huggingface", "local"] = "huggingface"
    repo_id: str | None = None
    # For llama.cpp: specific GGUF filename (required unless using sweep)
    filename: str | None = None
    revision: str = "main"
    # For local source: path to model file or directory
    local_path: str | None = None
    hf_token: str | None = None
    # vLLM: filter which files to download
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    # Quantization sweep — if set, runner creates one sub-run per entry
    sweep: list[SweepEntry] | None = None

    @model_validator(mode="after")
    def _validate(self) -> ModelConfig:
        if self.source == "local" and not self.local_path:
            raise ValueError("local_path is required when source='local'")
        return self

    def effective_name(self) -> str:
        if self.name:
            return self.name
        if self.repo_id:
            return self.repo_id.split("/")[-1]
        if self.local_path:
            return Path(self.local_path).stem
        return "unknown-model"


# ---------------------------------------------------------------------------
# Backend config
# ---------------------------------------------------------------------------

class LlamaCppConfig(BaseModel):
    binary: str = "llama-server"
    ctx_size: int = -1
    n_gpu_layers: int | str = "auto"
    parallel: int = 1
    extra_args: list[str] = Field(default_factory=list)


class VllmConfig(BaseModel):
    dtype: str = "auto"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    quantization: str | None = None
    enforce_eager: bool = False
    max_num_seqs: int | None = None
    api_key: str | None = None
    chat_template: str | None = None
    extra_args: list[str] = Field(default_factory=list)


class OpenAIConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    model: str | None = None  # model name to use in API requests (e.g. "gpt-4o")


class BackendConfig(BaseModel):
    type: Literal["llamacpp", "vllm", "openai"]
    host: str = "127.0.0.1"
    port: int | None = None  # defaults: llamacpp=8080, vllm=8000
    startup_timeout: int = 300  # seconds to wait for /health
    # IP that Docker containers use to reach the host (172.17.0.1 on Linux,
    # host.docker.internal on macOS/Docker Desktop)
    docker_gateway: str = "172.17.0.1"

    llamacpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    vllm: VllmConfig = Field(default_factory=VllmConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)

    def effective_port(self) -> int:
        if self.port is not None:
            return self.port
        return 8080 if self.type == "llamacpp" else 8000

    def base_url(self) -> str:
        if self.type == "openai":
            return self.openai.base_url
        return f"http://{self.host}:{self.effective_port()}/v1"


# ---------------------------------------------------------------------------
# Sampling config
# ---------------------------------------------------------------------------

class SamplingConfig(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int = 4096
    # Extra params forwarded verbatim to the OpenAI client extra_body
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    agent: str = "openhands"          # --agent
    env: str = "docker"               # --env (Harbor execution environment)
    attempts: int = 1                 # -k (attempts per instance)
    limit: int | None = None          # -l (max instances from dataset; None = all)
    trials: int = 1                   # -n (concurrent trials)
    setup_multiplier: float = 10.0    # --agent-setup-timeout-multiplier
    # --ak key="value" entries (openhands version/python_version auto-injected)
    agent_kwargs: list[str] = Field(default_factory=list)
    # Additional --ae KEY=VALUE entries for the agent container
    agent_env: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------

class EvaluationConfig(BaseModel):
    run_evaluation: bool = True


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    name: str
    dataset: str = "SWE-bench/SWE-bench_Verified"
    split: str = "test"
    instance_ids: list[str] = Field(default_factory=list)
    output_dir: str = "results"

    model: ModelConfig
    backend: BackendConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML config file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return RunConfig.model_validate(data)


def expand_sweep(config: RunConfig) -> list[RunConfig]:
    """
    If config.model.sweep is set, return one RunConfig per sweep entry
    (each with model.filename and model.name resolved for that entry,
    and any per-entry overrides applied).  Otherwise return [config].
    """
    if not config.model.sweep:
        return [config]

    runs: list[RunConfig] = []
    for entry in config.model.sweep:
        # Deep copy via re-validation
        data = config.model_dump()

        # Apply sweep entry fields
        data["name"] = f"{config.name}-{entry.label}"
        data["model"]["filename"] = entry.filename or config.model.filename
        data["model"]["name"] = entry.label
        # Remove sweep to avoid infinite recursion
        data["model"]["sweep"] = None

        if entry.sampling:
            data["sampling"] = {**data["sampling"], **entry.sampling.model_dump(exclude_none=True)}
        if entry.backend:
            data["backend"] = {**data["backend"], **entry.backend.model_dump(exclude_none=True)}

        runs.append(RunConfig.model_validate(data))

    return runs
