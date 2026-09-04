"""Pydantic config schema for auto-bench configs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Model source and identity config."""
    # Logical name written into the predictions JSONL
    name: str | None = None
    # HuggingFace source (default)
    source: Literal["huggingface", "local"] = "huggingface"
    repo_id: str | None = None
    # For llama.cpp: specific GGUF filename (single file) or list of shards
    filename: str | None = None
    filenames: list[str] | None = None
    revision: str = "main"
    # For local source: path to model file or directory
    local_path: str | None = None
    # Injected from local config; not written in experiment jsonnet
    hf_token: str | None = None
    # vLLM: filter which files to download
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    # llama.cpp speculative decoding: a separate draft/MTP GGUF loaded alongside
    # the target (--spec-draft-model). Gemma 4 and Qwen 3.8 ship theirs in the
    # same repo as the target, so draft_repo_id defaults to repo_id.
    draft_repo_id: str | None = None
    draft_filename: str | None = None
    draft_local_path: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> ModelConfig:
        if self.source == "local" and not self.local_path:
            raise ValueError("local_path is required when source='local'")
        if self.draft_repo_id and not self.draft_filename:
            raise ValueError("draft_filename is required when draft_repo_id is set")
        return self

    def effective_name(self) -> str:
        if self.name:
            return self.name
        if self.repo_id:
            return self.repo_id.split("/")[-1]
        if self.local_path:
            return Path(self.local_path).stem
        return "unknown-model"

    def effective_draft_repo_id(self) -> str | None:
        return self.draft_repo_id or self.repo_id


# ---------------------------------------------------------------------------
# Backend config
# ---------------------------------------------------------------------------

class LlamaCppConfig(BaseModel):
    cmd_template: str = "llama-server --model {model} --host {host} --port {port} {args}"
    perplexity_cmd_template: str = "llama-perplexity --model {model} --file {file} {args}"
    n_gpu_layers: int | str = "auto"
    auto_fit: bool = True
    extra_args: list[str] = Field(default_factory=list)


class VllmConfig(BaseModel):
    cmd_template: str = "vllm serve {model} --host {host} --port {port} {args}"
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int | Literal["auto"] = "auto"
    pipeline_parallel_size: int = 1
    quantization: str | None = None
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    api_key: str | None = None
    chat_template: str | None = None
    tool_call_parser: str | None = None
    extra_args: list[str] = Field(default_factory=list)


class BackendOptions(BaseModel):
    """Experiment-level backend settings."""
    ctx_size: int | None = None      # per-slot context size; llamacpp passes parallel*ctx_size
    parallel: int = 1                # parallel decoding slots / concurrent sequences
    # --chat-template-kwargs JSON (both backends), e.g. {"enable_thinking": false}
    chat_template_kwargs: dict[str, Any] | None = None


class OpenAIConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    model: str | None = None  # model name to use in API requests (e.g. "gpt-4o")


class BackendConfig(BaseModel):
    type: Literal["llamacpp", "vllm", "openai"]
    host: str = "172.17.0.1"  # address the server binds on (host-side)
    port: int | None = None  # defaults: llamacpp=8080, vllm=8000
    startup_timeout: int = 300  # seconds to wait for /health
    # address containers use to reach the host (docker_gateway:port → server)
    # 172.17.0.1 on Linux, host.docker.internal on macOS/Docker Desktop
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
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int = -1
    # Extra params forwarded verbatim to the OpenAI client extra_body
    extra: dict[str, Any] = Field(default_factory=dict)

    def non_defaults(self) -> dict[str, float | int]:
        """Return sampling params that differ from their defaults."""
        result: dict[str, float | int] = {}
        if self.temperature > 0.0:
            result["temperature"] = self.temperature
        if self.top_p < 1.0:
            result["top_p"] = self.top_p
        if self.top_k is not None:
            result["top_k"] = self.top_k
        if self.min_p is not None:
            result["min_p"] = self.min_p
        if self.presence_penalty is not None:
            result["presence_penalty"] = self.presence_penalty
        if self.repetition_penalty is not None:
            result["repetition_penalty"] = self.repetition_penalty
        return result


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
    agent_timeout_multiplier: float = 1.0  # --agent-timeout-multiplier
    max_retries: int = 0              # -r (max retry attempts per trial)
    max_iterations: int | None = None  # --ak max_iterations=N (OpenHands only)
    # --ak key="value" entries (openhands version/python_version auto-injected)
    agent_kwargs: list[str] = Field(default_factory=list)
    # Additional --ae KEY=VALUE entries for the agent container
    agent_env: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------

class PerplexityConfig(BaseModel):
    enabled: bool = True
    dataset: str = "Salesforce/wikitext"
    dataset_name: str = "wikitext-2-raw-v1"
    split: str = "test"
    max_chars: int = 40_000
    chunk_chars: int = 4096
    stride_chars: int = 2048


class KLDivConfig(BaseModel):
    enabled: bool = True  # only active in sweep mode with llamacpp backends


class EvaluationConfig(BaseModel):
    run_evaluation: bool = True
    perplexity: PerplexityConfig = Field(default_factory=PerplexityConfig)
    kl_divergence: KLDivConfig = Field(default_factory=KLDivConfig)


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
    backend_options: BackendOptions = Field(default_factory=BackendOptions)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    remove_downloaded_models: bool = False
    jobs_cleanup: Literal["none", "compress", "delete"] = "compress"


# ---------------------------------------------------------------------------
# File-format models (what user writes vs merged internal representation)
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """Portable experiment description: what to run."""
    name: str
    dataset: str = "SWE-bench/SWE-bench_Verified"
    split: str = "test"
    instance_ids: list[str] = Field(default_factory=list)
    output_dir: str = "results"
    backend_type: Literal["llamacpp", "vllm", "openai"]
    backend_options: BackendOptions = Field(default_factory=BackendOptions)
    model: ModelConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    remove_downloaded_models: bool = False
    jobs_cleanup: Literal["none", "compress", "delete"] = "compress"
    # Backend-specific overrides (merged with LocalConfig; experiment wins)
    llamacpp: LlamaCppConfig | None = None
    vllm: VllmConfig | None = None


class LocalConfig(BaseModel):
    """Machine-specific settings: how to run each backend."""
    host: str = "172.17.0.1"  # address the server binds on (host-side)
    port: int | None = None
    startup_timeout: int = 900
    docker_gateway: str = "172.17.0.1"  # address containers use to reach the host
    hf_token: str | None = None
    llamacpp: LlamaCppConfig | None = None
    vllm: VllmConfig | None = None
    openai: OpenAIConfig | None = None


def merge_configs(experiment: ExperimentConfig, local: LocalConfig) -> RunConfig:
    """Combine experiment and local config into a unified RunConfig."""
    data = experiment.model_dump()
    llamacpp_data = local.llamacpp.model_dump() if local.llamacpp else {}
    vllm_data = local.vllm.model_dump() if local.vllm else {}
    # Experiment backend overrides win over local defaults (only explicitly-set fields)
    if experiment.llamacpp:
        llamacpp_data.update(experiment.llamacpp.model_dump(exclude_unset=True))
    if experiment.vllm:
        vllm_data.update(experiment.vllm.model_dump(exclude_unset=True))
    data["backend_options"] = experiment.backend_options.model_dump()
    data["backend"] = {
        "type": experiment.backend_type,
        "host": local.host,
        "port": local.port,
        "startup_timeout": local.startup_timeout,
        "docker_gateway": local.docker_gateway,
        "llamacpp": llamacpp_data,
        "vllm": vllm_data,
        "openai": local.openai.model_dump() if local.openai else {},
    }
    if local.hf_token:
        data["model"]["hf_token"] = local.hf_token
    return RunConfig.model_validate(data)


def evaluate_experiment_config(path: Path) -> Any:
    """Evaluate a jsonnet config file and return its decoded JSON value."""
    import _jsonnet

    path = Path(path)
    json_str = _jsonnet.evaluate_file(str(path))
    return json.loads(json_str)


def validate_experiment_configs(data: Any, path: Path) -> list[ExperimentConfig]:
    """Validate decoded jsonnet output and return ExperimentConfigs.

    The jsonnet file may output either a single object (one run) or a list
    of objects (sweep). Each object is validated as an ExperimentConfig.
    """
    path = Path(path)
    if isinstance(data, dict):
        entries = [data]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(
            f"jsonnet file {path} must produce an object or list, "
            f"got {type(data).__name__}"
        )

    configs: list[ExperimentConfig] = []
    errors: list[str] = []
    for i, entry in enumerate(entries):
        try:
            configs.append(ExperimentConfig.model_validate(entry))
        except Exception as exc:
            errors.append(f"  Entry {i} in {path}: {exc}")

    if errors:
        raise ValueError(f"Experiment config validation failed:\n" + "\n".join(errors))

    return configs


def load_experiment_configs(path: Path) -> list[ExperimentConfig]:
    """Evaluate and validate a jsonnet config file."""
    return validate_experiment_configs(evaluate_experiment_config(path), path)
