# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**auto-bench** orchestrates fully-automated SWE-bench runs against locally-hosted LLMs. The pipeline is:

```
YAML config → download model → start backend server → run Harbor agent → collect results
```

It shells out to two external tools: `llama-server` or `vllm serve` (inference) and `uvx harbor run` (agent + evaluation). `uv` must be installed separately for `uvx` to work.

## Setup & Commands

```bash
uv sync                                                # install dependencies
uv run auto-bench validate configs/smoke-test.yaml # validate config
uv run auto-bench download configs/smoke-test.yaml # download model only
uv run auto-bench run configs/smoke-test.yaml      # full pipeline
uv run auto-bench serve configs/smoke-test.yaml    # download + start server, block until Ctrl+C
```

There is no test suite. `configs/smoke-test.yaml` (1 instance) is the standard quick check.

All commands accept `--local` / `-l` to specify a local config file (default: `~/.config/auto-bench/local.yaml`). `run` also accepts `--skip-eval` and `--skip-download`. `serve` accepts `--dry-run` / `-n` to print the backend command without starting it.

## Architecture

### Two-Tier Config System (`config.py`)

Experiment YAML files use `ExperimentConfig` (portable: what to run). Machine-specific settings live in `LocalConfig` (how to run: host, ports, HF token, backend-specific overrides). `merge_configs()` combines them into a unified `RunConfig`.

```python
ExperimentConfig  # name, dataset, split, instance_ids, output_dir, backend_type,
                  # backend_options, model, sampling, agent, evaluation

LocalConfig       # host, port, startup_timeout, docker_gateway, hf_token,
                  # llamacpp, vllm, openai  (each optional backend-specific overrides)

RunConfig         # merged: model, backend, backend_options, sampling, agent, evaluation
```

`expand_sweep()` transforms a config with a `model.sweep` list into multiple `RunConfig` objects — one per sweep entry — enabling automated comparisons. Sweep run outputs are collected under `sweep_{name}_{timestamp}/`.

### Backend Abstraction (`backends/`)

```
Backend (ABC)
└── OpenAIBackend      — no-op lifecycle (connects to pre-running server)
    ├── LlamaCppBackend — downloads a GGUF, launches llama-server subprocess
    └── VllmBackend    — downloads an HF snapshot, launches vllm subprocess
```

`OpenAIBackend` serves double duty: it is the concrete backend for `type: openai` and also the base class for `LlamaCppBackend` and `VllmBackend`. All backends expose an OpenAI-compatible `/v1/chat/completions` endpoint on `host:port`. `Backend.wait_ready()` polls `/health`; subclasses override `check_alive()` to fail fast if the subprocess dies before the server is up.

### Key Config Fields

**ModelConfig** — `source` (huggingface/local), `repo_id`, `filename` (GGUF only), `revision`, `local_path`, `allow_patterns`/`ignore_patterns` (vLLM only), `sweep` (list of `SweepEntry`)

**BackendConfig** — `type` (llamacpp/vllm/openai), `host`, `port`, `startup_timeout`, `docker_gateway`
- `LlamaCppConfig`: `cmd_template`, `perplexity_cmd_template`, `n_gpu_layers` (int or "auto"/"all"), `parallel`, `extra_args`
- `VllmConfig`: `dtype`, `gpu_memory_utilization`, `tensor_parallel_size`, `pipeline_parallel_size`, `quantization`, `enforce_eager`, `max_num_seqs`, `api_key`, `chat_template`, `extra_args`
- `OpenAIConfig`: `base_url`, `api_key`, `model`

**BackendOptions** — `ctx_size` (llamacpp `--ctx-size`, default 0) and `max_model_len` (vllm `--max-model-len`)

**SamplingConfig** — `temperature`, `top_p`, `top_k`, `min_p`, `max_tokens`, `extra` (forwarded to OpenAI client `extra_body`)

**AgentConfig** — `agent`, `env`, `attempts`, `limit`, `trials`, `setup_multiplier`, `agent_kwargs`, `agent_env`, `extra_args`

**SweepEntry** — `label`, `filename`, `sampling`, `overrides` (arbitrary nested dict deep-merged into the run config)

### Agent Invocation (`agent.py`)

Builds and runs `uvx harbor run`. Harbor runs the agent (e.g. OpenHands) inside Docker containers; the container reaches the local inference server via `backend.docker_gateway` (default `172.17.0.1`; use `host.docker.internal` on macOS/Docker Desktop). Harbor's output lands in a `jobs/` directory under the run's output dir.

Harbor flag mapping from `AgentConfig`:
- `attempts` → `-k` (attempts per instance)
- `limit` → `-l` (max instances from dataset)
- `trials` → `-n`
- `setup_multiplier` → `--agent-setup-timeout-multiplier`
- `agent_kwargs` → `--ak` (repeated); openhands `version`/`python_version` are auto-injected
- `agent_env` → `--ae` (repeated; `OPENAI_BASE_URL` and `OPENAI_API_KEY` also auto-injected)
- `extra_args` → appended verbatim

Dataset names are normalized (e.g. `SWE-bench/SWE-bench_Verified` → `swe-bench/swe-bench-verified`). Localhost URLs are rewritten to the Docker gateway IP for agent containers.

### Evaluation (`evaluator.py`)

Evaluation happens inline during `run`, not as a separate CLI command. `collect_harbor_results(jobs_dir)` walks `jobs/*/verifier/reward.txt` files produced by Harbor (each contains `"0"` or `"1"`) to compute resolved/total counts. `parse_results()` extracts the (resolved, total, pct) tuple.

## Key Design Points

- **Two-tier config**: Experiment YAML describes what to run; `~/.config/auto-bench/local.yaml` describes how to run it (host, ports, tokens). This keeps experiment configs portable.
- **Sweep mode**: A single YAML with `model.sweep` produces a comparison table and `summary.md`. Each sweep entry can set `filename`, `sampling`, and arbitrary `overrides` for deep-merging into the run config (e.g. `backend_options.ctx_size`).
- **OpenAI backend as base class**: `type: openai` skips download/start/stop entirely and points at an already-running server. `LlamaCppBackend` and `VllmBackend` inherit from it for shared properties like `model_name` and `base_url`.
- **Docker gateway**: Set `backend.docker_gateway` to the IP/hostname Docker containers use to reach the host. Default `172.17.0.1` works on Linux. On macOS with Docker Desktop use `host.docker.internal`.
- **OpenHands --ak defaults**: `version="0.57.0"` and `python_version="3.12"` are automatically prepended to `--ak` for OpenHands as a workaround for a Harbor bug, unless already specified in `agent_kwargs`.
- **Context size**: Set via `backend_options.ctx_size` (llamacpp) or `backend_options.max_model_len` (vllm). These are experiment-level settings separate from backend-specific config.
- **Perplexity/KL**: Only supported for `type: llamacpp`. Uses `llama-perplexity` (configured via `LlamaCppConfig.perplexity_cmd_template`, default `"llama-perplexity --model {model} --file {file} {args}"`). Runs before the server starts to avoid VRAM conflicts. In sweep mode, the first entry saves a reference logits file (`reference_logits.bin` in the sweep dir) via `--save-all-logits`; subsequent entries compute KL divergence against it via `--kl-divergence`.
