# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**auto-swe-bench** orchestrates fully-automated SWE-bench runs against locally-hosted LLMs. The pipeline is:

```
YAML config → download model → start backend server → run Harbor agent → collect results
```

It shells out to two external tools: `llama-server` or `vllm serve` (inference) and `uvx harbor run` (agent + evaluation). `uv` must be installed separately for `uvx` to work.

## Setup & Commands

```bash
uv sync                                                # install dependencies
uv run auto-swe-bench validate configs/smoke-test.yaml # validate config
uv run auto-swe-bench download configs/smoke-test.yaml # download model only
uv run auto-swe-bench run configs/smoke-test.yaml      # full pipeline
uv run auto-swe-bench evaluate <config> <preds.jsonl>  # evaluate predictions only
```

There is no test suite. `configs/smoke-test.yaml` (1 instance) is the standard quick check.

## Architecture

### Config-Driven Pipeline (`config.py`, `runner.py`)

All behavior is controlled by a YAML file validated with strict Pydantic models. `expand_sweep()` transforms a config with a `model.sweep` list into multiple `RunConfig` objects — one per quantization — enabling automated comparisons. `runner.run_pipeline()` is the main orchestration entry point; `run_single()` handles one config end-to-end.

### Backend Abstraction (`backends/`)

```
Backend (ABC)
├── OpenAIBackend      — connects to a pre-running OpenAI-compatible server (no-op lifecycle)
├── LlamaCppBackend    — downloads a GGUF, launches llama-server subprocess
└── VllmBackend        — downloads an HF snapshot, launches vllm subprocess
```

All backends expose an OpenAI-compatible `/v1/chat/completions` endpoint on `host:port`. `Backend.wait_ready()` polls `/health`; subclasses override `check_alive()` to fail fast if the subprocess dies before the server is up.

### Agent Invocation (`agent.py`)

Builds and runs `uvx harbor run`. Harbor runs the agent (e.g. OpenHands) inside Docker containers; the container reaches the local inference server via `backend.docker_gateway` (default `172.17.0.1`; use `host.docker.internal` on macOS/Docker Desktop). Harbor's output lands in a `jobs/` directory under the run's output dir.

Harbor flag mapping from `AgentConfig`:
- `attempts` → `-k` (attempts per instance)
- `limit` → `-l` (max instances from dataset)
- `trials` → `-n`
- `setup_multiplier` → `--agent-setup-multiplier`
- `agent_kwargs` → `--ak` (repeated); openhands `version`/`python_version` are auto-injected
- `agent_env` → `--ae` (repeated; `OPENAI_BASE_URL` and `OPENAI_API_KEY` also auto-injected)

### Evaluation (`evaluator.py`)

`collect_harbor_results(jobs_dir)` walks `jobs/*/verifier/reward.txt` files produced by Harbor (each contains `"0"` or `"1"`) to compute resolved/total counts. `parse_results()` extracts the (resolved, total, pct) tuple.

## Key Design Points

- **Sweep mode**: A single YAML with `model.sweep` produces a comparison table written to `results/summary.md`. Each sweep entry can override `filename`, `local_path`, `quantization`, and other model-level fields.
- **No-op OpenAI backend**: `type: openai` skips download/start/stop entirely and points directly at an already-running server. Useful for remote APIs or pre-started local servers.
- **Flash attention flag**: The `flash_attn` field in `LlamaCppConfig` accepts `true/false/auto/on/off/0/1`; the backend maps these to `--flash-attn <value>`.
- **Docker gateway**: Set `backend.docker_gateway` to the IP/hostname Docker containers use to reach the host. Default `172.17.0.1` works on Linux. On macOS with Docker Desktop use `host.docker.internal`.
- **OpenHands --ak defaults**: `version="0.57.0"` and `python_version="3.12"` are automatically prepended to `--ak` for OpenHands as a workaround for a Harbor bug, unless already specified in `agent_kwargs`.
