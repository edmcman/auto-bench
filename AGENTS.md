# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**auto-swe-bench** orchestrates fully-automated SWE-bench runs against locally-hosted LLMs. The pipeline is:

```
YAML config → download model → start backend server → run mini-SWE-agent → evaluate → results
```

It shells out to three external tools: `llama-server` or `vllm serve` (inference), `mini-extra swebench` (the agent), and `python -m swebench.harness.run_evaluation` (scoring).

## Setup & Commands

```bash
uv sync                                      # install dependencies
auto-swe-bench validate configs/smoke-test.yaml   # validate config
auto-swe-bench download configs/smoke-test.yaml   # download model only
auto-swe-bench run configs/smoke-test.yaml        # full pipeline
auto-swe-bench evaluate <config> <preds.jsonl>    # evaluate predictions only
```

There is no test suite. `configs/smoke-test.yaml` (3 instances, no evaluation) is the standard quick check.

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

All backends ultimately expose an OpenAI-compatible `/v1/chat/completions` endpoint. `agent.py` addresses it as `openai/{backend.model_name}` via litellm. `Backend.wait_ready()` polls `/health`; subclasses override `check_alive()` to fail fast if the subprocess dies before the server is up.

### Agent Invocation (`agent.py`)

Builds and runs a `mini-extra swebench` command. Model/sampling parameters are passed as `-c key=value` config overrides (not CLI flags). The split, subset, instance filter (as regex), and worker count come from `RunConfig`. Output predictions land in a timestamped subdirectory of the output dir; `find_predictions()` searches for `preds.json` recursively.

### Evaluation (`evaluator.py`)

Shells out to `swebench.harness.run_evaluation`. On ARM/Apple Silicon, `force_local_build` is auto-detected to trigger local Docker image builds. `parse_results()` reads the resulting JSON to extract resolved/total/%.

## Key Design Points

- **Sweep mode**: A single YAML with `model.sweep` produces a comparison table written to `results/summary.md`. Each sweep entry can override `filename`, `local_path`, `quantization`, and other model-level fields.
- **No-op OpenAI backend**: `type: openai` skips download/start/stop entirely and points directly at an already-running server. Useful for remote APIs or pre-started local servers.
- **Flash attention flag**: The `flash_attn` field in `LlamaCppConfig` accepts `true/false/auto/on/off/0/1`; the backend maps these to `--flash-attn <value>`.
- **Retry limit**: `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=5` is injected into the agent subprocess environment to cap retries on inference failures.
