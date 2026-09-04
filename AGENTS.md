# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**auto-bench** orchestrates fully-automated SWE-bench runs against locally-hosted LLMs. The pipeline is:

```
jsonnet config → download model → start backend server → run Harbor agent → collect results
```

It shells out to two external tools: `llama-server` or `vllm serve` (inference) and `uvx harbor run` (agent + evaluation). `uv` must be installed separately for `uvx` to work.

## Setup & Commands

```bash
uv sync                                                     # install dependencies
uv run auto-bench validate configs/smoke-test.jsonnet    # validate config
uv run auto-bench download configs/smoke-test.jsonnet    # download model only
uv run auto-bench run configs/smoke-test.jsonnet          # full pipeline
uv run auto-bench serve configs/smoke-test.jsonnet        # download + start server, block until Ctrl+C
```

There is no test suite. `configs/smoke-test.jsonnet` (1 instance) is the standard quick check.

All commands accept `--local` / `-l` to specify a local config file (default: `~/.config/auto-bench/local.yaml`). `run` also accepts `--skip-eval` and `--skip-download`. `serve` accepts `--dry-run` / `-n` to print the backend command without starting it.

## Architecture

### Two-Tier Config System (`config.py`)

Experiment jsonnet files use `ExperimentConfig` (portable: what to run). Machine-specific settings live in `LocalConfig` (how to run: host, ports, HF token, backend-specific overrides). `merge_configs()` combines them into a unified `RunConfig`.

```python
ExperimentConfig  # name, dataset, split, instance_ids, output_dir, backend_type,
                  # backend_options, model, sampling, agent, evaluation,
                  # llamacpp, vllm  (optional backend overrides, merged with LocalConfig)

LocalConfig       # host, port, startup_timeout, docker_gateway, hf_token,
                  # llamacpp, vllm, openai  (each optional backend-specific overrides)

RunConfig         # merged: model, backend, backend_options, sampling, agent, evaluation
```

jsonnet configs produce either a single object (one run) or a list of objects (sweep). `load_experiment_configs()` evaluates the jsonnet file, validates each object as `ExperimentConfig`, and returns a list. Each is then merged with `LocalConfig` to produce `list[RunConfig]`.

Sweeps are defined natively in jsonnet using `std.map` and `import`, eliminating the need for a separate Python-side sweep expansion step. Shared data (e.g., quant lists) can be extracted into `configs/lib/*.libsonnet` files.

### Backend Abstraction (`backends/`)

```
Backend (ABC)
└── OpenAIBackend      — no-op lifecycle (connects to pre-running server)
    ├── LlamaCppBackend — downloads a GGUF, launches llama-server subprocess
    └── VllmBackend    — downloads an HF snapshot, launches vllm subprocess
```

`OpenAIBackend` serves double duty: it is the concrete backend for `type: openai` and also the base class for `LlamaCppBackend` and `VllmBackend`. All backends expose an OpenAI-compatible `/v1/chat/completions` endpoint on `host:port`. `Backend.wait_ready()` polls `/health`; subclasses override `check_alive()` to fail fast if the subprocess dies before the server is up.

### Key Config Fields

**ModelConfig** — `source` (huggingface/local), `repo_id`, `filename` (GGUF only), `revision`, `local_path`, `allow_patterns`/`ignore_patterns` (vLLM only), `draft_filename`/`draft_repo_id`/`draft_local_path` (llama.cpp speculative decoding: a second GGUF passed as `--spec-draft-model`; `draft_repo_id` defaults to `repo_id`)

**ExperimentConfig** also accepts optional `llamacpp` and `vllm` overrides (same schema as `LlamaCppConfig`/`VllmConfig`). These are merged with `LocalConfig` defaults during `merge_configs()`, with experiment values winning. This enables experiment-level backend tuning like `llamacpp: { parallel: 4 }` without modifying the local config.

**BackendConfig** — `type` (llamacpp/vllm/openai), `host`, `port`, `startup_timeout`, `docker_gateway`
- `LlamaCppConfig`: `cmd_template`, `perplexity_cmd_template`, `n_gpu_layers` (int or "auto"/"all"), `extra_args`
- `VllmConfig`: `dtype`, `gpu_memory_utilization`, `tensor_parallel_size`, `pipeline_parallel_size`, `quantization`, `enforce_eager`, `api_key`, `chat_template`, `extra_args`
- `OpenAIConfig`: `base_url`, `api_key`, `model`

**BackendOptions** — `ctx_size` (per-slot context size; llamacpp passes `parallel * ctx_size` to `--ctx-size`, vllm passes it to `--max-model-len`), `parallel`, `chat_template_kwargs` (dict passed as `--chat-template-kwargs` JSON to both backends, e.g. `{enable_thinking: false}`)

**SamplingConfig** — `temperature`, `top_p`, `top_k`, `min_p`, `max_tokens`, `extra` (forwarded to OpenAI client `extra_body`)

**AgentConfig** — `agent`, `env`, `attempts`, `limit`, `trials`, `setup_multiplier`, `agent_timeout_multiplier`, `max_retries`, `max_iterations`, `agent_kwargs`, `agent_env`, `extra_args`

### Agent Invocation (`agent.py`)

Builds and runs `uvx harbor run`. Harbor runs the agent (e.g. OpenHands) inside Docker containers. The `OPENAI_BASE_URL` passed to containers is built from `docker_gateway:port` (not `host`), since `host` is the host-side bind address and `docker_gateway` is the address containers use to reach the host. Harbor's output lands in a `jobs/` directory under the run's output dir.

Harbor flag mapping from `AgentConfig`:
- `attempts` → `-k` (attempts per instance)
- `limit` → `-l` (max instances from dataset)
- `trials` → `-n`
- `setup_multiplier` → `--agent-setup-timeout-multiplier`
- `agent_timeout_multiplier` → `--agent-timeout-multiplier` (multiplier on each task's agent execution timeout)
- `max_retries` → `--max-retries` (max retry attempts per trial)
- `max_iterations` → `--ak max_iterations=N` (OpenHands only; limits agent steps per task)
- `agent_kwargs` → `--ak` (repeated); openhands `version`/`python_version` are auto-injected
- `agent_env` → `--ae` (repeated; `OPENAI_BASE_URL` and `OPENAI_API_KEY` also auto-injected)
- `extra_args` → appended verbatim

Dataset names are normalized (e.g. `SWE-bench/SWE-bench_Verified` → `swe-bench/swe-bench-verified`). Localhost URLs are rewritten to the Docker gateway IP for agent containers.

### Evaluation (`evaluator.py`)

Evaluation happens inline during `run`, not as a separate CLI command. `collect_harbor_results(jobs_dir)` walks `jobs/*/verifier/reward.txt` files produced by Harbor (each contains `"0"` or `"1"`) to compute resolved/total counts. `parse_results()` extracts the (resolved, total, pct) tuple.

### Parallel Modifier (`configs/lib/parallel.libsonnet`)

`parallel.apply(config, n)` adjusts a config for N-way parallelism:
- `agent.trials = n` — parallel Harbor runs
- `backend_options.parallel = n` — parallel decoding slots (llama.cpp `--parallel`) / concurrent sequences (vLLM `--max-num-seqs`)

Context size is handled automatically: `backend_options.ctx_size` is the per-slot size, and the llamacpp backend passes `parallel * ctx_size` to `--ctx-size`.

Usage:
```jsonnet
local parallel = import 'lib/parallel.libsonnet';
parallel.apply(base_config, 4)  // 4x parallelism
```

### Model Family Libs (`configs/lib/qwen35.libsonnet`, `qwen36`, `qwen38`)

One lib per Qwen release. Each exports the family's recommended sampling presets
(`thinking_general`, `thinking_coding`, `nonthinking_general`, …), already merged with
`defaults.libsonnet` and with `enable_thinking` set to match the mode, plus a `models`
catalog. Qwen 3.8 additionally exposes `thinking_xhigh`/`thinking_medium`/`thinking_low`,
which pin its `reasoning_effort` chat-template kwarg.

Sampling values differ per family — notably thinking-mode `presence_penalty` is 1.5 for 3.5
but 0.0 for 3.6/3.8 — so always take them from the matching lib rather than copying.

### Model Catalogs (`configs/lib/gguf.libsonnet`)

`g.model(name, quants)` describes an Unsloth GGUF repo (`unsloth/<name>-GGUF`) and its
upstream Qwen repo (`Qwen/<name>`), listing every quant published there. Family libs use it
to populate `models`; configs then name a model and a quant label instead of writing repo ids
and `.gguf` filenames:

```jsonnet
local d = import 'lib/qwen36.libsonnet';
local m = d.models['27b'];
m.gguf("UD-Q4_K_XL")  // llamacpp model block (repo_id + filename)
m.gguf("BF16")        // sharded quant → filenames list
m.hf()                // vllm model block (upstream Qwen repo)
m.quants              // every quant, as {label, filename|filenames}
m.pick(["BF16", "Q8_0"])
```

Quants are declared compactly and expanded by `g.model`: `"Q8_0"` → `<name>-Q8_0.gguf`;
`{label: "BF16", shards: 2}` → `BF16/<name>-BF16-00001-of-00002.gguf`, …; an explicit
`{label, filename|filenames}` object passes through. An unknown label is a jsonnet error
listing the available ones, rather than a 404 at download time. Catalogs cover the quants
and (via `drafts`) the MTP drafters — mmproj and imatrix files are omitted.

MTP (multi-token prediction, for speculative decoding) is published in two shapes, and both
need `llamacpp+: gguf.mtp_spec_n(n)` (`--spec-type draft-mtp --spec-draft-n-max n`;
`mtp_spec` is `n=2`) plus a llama.cpp build from after 2026-06-07:

- **Baked into the weights**, in a parallel `unsloth/<name>-MTP-GGUF` repo (identical weights
  with the MTP layer kept, ~2% larger) — Qwen 3.5/3.6. These get a second catalog entry built
  by `g.mtp`, e.g. `d.models['27b-mtp']`, and run like any other GGUF. Published quant lists
  differ between a base repo and its MTP repo, so each is listed separately.
- **A separate drafter GGUF** under `MTP/` in the *base* repo — Gemma 4 and Qwen 3.8. No extra
  catalog entry: the model declares `drafts: ["Q8_0", …]` (precisions of
  `MTP/mtp-<name>-<label>.gguf`, with `draft_name` overriding that `<name>` where a repo names
  its drafter after a different model, as the Gemma 4 qat repo does), and a config asks for one
  alongside the target quant with `m.gguf("UD-Q4_K_XL", draft="Q8_0")`. That sets
  `model.draft_filename`, which the llamacpp backend downloads and passes to
  `--spec-draft-model`.

`quant_sweep.make` takes a catalog entry as `model:` and label strings as `quants:`
(defaulting to the whole repo).

## Key Design Points

- **Two-tier config**: Experiment jsonnet describes what to run; `~/.config/auto-bench/local.yaml` describes how to run it (host, ports, tokens). This keeps experiment configs portable.
- **Sweep mode in jsonnet**: A jsonnet file outputs a list of config objects to define a sweep. Shared data (sampling presets, model/quant catalogs) lives in `configs/lib/*.libsonnet` and is imported. The sweep directory is named after the config file stem (e.g., `sweep_qwen-2b-quant-sweep_20250512_120000`).
- **OpenAI backend as base class**: `type: openai` skips download/start/stop entirely and points at an already-running server. `LlamaCppBackend` and `VllmBackend` inherit from it for shared properties like `model_name` and `base_url`.
- **Docker gateway vs host**: `host` is the address the server binds on (host-side). `docker_gateway` is the address containers use to reach the host. They default to the same value (`172.17.0.1` on Linux, `host.docker.internal` on macOS/Docker Desktop) but differ when, e.g., the server binds on `127.0.0.1` while containers still need the bridge IP.
- **OpenHands --ak defaults**: `version="0.57.0"` and `python_version="3.12"` are automatically prepended to `--ak` for OpenHands as a workaround for a Harbor bug, unless already specified in `agent_kwargs`.
- **Context size**: `backend_options.ctx_size` sets the per-slot context size. The llamacpp backend automatically passes `parallel * ctx_size` to `--ctx-size`, so N parallel slots each get the specified context. vLLM passes `ctx_size` directly to `--max-model-len`.
- **Perplexity/KL**: Only supported for `type: llamacpp`. Uses `llama-perplexity` (configured via `LlamaCppConfig.perplexity_cmd_template`, default `"llama-perplexity --model {model} --file {file} {args}"`). Runs before the server starts to avoid VRAM conflicts. In sweep mode, the first entry saves a reference logits file (`reference_logits.bin` in the sweep dir) via `--save-all-logits`; subsequent entries compute KL divergence against it via `--kl-divergence`.