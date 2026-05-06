# auto-bench

Fully automated SWE-bench Verified runs against locally-hosted LLMs.

Given a YAML config file, this tool:

1. Downloads the model (GGUF via `huggingface_hub` for llama.cpp; HF snapshot for vLLM)
2. Starts the backend server (llama.cpp or vLLM)
3. Runs [Harbor](https://github.com/av/harbor) (with [OpenHands](https://github.com/All-Hands-AI/OpenHands) by default) against the dataset
4. Stops the server
5. Collects pass/fail results from Harbor's verifier output

Sweep mode lets you test multiple quantizations (or other parameter variations) of the same model in a single config, with an automatic comparison table at the end.

---

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (for `uvx harbor`)
- Docker (for Harbor agent containers)
- llama.cpp binary (`llama-server`) in PATH — or specify path in local config
- OR: vLLM installed (`pip install vllm`) for the vLLM backend

---

## Installation

```bash
git clone https://github.com/yourname/auto-bench
cd auto-bench
uv sync
```

---

## Quick Start

```bash
# Validate a config
uv run auto-bench validate configs/smoke-test.yaml

# Download model only (no inference)
uv run auto-bench download configs/smoke-test.yaml

# Run the full pipeline (download → server → agent → evaluate)
uv run auto-bench run configs/smoke-test.yaml

# Run inference only, skip evaluation
uv run auto-bench run configs/smoke-test.yaml --skip-eval

# Start backend server and block until Ctrl+C (useful for manual testing)
uv run auto-bench serve configs/smoke-test.yaml

# Run a quantization sweep
uv run auto-bench run configs/llama3.1-8b-quant-sweep.yaml

# Resume a partially-completed sweep (specify directory)
uv run auto-bench run configs/llama3.1-8b-quant-sweep.yaml --resume-from results/sweep_llama3.1-8b-quant-sweep_20260428_153012

# Resume the most recent sweep automatically
uv run auto-bench run configs/llama3.1-8b-quant-sweep.yaml --resume
```

All commands accept `--local` / `-l` to specify a local config file (default: `~/.config/auto-bench/local.yaml`).

---

## Two-Tier Config System

Configs are split into two files:

| File | Purpose | Committed to repo? |
|------|---------|-------------------|
| Experiment YAML | **What** to run: model, dataset, sampling, agent settings | Yes |
| `~/.config/auto-bench/local.yaml` | **How** to run: host, ports, HF token, backend paths | No |

This keeps experiment configs portable across machines while keeping machine-specific settings out of version control.

---

## Local Config (`~/.config/auto-bench/local.yaml`)

Create this file on each machine you run auto-bench on. All fields are optional.

```yaml
# Network settings — where the backend server listens
host: 127.0.0.1          # bind address (default: 172.17.0.1)
port: 8080               # override port (default: 8080 for llamacpp, 8000 for vllm)
startup_timeout: 300     # seconds to wait for backend /health endpoint

# IP that Docker agent containers use to reach the host backend
# Linux (Docker Engine): 172.17.0.1  (default)
# macOS / Docker Desktop: host.docker.internal
docker_gateway: "172.17.0.1"

# HuggingFace token for gated models
hf_token: hf_xxxxxxxxxxxxxxxxxxxx

# Backend-specific overrides (merged with experiment config)
llamacpp:
  binary: /usr/local/bin/llama-server   # path to llama-server binary
  n_gpu_layers: all                     # int, "auto", or "all"
  parallel: 4                           # concurrent request slots

vllm:
  dtype: bfloat16
  gpu_memory_utilization: 0.95
  tensor_parallel_size: 2

openai:
  base_url: https://api.openai.com/v1
  api_key: sk-xxxxxxxxxxxxxxxxxxxx
```

---

## Experiment Config Reference

```yaml
name: my-run                              # run name (used for output dirs)
dataset: SWE-bench/SWE-bench_Verified     # HF dataset (default: Verified)
split: test                               # dataset split
instance_ids: []                          # optional: only run specific instances
output_dir: results                       # where to write outputs
backend_type: llamacpp                    # llamacpp | vllm | openai
remove_downloaded_models: false           # delete model after run to free disk

model:
  name: my-model                          # logical name in predictions JSONL
  source: huggingface                     # huggingface | local
  repo_id: bartowski/Llama-3.1-8B-GGUF   # HF repo
  filename: Llama-3.1-8B-Q4_K_M.gguf    # GGUF filename (llamacpp only)
  revision: main                          # branch/tag/commit
  local_path: null                        # for source: local
  allow_patterns: null                    # vLLM: files to include (e.g. ["*.safetensors"])
  ignore_patterns: null                   # vLLM: files to exclude

  # Optional: sweep across multiple quantizations / variants
  sweep:
    - label: Q4_K_M
      filename: Llama-3.1-8B-Q4_K_M.gguf
    - label: Q8_0
      filename: Llama-3.1-8B-Q8_0.gguf
      sampling:                           # optional per-entry sampling overrides
        temperature: 0.5
      overrides:                          # arbitrary deep-merged overrides
        backend_options:
          ctx_size: 65536

backend_options:                          # experiment-level context-length settings
  ctx_size: null                          # llamacpp --ctx-size (null = model default)

sampling:
  temperature: 0.0
  top_p: 1.0
  max_tokens: -1                          # -1 = model/server default
  top_k: null
  min_p: null
  presence_penalty: null
  repetition_penalty: null
  extra: {}                              # forwarded verbatim to API extra_body

agent:
  agent: openhands                       # Harbor agent name
  env: docker                            # Harbor execution environment
  attempts: 1                            # -k (attempts per instance)
  limit: null                            # -l (max instances from dataset; null = all)
  trials: 1                              # -n (concurrent trials)
  setup_multiplier: 10.0                 # --agent-setup-timeout-multiplier
  agent_kwargs: []                       # --ak key="value" entries
                                         # openhands version/python_version auto-injected
  agent_env: []                          # extra --ae KEY=VALUE entries
  extra_args: []                         # extra Harbor CLI flags

evaluation:
  run_evaluation: true                   # set false to skip result collection
  perplexity:
    enabled: true                        # measure WikiText-2 perplexity after starting server
    dataset: wikitext
    dataset_name: wikitext-2-raw-v1
    split: test
    max_tokens: 10000                    # characters of text to evaluate
    chunk_chars: 2000                    # per-request chunk size
```

---

## Outputs

Each run creates a timestamped directory under `output_dir`:

```
results/
└── my-run_20240428_153012/
    ├── run_meta.json         # perplexity and runtime saved for --resume
    └── jobs/                 # Harbor output
        └── <instance-id>/
            └── verifier/
                └── reward.txt    # "1" = resolved, "0" = failed
```

Sweep runs collect entries under a parent `sweep_{name}_{timestamp}/` directory:

```
results/
└── sweep_llama3.1-8b-quant-sweep_20240428_153012/
    ├── summary.md            # comparison table (updated after each entry)
    ├── llama3.1-8b-quant-sweep-Q4_K_M_20240428_153012/
    │   ├── run_meta.json
    │   └── jobs/...
    └── llama3.1-8b-quant-sweep-Q8_0_20240428_160512/
        ├── run_meta.json
        └── jobs/...
```

---

## Adding a New Backend

1. Create `auto_bench/backends/mybackend.py` subclassing `Backend`
2. Implement `download()`, `start()`, `stop()`, `model_name`
3. Add the backend type to `BackendConfig.type` in `config.py`
4. Register it in `runner.py:make_backend()`
