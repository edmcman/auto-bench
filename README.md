# auto-swe-bench

Fully automated SWE-bench Verified runs against locally-hosted LLMs.

Given a YAML config file, this tool:

1. Downloads the model (GGUF via `huggingface_hub` for llama.cpp; HF snapshot for vLLM)
2. Starts the backend server (llama.cpp or vLLM)
3. Runs [SWE-agent](https://github.com/SWE-agent/SWE-agent) against all 500 SWE-bench Verified instances
4. Stops the server
5. Runs the SWE-bench Docker evaluation harness and reports % resolved

Sweep mode lets you test multiple quantizations (or other parameter variations) of the same model in a single config, with an automatic comparison table at the end.

---

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Docker (for the evaluation harness)
- llama.cpp binary (`llama-server`) in PATH — or specify path in config
- OR: vLLM installed (`pip install vllm`) for the vLLM backend

---

## Installation

```bash
git clone https://github.com/yourname/auto-swe-bench
cd auto-swe-bench
uv sync
```

---

## Quick Start

```bash
# Validate a config
auto-swe-bench validate configs/llama3.1-8b-llamacpp.yaml

# Download model only (no inference)
auto-swe-bench download configs/llama3.1-8b-llamacpp.yaml

# Run the full pipeline (download → server → agent → evaluate)
auto-swe-bench run configs/llama3.1-8b-llamacpp.yaml

# Run inference only, skip evaluation harness
auto-swe-bench run configs/llama3.1-8b-llamacpp.yaml --skip-eval

# Run a quantization sweep
auto-swe-bench run configs/llama3.1-8b-quant-sweep.yaml

# Run evaluation on existing predictions
auto-swe-bench evaluate configs/llama3.1-8b-llamacpp.yaml results/myrun/predictions.jsonl
```

---

## YAML Config Reference

```yaml
name: my-run                              # run name (used for output dirs)
dataset: SWE-bench/SWE-bench_Verified     # HF dataset (default: Verified)
instance_ids: []                          # optional: only run specific instances
output_dir: results                       # where to write outputs

model:
  name: my-model                          # logical name in predictions JSONL
  source: huggingface                     # huggingface | local
  repo_id: bartowski/Llama-3.1-8B-GGUF   # HF repo
  filename: Llama-3.1-8B-Q4_K_M.gguf    # GGUF filename (llamacpp only)
  revision: main                          # branch/tag/commit
  local_path: null                        # for source: local
  hf_token: null                          # or set HF_TOKEN env var

  # Optional: sweep across multiple quantizations / variants
  sweep:
    - label: Q4_K_M
      filename: Llama-3.1-8B-Q4_K_M.gguf
    - label: Q8_0
      filename: Llama-3.1-8B-Q8_0.gguf
      sampling:                           # optional per-entry overrides
        max_tokens: 8192

backend:
  type: llamacpp                          # llamacpp | vllm
  host: 127.0.0.1
  port: 8080                             # default: 8080 (llamacpp), 8000 (vllm)
  startup_timeout: 300                   # seconds to wait for /health

  llamacpp:
    binary: llama-server                 # or full path
    ctx_size: 32768
    n_gpu_layers: auto                   # int, "auto", or "all"
    parallel: 4
    flash_attn: true
    extra_args: []                       # passthrough CLI flags

  vllm:
    dtype: bfloat16
    max_model_len: 32768
    gpu_memory_utilization: 0.9
    tensor_parallel_size: 1
    quantization: null                   # awq, gptq, etc.
    extra_args: []

sampling:
  temperature: 0.0
  top_p: 1.0
  max_tokens: 4096
  top_k: null
  min_p: null
  extra: {}                              # forwarded verbatim to API extra_body

agent:
  max_steps: 30
  timeout: 300                           # seconds per instance
  extra_args: []                         # extra sweagent CLI flags

evaluation:
  run_evaluation: true
  max_workers: 4                         # parallel Docker containers
  cache_level: env                       # none | base | env | instance
  clean: false
  force_local_build: null               # auto-detected on ARM; set true to force
```

---

## Example Configs

| Config | Description |
|--------|-------------|
| `configs/llama3.1-8b-llamacpp.yaml` | Llama 3.1 8B Q4_K_M via llama.cpp |
| `configs/qwen2.5-72b-vllm.yaml` | Qwen2.5 72B via vLLM (4× GPU) |
| `configs/llama3.1-8b-quant-sweep.yaml` | Q4_K_M / Q5_K_M / Q6_K / Q8_0 sweep |
| `configs/smoke-test.yaml` | 3 instances only, no evaluation (quick test) |

---

## Outputs

Each run creates a directory under `results/<run_id>/`:

```
results/my-run-20240428_153012/
├── predictions.jsonl     # model patches (input to evaluator)
├── <run_id>.json         # evaluation results
└── logs/                 # per-instance agent logs
```

For sweep runs, a `results/<sweep_name>/summary.md` comparison table is also written.

---

## Adding a New Backend

1. Create `auto_swe_bench/backends/mybackend.py` subclassing `Backend`
2. Implement `download()`, `start()`, `stop()`, `model_name`
3. Add the backend type to `BackendConfig.type` in `config.py`
4. Register it in `runner.py:make_backend()`
