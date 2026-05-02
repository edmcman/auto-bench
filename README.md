# auto-swe-bench

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
uv run auto-swe-bench validate configs/smoke-test.yaml

# Download model only (no inference)
uv run auto-swe-bench download configs/smoke-test.yaml

# Run the full pipeline (download → server → agent → evaluate)
uv run auto-swe-bench run configs/smoke-test.yaml

# Run inference only, skip evaluation
uv run auto-swe-bench run configs/smoke-test.yaml --skip-eval

# Run a quantization sweep
uv run auto-swe-bench run configs/llama3.1-8b-quant-sweep.yaml
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
  type: llamacpp                          # llamacpp | vllm | openai
  host: 127.0.0.1
  port: 8080                             # default: 8080 (llamacpp), 8000 (vllm)
  startup_timeout: 300                   # seconds to wait for /health
  docker_gateway: "172.17.0.1"          # IP Docker containers use to reach host
                                         # use host.docker.internal on macOS

  llamacpp:
    binary: llama-server                 # or full path
    ctx_size: -1                         # -1 = use model's default context size
    n_gpu_layers: auto                   # int, "auto", or "all"
    parallel: 1
    extra_args: []                       # passthrough CLI flags

  vllm:
    dtype: auto                          # auto, bfloat16, float16, etc.
    max_model_len: null
    gpu_memory_utilization: 0.9
    tensor_parallel_size: 1
    pipeline_parallel_size: 1
    quantization: null                   # awq, gptq, etc.
    enforce_eager: false
    max_num_seqs: null
    api_key: null
    chat_template: null
    extra_args: []

  openai:                                # for type: openai (pre-running server)
    base_url: https://api.openai.com/v1
    api_key: null
    model: null                          # model name for API requests

sampling:
  temperature: 0.0
  top_p: 1.0
  max_tokens: 4096
  top_k: null
  min_p: null
  extra: {}                              # forwarded verbatim to API extra_body

agent:
  agent: openhands                       # Harbor agent name
  env: docker                            # Harbor execution environment
  attempts: 1                            # -k (attempts per instance)
  limit: null                            # -l (max instances from dataset; null = all)
  trials: 1                              # -n (concurrent trials)
  setup_multiplier: 10.0                 # --agent-setup-multiplier
  agent_kwargs: []                       # --ak key="value" entries
                                         # openhands version/python_version auto-injected
  agent_env: []                          # extra --ae KEY=VALUE entries
  extra_args: []                         # extra Harbor CLI flags

evaluation:
  run_evaluation: true                   # set false to skip result collection (same as --skip-eval)
```

---

## Outputs

Each run creates a directory under `results/<run_id>/`:

```
results/my-run-20240428_153012/
└── jobs/                     # Harbor output
    └── <instance-id>/
        └── verifier/
            └── reward.txt    # "1" = resolved, "0" = failed
```

For sweep runs, a `results/summary.md` comparison table is also written.

---

## Adding a New Backend

1. Create `auto_swe_bench/backends/mybackend.py` subclassing `Backend`
2. Implement `download()`, `start()`, `stop()`, `model_name`
3. Add the backend type to `BackendConfig.type` in `config.py`
4. Register it in `runner.py:make_backend()`
