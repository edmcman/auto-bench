// Targeted repro of the vLLM 30-minute request stalls.
//
// Runs ONLY the three known-bad instances that timed out in both the
// nparallel=10 and nparallel=5 sweep runs, with per-request logging enabled
// on BOTH layers so a single stall can be correlated end-to-end:
//   - vLLM:    --enable-log-requests  -> "Received/Finished/Aborted request <id>"
//   - litellm: LITELLM_LOG=DEBUG      -> raw HTTP request + streaming chunks
//
// Goal: at the stall timestamp, find the request id in vllm.log. If vllm logs
// "Finished request <id>" promptly but litellm still times out at +30min, the
// hang is in the transport/streaming/tool-parser layer. If vllm never finishes
// (or "Aborted"), the hang is inside vllm for that specific request.
local d = import 'lib/qwen35.libsonnet';
local parallel = import 'lib/parallel.libsonnet';

parallel.apply(d.thinking_coding {
  name: "qwen-35b-a3b-vllm-repro",
  backend_type: "vllm",
  model: { source: "huggingface", repo_id: "Qwen/Qwen3.5-35B-A3B" },
  sampling+: { max_tokens: 32768 },
  agent+: {
    attempts: 1,
    agent_timeout_multiplier: 3,
    agent_env: ["LITELLM_LOG=DEBUG"],
  },
  instance_ids: [
    "swe-bench/matplotlib__matplotlib-25479",
    "swe-bench/matplotlib__matplotlib-26208",
    "swe-bench/django__django-14311",
  ],
  // vllm bug #45198 workaround + per-request logging for stall diagnosis
  vllm+: { extra_args: ["--disable-custom-all-reduce", "--enable-log-requests"] },
}, 3)
