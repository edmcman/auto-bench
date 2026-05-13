// Smoke-test with 4x parallelism (4 parallel slots, 4 parallel Harbor trials)
// Usage: auto-bench run configs/smoke-test-4x.jsonnet
local d = import 'lib/qwen35.libsonnet';
local parallel = import 'lib/parallel.libsonnet';
parallel.apply(d {
  name: "smoke-test-4x",
  instance_ids: ["swe-bench/sympy__sympy-22914"],
  model: {
    name: "Qwen3.5-2B-UD-Q5_K_XL",
    source: "huggingface",
    repo_id: "unsloth/Qwen3.5-2B-GGUF",
    filename: "Qwen3.5-2B-UD-Q5_K_XL.gguf",
  },
  agent+: { attempts: 1, limit: 1 },
  evaluation+: { perplexity: { enabled: true } },
}, 4)