// Quick smoke-test on 1 instance with a small local model
// Usage: auto-bench run configs/smoke-test.jsonnet
local d = import 'lib/qwen35.libsonnet';
d.nonthinking_general {
  name: "smoke-test",
  instance_ids: ["swe-bench/sympy__sympy-22914"],
  model: {
    name: "Qwen3.5-2B-UD-Q5_K_XL",
    source: "huggingface",
    repo_id: "unsloth/Qwen3.5-2B-GGUF",
    filename: "Qwen3.5-2B-UD-Q5_K_XL.gguf",
  },
  agent+: { attempts: 1, limit: 1 },
  evaluation+: { perplexity: { enabled: true } },
}