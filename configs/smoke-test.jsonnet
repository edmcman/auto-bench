// Quick smoke-test on 1 instance with a small local model
// Usage: auto-bench run configs/smoke-test.jsonnet
local d = import 'lib/qwen35.libsonnet';
d.nonthinking_general {
  name: "smoke-test",
  instance_ids: ["swe-bench/sympy__sympy-22914"],
  model: d.models["2b"].gguf("UD-Q5_K_XL"),
  agent+: { attempts: 1, limit: 1 },
  evaluation+: { perplexity: { enabled: true } },
}