// Qwen 3.5 defaults: general SWE-bench defaults + Qwen 3.5 sampling
// Use: local d = import 'lib/qwen35.libsonnet'; d { name: "...", model: {...} }
local defaults = import 'defaults.libsonnet';
defaults + {
  sampling: {
    temperature: 0.7,
    top_p: 0.8,
    top_k: 20,
    min_p: 0.0,
    presence_penalty: 1.5,
    repetition_penalty: 1.0,
  },
}