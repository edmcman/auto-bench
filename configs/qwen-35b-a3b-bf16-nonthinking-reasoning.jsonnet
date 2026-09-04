// Qwen3.5-35B-A3B BF16, nonthinking-reasoning only, on all SWE-bench Verified instances.
// Built directly (not filtered out of the full quant sweep).
local d = import 'lib/qwen35.libsonnet';
local parallel = import 'lib/parallel.libsonnet';

parallel.apply(d.nonthinking_reasoning {
  name: "qwen-35b-a3b-bf16-nonthinking-reasoning",
  model: d.models["35b-a3b"].gguf("BF16"),
  sampling+: { max_tokens: 32768 },
  agent+: { attempts: 1, agent_timeout_multiplier: 3 },
  remove_downloaded_models: true,
}, 8)
