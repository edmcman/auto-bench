// Qwen3.5-35B-A3B BF16, nonthinking-reasoning only, on all SWE-bench Verified instances.
// Built directly (not filtered out of the full quant sweep).
local d = import 'lib/qwen35.libsonnet';
local parallel = import 'lib/parallel.libsonnet';

parallel.apply(d.nonthinking_reasoning {
  name: "qwen-35b-a3b-bf16-nonthinking-reasoning",
  model: {
    source: "huggingface",
    repo_id: "unsloth/Qwen3.5-35B-A3B-GGUF",
    filenames: [
      "BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf",
      "BF16/Qwen3.5-35B-A3B-BF16-00002-of-00002.gguf",
    ],
  },
  sampling+: { max_tokens: 32768 },
  agent+: { attempts: 1, agent_timeout_multiplier: 3 },
  remove_downloaded_models: true,
}, 8)
