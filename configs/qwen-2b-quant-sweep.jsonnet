// Benchmark all 22 quants of Qwen3.5-2B on a single SWE-bench instance
// Usage: auto-bench run configs/qwen-2b-quant-sweep.jsonnet
local d = import 'lib/qwen35.libsonnet';
local quants = import 'lib/quants.libsonnet';
local entries = quants.qwen_2b;

local base = d.nonthinking_general {
  name: "qwen-2b-quant-sweep",
  instance_ids: ["swe-bench/sympy__sympy-22914"],
  model: {
    name: "Qwen3.5-2B",
    source: "huggingface",
    repo_id: "unsloth/Qwen3.5-2B-GGUF",
    filename: "Qwen3.5-2B-UD-Q5_K_XL.gguf",
  },
  agent+: { attempts: 8 },
  remove_downloaded_models: false,
};

std.map(
  function(q)
    base {
      name+: "-" + q.label,
      model+: { filename: q.filename },
    },
  entries
) + [
  base {
    name+: "-vllm",
    backend_type: "vllm",
    model: { name: "Qwen3.5-2B", source: "huggingface", repo_id: "Qwen/Qwen3.5-2B" },
  },
]