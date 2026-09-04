// Benchmark all 22 quants of Qwen3.5-2B on a single SWE-bench instance
// Usage: auto-bench run configs/qwen-2b-quant-sweep.jsonnet
local d = import 'lib/qwen35.libsonnet';
local m = d.models["2b"];

local base = d.nonthinking_general {
  name: "qwen-2b-quant-sweep",
  instance_ids: ["swe-bench/sympy__sympy-22914"],
  model: m.gguf("UD-Q5_K_XL"),
  agent+: { attempts: 8 },
  remove_downloaded_models: false,
};

std.map(
  function(q)
    base {
      name+: "-" + q.label,
      model: m.gguf_of(q),
    },
  m.quants
) + [
  base {
    name+: "-vllm",
    backend_type: "vllm",
    model: m.hf(),
  },
]
