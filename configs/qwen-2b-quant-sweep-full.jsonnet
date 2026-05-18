// Benchmark all 22 quants of Qwen3.5-2B on all SWE-bench instances
local base_sweep = import 'qwen-2b-quant-sweep.jsonnet';
std.map(
  function(entry) (entry + { instance_ids:: null }) {
    name: std.strReplace(entry.name, "qwen-2b-quant-sweep", "qwen-2b-quant-sweep-full"),
    sampling+: { max_tokens: 32768 },
    agent+: { attempts: 1 },
  },
  base_sweep
)
