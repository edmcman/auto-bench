// Qwen3.5-27B quant sweep on all SWE-bench Verified instances
local d = import 'lib/qwen35.libsonnet';
local quant_sweep = import 'lib/quant_sweep.libsonnet';

quant_sweep.make({
  name_prefix: "qwen-27b-quant-sweep",
  model: d.models["27b"],
  quants: ["BF16", "Q8_0", "Q5_K_M"],
  modes: [
    { label: "thinking", preset: d.thinking_coding },
    { label: "nonthinking", preset: d.nonthinking_general },
  ],
  nparallel: 8,
})
