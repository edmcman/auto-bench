// Qwen3.5-35B-A3B quant sweep on all SWE-bench Verified instances
local d = import 'lib/qwen35.libsonnet';
local quant_sweep = import 'lib/quant_sweep.libsonnet';

quant_sweep.make({
  name_prefix: "qwen-35b-a3b-quant-sweep",
  model: d.models["35b-a3b"],
  quants: ["BF16", "Q8_0", "Q5_K_M"],
  modes: [
    { label: "thinking-general", preset: d.thinking_general },
    { label: "thinking-coding", preset: d.thinking_coding },
    { label: "nonthinking-general", preset: d.nonthinking_general },
    { label: "nonthinking-reasoning", preset: d.nonthinking_reasoning },
  ],
  nparallel: 8,
  // Work around annoying vllm bug #45198
  vllm: { extra_args+: ["--disable-custom-all-reduce"] },
})
