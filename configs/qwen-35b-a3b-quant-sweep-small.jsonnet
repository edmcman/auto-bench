// Qwen3.5-35B-A3B quant sweep on all SWE-bench Verified instances
local d = import 'lib/qwen35.libsonnet';
local quant_sweep = import 'lib/quant_sweep.libsonnet';

local llamacpp_quants = [
  { label: "BF16",   filenames: ["BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf", "BF16/Qwen3.5-35B-A3B-BF16-00002-of-00002.gguf"] },
];

quant_sweep.make({
  name_prefix: "qwen-35b-a3b-quant-sweep",
  gguf_repo_id: "unsloth/Qwen3.5-35B-A3B-GGUF",
  vllm_repo_id: "Qwen/Qwen3.5-35B-A3B",
  llamacpp_quants: llamacpp_quants,
  modes: [
    { label: "thinking-general", preset: d.thinking_general },
    { label: "nonthinking-reasoning", preset: d.nonthinking_reasoning },
  ],
  nparallel: 8,
  // Work around annoying vllm bug #45198
  vllm: { extra_args: ["--disable-custom-all-reduce"] },
})
