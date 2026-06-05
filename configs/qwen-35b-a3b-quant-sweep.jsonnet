// Qwen3.5-35B-A3B quant sweep on all SWE-bench Verified instances
local d = import 'lib/qwen35.libsonnet';

local llamacpp_quants = [
  { label: "BF16",   filenames: ["BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf", "BF16/Qwen3.5-35B-A3B-BF16-00002-of-00002.gguf"] },
  { label: "Q8_0",   filename: "Qwen3.5-35B-A3B-Q8_0.gguf" },
  { label: "Q5_K_M", filename: "Qwen3.5-35B-A3B-Q5_K_M.gguf" },
];

local base = d {
  name: "qwen-35b-a3b-quant-sweep",
  model: {
    source: "huggingface",
    repo_id: "unsloth/Qwen3.5-35B-A3B-GGUF",
  },
  sampling+: { max_tokens: 32768 },
  agent+: { attempts: 1 },
};

std.map(
  function(q) base {
    name+: "-" + q.label,
    model+: if std.objectHas(q, 'filenames')
      then { filenames: q.filenames }
      else { filename: q.filename },
  },
  llamacpp_quants
) + [
  base {
    name+: "-vllm",
    backend_type: "vllm",
    model: { source: "huggingface", repo_id: "Qwen/Qwen3.5-35B-A3B" },
  },
]
