// Qwen3.5-35B-A3B quant sweep on all SWE-bench Verified instances
local d = import 'lib/qwen35.libsonnet';
local parallel = import 'lib/parallel.libsonnet';
local nparallel = 8;

local llamacpp_quants = [
  { label: "BF16",   filenames: ["BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf", "BF16/Qwen3.5-35B-A3B-BF16-00002-of-00002.gguf"] },
  { label: "Q8_0",   filename: "Qwen3.5-35B-A3B-Q8_0.gguf" },
  { label: "Q5_K_M", filename: "Qwen3.5-35B-A3B-Q5_K_M.gguf" },
];

local modes = [
  { label: "thinking", preset: d.thinking_coding },
  { label: "nonthinking", preset: d.nonthinking_general },
];

local base(mode) = mode.preset {
    name: "qwen-35b-a3b-quant-sweep-" + mode.label,
    model: {
      source: "huggingface",
      repo_id: "unsloth/Qwen3.5-35B-A3B-GGUF",
    },
    sampling+: { max_tokens: 32768 },
    agent+: { attempts: 1, agent_timeout_multiplier: 3 },
    remove_downloaded_models: true,
  };

std.flattenArrays(std.map(
  function(mode)
    std.map(
      function(q) parallel.apply(base(mode) {
        name+: "-" + q.label,
        model+: if std.objectHas(q, 'filenames')
          then { filenames: q.filenames }
          else { filename: q.filename },
      }, nparallel),
      llamacpp_quants
    ) + [
      parallel.apply(base(mode) {
        name+: "-vllm",
        backend_type: "vllm",
        model: { source: "huggingface", repo_id: "Qwen/Qwen3.5-35B-A3B" },
        // Work around annoying vllm bug #45198
        vllm+: { extra_args: ["--disable-custom-all-reduce"] },
      }, nparallel),
    ],
  modes
))
