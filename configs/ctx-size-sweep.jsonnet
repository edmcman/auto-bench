// Sweep over various context sizes
// Usage: auto-bench run configs/ctx-size-sweep.jsonnet
local d = import 'lib/qwen35.libsonnet';
local ctx_sizes = [
  { label: "ctx-1k", ctx_size: 1024 },
  { label: "ctx-4k", ctx_size: 4096 },
  { label: "ctx-8k", ctx_size: 8192 },
  { label: "ctx-16k", ctx_size: 16384 },
  { label: "ctx-32k", ctx_size: 32768 },
  { label: "ctx-64k", ctx_size: 65536 },
  { label: "ctx-128k", ctx_size: 131072 },
];

local base = d.nonthinking_general {
  name: "ctx-size-sweep",
  instance_ids: ["swe-bench/sympy__sympy-22914"],
  model: d.models["2b"].gguf("UD-Q5_K_XL"),
  agent+: { attempts: 1, limit: 1 },
};

std.map(
  function(cs)
    base {
      name+: "-" + cs.label,
      backend_options+: { ctx_size: cs.ctx_size },
    },
  ctx_sizes
)