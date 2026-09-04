// Generic quant x mode sweep generator for llamacpp/vllm backend comparisons.
// Produces one entry per (mode, llamacpp quant) pair, plus a trailing vllm entry
// per mode. Model-agnostic: callers supply their own `modes` (each a {label, preset}
// pair, where `preset` is the mode's base config, e.g. a sampling profile) so this
// helper isn't tied to any particular model family.
// Usage:
//   local d = import 'lib/qwen35.libsonnet';
//   local quant_sweep = import 'lib/quant_sweep.libsonnet';
//   quant_sweep.make({
//     name_prefix: "qwen-27b-quant-sweep",
//     model: d.models['27b'],
//     quants: ["BF16", "Q8_0", "Q5_K_M"],   // labels from the model's catalog
//     modes: [
//       { label: "thinking", preset: d.thinking_coding },
//       { label: "nonthinking", preset: d.nonthinking_general },
//     ],
//     nparallel: 8,
//   })
// `quants` may also hold explicit {label, filename} / {label, filenames} objects
// for GGUFs outside the catalog; omit it to sweep every quant in the repo.
local parallel = import 'parallel.libsonnet';

{
  // params:
  //   name_prefix, model (a gguf.libsonnet catalog entry), modes, nparallel  (required)
  //   quants (default: every quant in the model's repo),
  //   max_tokens (default 32768), agent (default {attempts: 1, agent_timeout_multiplier: 3}),
  //   vllm (default {}, merged into the vllm entry), remove_downloaded_models (default true)
  make(params)::
    local name_prefix = params.name_prefix;
    local m = params.model;
    local quants = [
      if std.isString(q) then m.quant(q) else q
      for q in std.get(params, 'quants', m.quants)
    ];
    local modes = params.modes;
    local nparallel = params.nparallel;
    local max_tokens = std.get(params, 'max_tokens', 32768);
    local agent_overrides = std.get(params, 'agent', { attempts: 1, agent_timeout_multiplier: 3 });
    local vllm_overrides = std.get(params, 'vllm', {});
    local remove_downloaded_models = std.get(params, 'remove_downloaded_models', true);

    local base(mode) = mode.preset {
      name: name_prefix + "-" + mode.label,
      sampling+: { max_tokens: max_tokens },
      agent+: agent_overrides,
      remove_downloaded_models: remove_downloaded_models,
    };

    std.flattenArrays(std.map(
      function(mode)
        std.map(
          function(q) parallel.apply(base(mode) {
            name+: "-" + q.label,
            model: m.gguf_of(q),
          }, nparallel),
          quants
        ) + [
          parallel.apply(base(mode) {
            name+: "-vllm",
            backend_type: "vllm",
            model: m.hf(),
            vllm+: vllm_overrides,
          }, nparallel),
        ],
      modes
    )),
}
