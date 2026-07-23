// Generic quant x mode sweep generator for llamacpp/vllm backend comparisons.
// Produces one entry per (mode, llamacpp quant) pair, plus a trailing vllm entry
// per mode. Model-agnostic: callers supply their own `modes` (each a {label, preset}
// pair, where `preset` is the mode's base config, e.g. a sampling profile) so this
// helper isn't tied to any particular model family.
// Usage:
//   local quant_sweep = import 'lib/quant_sweep.libsonnet';
//   quant_sweep.make({
//     name_prefix: "qwen-27b-quant-sweep",
//     gguf_repo_id: "unsloth/Qwen3.5-27B-GGUF",
//     vllm_repo_id: "Qwen/Qwen3.5-27B",
//     llamacpp_quants: [
//       { label: "Q8_0", filename: "Qwen3.5-27B-Q8_0.gguf" },
//     ],
//     modes: [
//       { label: "thinking", preset: d.thinking_coding },
//       { label: "nonthinking", preset: d.nonthinking_general },
//     ],
//     nparallel: 8,
//   })
local parallel = import 'parallel.libsonnet';

{
  // params:
  //   name_prefix, gguf_repo_id, vllm_repo_id, llamacpp_quants, modes, nparallel  (required)
  //   max_tokens (default 32768), agent (default {attempts: 1, agent_timeout_multiplier: 3}),
  //   vllm (default {}, merged into the vllm entry), remove_downloaded_models (default true)
  make(params)::
    local name_prefix = params.name_prefix;
    local gguf_repo_id = params.gguf_repo_id;
    local vllm_repo_id = params.vllm_repo_id;
    local llamacpp_quants = params.llamacpp_quants;
    local modes = params.modes;
    local nparallel = params.nparallel;
    local max_tokens = std.get(params, 'max_tokens', 32768);
    local agent_overrides = std.get(params, 'agent', { attempts: 1, agent_timeout_multiplier: 3 });
    local vllm_overrides = std.get(params, 'vllm', {});
    local remove_downloaded_models = std.get(params, 'remove_downloaded_models', true);

    local base(mode) = mode.preset {
      name: name_prefix + "-" + mode.label,
      model: {
        source: "huggingface",
        repo_id: gguf_repo_id,
      },
      sampling+: { max_tokens: max_tokens },
      agent+: agent_overrides,
      remove_downloaded_models: remove_downloaded_models,
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
            model: { source: "huggingface", repo_id: vllm_repo_id },
            vllm+: vllm_overrides,
          }, nparallel),
        ],
      modes
    )),
}
