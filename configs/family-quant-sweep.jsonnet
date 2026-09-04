// Cross-family quant sweep: Qwen 3.5 / 3.6 / 3.8 vs Gemma 4.
//
// The other sweeps in here vary quant (and mode) within a single model. This one
// varies the model *family* too, so quantization sensitivity can be compared
// across vendors on identical instances: four models x two modes x three GGUF
// precisions, plus the unquantized upstream weights via vLLM as the reference.
// 32 runs in total.
//
// Model choice is MoE where a family has one -- Qwen3.5/3.6-35B-A3B and
// gemma-4-26B-A4B. Qwen 3.8 has no MoE entry catalogued (only Qwen3.8-27B is,
// and 2.4T-A95B / Flash-Next are far out of range), so its slot is a *dense*
// 27B. That makes the 3.8 column not strictly comparable to the other three on
// architecture -- read it as "the best 3.8 we can run", not as a like-for-like
// MoE data point.
//
// MTP speculative decoding is on wherever the stack supports it. It is
// distribution-preserving, so resolve rates should match a non-MTP run; it only
// buys throughput.
//
// llama.cpp -- on for all four families, in whichever shape each publishes:
//   Qwen 3.5/3.6 come from their `-MTP-GGUF` repos (the layer baked into the
//   weights); Qwen 3.8 and Gemma 4 load a separate drafter out of the base repo.
//   Needs a build from after 2026-06-07 (ggml-org/llama.cpp#23398).
//
// vLLM -- on for all four, in two shapes, and needs vllm >= 0.21 (pyproject):
//   * The three Qwen checkpoints declare model_type qwen3_5 / qwen3_5_moe and
//     ship the head in-checkpoint (mtp.* tensors, mtp_num_hidden_layers: 1),
//     which vllm maps to Qwen3_5MTP / Qwen3_5MoeMTP. No drafter repo needed.
//   * Gemma 4 keeps its drafter in a separate assistant repo, named by the
//     `model` field of --speculative-config. Support landed in vllm 0.21.0
//     (PR vllm-project/vllm#41745), hence the floor.
//
// Scope is `limit: 50`, i.e. the first 50 SWE-bench Verified instances. Harbor
// takes them in dataset order, so every entry sees the same 50. Drop `limit`
// from `agent` below to promote this to the full 500.
//
// Usage: auto-bench run configs/family-quant-sweep.jsonnet
local qwen35 = import 'lib/qwen35.libsonnet';
local qwen36 = import 'lib/qwen36.libsonnet';
local qwen38 = import 'lib/qwen38.libsonnet';
local gemma4 = import 'lib/gemma4.libsonnet';
local g = import 'lib/gguf.libsonnet';
local quant_sweep = import 'lib/quant_sweep.libsonnet';

// Published by all four repos (the `-MTP-GGUF` ones included), so the same three
// labels sweep every family. quant_sweep appends the vllm/upstream-weights entry
// per mode on its own.
local quants = ["BF16", "UD-Q5_K_XL", "UD-Q4_K_XL"];

local agent = { attempts: 1, agent_timeout_multiplier: 3, limit: 50 };

// vLLM overrides. Always `extra_args+:` -- the family libs already put a
// --reasoning-parser in there, and a plain `extra_args:` would drop it.
//
// Work around annoying vllm bug #45198, as the 35B-A3B sweeps do.
local moe_vllm = { extra_args+: ["--disable-custom-all-reduce"] };
// In-checkpoint MTP head, drafting n tokens ahead. The head is a single layer,
// so n > 1 reruns the same layer (vllm warns, and acceptance drops); n therefore
// comes from each model's own vLLM recipe -- 1 for 3.5/3.6, 3 for 3.8 -- rather
// than being held equal across families or matched to the llama.cpp depth.
local vllm_mtp(n) = {
  extra_args+: ["--speculative-config", '{"method":"mtp","num_speculative_tokens":%d}' % n],
};
// Same, for a drafter that lives in its own repo rather than in the checkpoint.
local vllm_mtp_model(repo, n) = {
  extra_args+: [
    "--speculative-config",
    '{"method":"mtp","model":"%s","num_speculative_tokens":%d}' % [repo, n],
  ],
};

// One entry per family. Presets always come from the matching lib -- sampling
// differs per family (thinking presence_penalty is 1.5 for 3.5 but 0.0 for
// 3.6/3.8, and Gemma 4 recommends one profile for everything), and the libs do
// not even expose the same preset names: 3.5/3.6 have a coding-specific
// thinking preset, 3.8 and Gemma 4 do not.
//
// `spec` is the llama.cpp override enabling MTP; `draft` (where the drafter is a
// separate file rather than baked into the weights) is the precision to pull.
local families = [
  {
    label: "qwen35-35b-a3b",
    model: qwen35.models["35b-a3b-mtp"],
    spec: g.mtp_spec,
    modes: [
      { label: "thinking", preset: qwen35.thinking_coding },
      { label: "nonthinking", preset: qwen35.nonthinking_general },
    ],
    vllm: moe_vllm + vllm_mtp(1),  // recipes.vllm.ai/Qwen/Qwen3.5-35B-A3B
  },
  {
    label: "qwen36-35b-a3b",
    model: qwen36.models["35b-a3b-mtp"],
    spec: g.mtp_spec,
    modes: [
      { label: "thinking", preset: qwen36.thinking_coding },
      { label: "nonthinking", preset: qwen36.nonthinking_general },
    ],
    vllm: moe_vllm + vllm_mtp(1),  // recipes.vllm.ai/Qwen/Qwen3.6-27B
  },
  {
    label: "qwen38-27b",
    model: qwen38.models["27b"],
    draft: "Q4_0",  // the only drafter precision Unsloth publishes for 3.8
    spec: g.mtp_spec,
    modes: [
      // No coding preset for 3.8; thinking_general is its xhigh default effort.
      { label: "thinking", preset: qwen38.thinking_general },
      { label: "nonthinking", preset: qwen38.nonthinking_general },
    ],
    vllm: vllm_mtp(3),  // recipes.vllm.ai/Qwen/Qwen3.8-27B
  },
  {
    label: "gemma4-26b-a4b",
    model: gemma4.models["26b-a4b"],
    draft: "Q8_0",     // ~462MB; BF16/F16 are the same drafter, larger
    spec: g.mtp_spec,  // unsloth.ai/docs/models/mtp: start at 2, tune 1-6 per host
    modes: [
      { label: "thinking", preset: gemma4.thinking },
      { label: "nonthinking", preset: gemma4.nonthinking },
    ],
    // Drafter from the assistant repo; 2 per Unsloth's own vllm example.
    vllm: vllm_mtp_model("google/gemma-4-26B-A4B-it-assistant", 2),
  },
];

// quant_sweep has no hook for `llamacpp` or for a drafter, so MTP is layered on
// afterwards -- onto the GGUF entries only, since the vllm ones take theirs
// through --speculative-config instead.
local with_mtp(f, entries) = std.map(
  function(e)
    if e.backend_type == "vllm" then
      // The `-MTP-GGUF` catalog entries call themselves "<model>-MTP", but hf()
      // points at the plain upstream repo, so don't carry that suffix into the
      // predictions file.
      e { model+: { name: std.strReplace(e.model.name, "-MTP", "") } }
    else
      e {
        model+: if std.objectHas(f, "draft") then f.model.draft(f.draft) else {},
        llamacpp+: f.spec,
      },
  entries
);

std.flattenArrays(std.map(
  function(f)
    with_mtp(f, quant_sweep.make({
      name_prefix: "family-quant-sweep-" + f.label,
      model: f.model,
      quants: quants,
      modes: f.modes,
      nparallel: 8,
      agent: agent,
      vllm: std.get(f, "vllm", {}),
    })),
  families
))
