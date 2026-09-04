// Gemma 4 defaults: general SWE-bench defaults + Gemma 4 sampling.
// Use: local d = import 'lib/gemma4.libsonnet'; d.nonthinking { name: "...", model: {...} }
//
// Unlike the Qwen libs, Gemma 4 recommends a *single* sampling profile for every
// use case -- temperature 1.0, top_p 0.95, top_k 64 (see
// https://huggingface.co/google/gemma-4-26B-A4B-it and
// https://unsloth.ai/docs/models/gemma-4). So the two presets here differ only
// in whether thinking is on, and are named for just that: `thinking` and
// `nonthinking`, rather than Qwen's thinking_general / thinking_coding / ...
// No min_p, presence_penalty or repetition_penalty is recommended, so none is
// set (SamplingConfig leaves them None and emits no flag).
//
// Note: Unsloth reports that the larger Gemma 4 models may still emit an empty
// thought block even with thinking disabled.
//
// `models` is the family's model catalog (see gguf.libsonnet): repo ids and
// every quant published in each Unsloth GGUF repo. Only 26B-A4B is catalogued
// so far; Gemma 4 also ships E2B, E4B, 12B and 31B.
//   local m = d.models['26b-a4b'];  m.gguf("UD-Q4_K_XL") / m.hf() / m.quants
//
// MTP: unlike Qwen 3.5/3.6 there is no parallel `-MTP-GGUF` model repo, so there
// is no `-mtp` catalog entry. The base repo instead ships separate draft modules
// (`mtp-gemma-4-26B-A4B-it.gguf` and an `MTP/` directory). Using one would mean
// pointing llama-server at it with --spec-draft-model / --spec-draft-hf, which no
// config field models today -- untested here.
local defaults = import 'defaults.libsonnet';
local g = import 'gguf.libsonnet';

// The one recommended sampling profile, used by both modes.
local sampling = {
  sampling: {
    temperature: 1.0,
    top_p: 0.95,
    top_k: 64
  }
};

// Thinking-mode templates. chat_template_kwargs is backend-agnostic (both vllm
// and llama-server take --chat-template-kwargs), so it lives in backend_options.
local thinking_template = { backend_options+: { chat_template_kwargs: { enable_thinking: true } } };
local nonthinking_template = { backend_options+: { chat_template_kwargs: { enable_thinking: false } } };

local common = defaults {
  backend_options: { ctx_size: 262144 },  // native context length
  // vllm adds --enable-auto-tool-choice itself whenever tool_call_parser is set.
  vllm: {
    tool_call_parser: "gemma4",
    extra_args: ["--reasoning-parser", "gemma4"],
  },
};

{
  thinking: common + sampling + thinking_template,
  nonthinking: common + sampling + nonthinking_template,

  models: {
    "26b-a4b": g.model("gemma-4-26B-A4B-it", [
      { label: "BF16", shards: 2 },
      "MXFP4_MOE", "Q8_0", "UD-IQ2_M", "UD-IQ2_XXS",
      "UD-IQ3_S", "UD-IQ3_XXS", "UD-IQ4_NL", "UD-IQ4_XS",
      "UD-Q2_K_XL", "UD-Q3_K_M", "UD-Q3_K_XL", "UD-Q4_K_M",
      "UD-Q4_K_S", "UD-Q4_K_XL", "UD-Q5_K_M", "UD-Q5_K_S",
      "UD-Q5_K_XL", "UD-Q6_K", "UD-Q6_K_XL", "UD-Q8_K_XL",
    ], { vllm_repo_id: "google/gemma-4-26B-A4B-it" }),

    // Quantization-aware-trained weights: quantized during training rather than
    // after, so 4-bit holds up far better. The repo publishes this one quant.
    "26b-a4b-qat": g.model("gemma-4-26B-A4B-it-qat", [
      "UD-Q4_K_XL",
    ], { vllm_repo_id: "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized" }),
  },
}
