// Qwen 3.8 defaults: general SWE-bench defaults + Qwen 3.8 sampling.
// Use: local d = import 'lib/qwen38.libsonnet'; d { name: "...", model: {...} }
//
// Sampling presets (Qwen's recommended settings, see
// https://huggingface.co/Qwen/Qwen3.8-27B and https://unsloth.ai/docs/models/qwen3.8)
// are exposed as fields so a config can pick a mode:
//   d.thinking_medium { name: "...", model: {...} }
// Each preset also sets `enable_thinking` (via --chat-template-kwargs on both
// backends) to match its mode: thinking_* enable it, nonthinking_* disable it.
//
// Qwen 3.8 adds a `reasoning_effort` chat-template kwarg on top of
// `enable_thinking`, so the thinking_{xhigh,medium,low} presets are the same
// sampling settings with that kwarg pinned. "xhigh" is the model's own default,
// so thinking_general and thinking_xhigh should behave identically -- the
// explicit preset exists so effort levels read uniformly as sweep modes.
//
// `preserve_thinking` (default true) is deliberately left unset; override it
// per config with backend_options+.chat_template_kwargs+ if needed.
//
// Note: unlike qwen35.libsonnet, thinking mode uses presence_penalty 0.0 (not
// 1.5) -- that is what Qwen recommends for 3.8, not a typo. Qwen documents no
// coding-specific thinking settings and no non-thinking reasoning mode for 3.8,
// so there are no thinking_coding / nonthinking_reasoning presets here.
//
// `models` is the family's model catalog (see gguf.libsonnet): repo ids and
// every quant published in the Unsloth GGUF repo. Only 27B is catalogued --
// Qwen3.8-2.4T-A95B and Qwen3.8-Flash-Next also ship GGUFs, but nothing here
// runs them yet. Unlike 3.5/3.6 there are no `-MTP-GGUF` repos and so no `-mtp`
// entries; 3.8 instead ships a separate MTP drafter inside the base repo, asked
// for alongside the target quant:
//   model: m.gguf("Q8_0", draft="Q4_0"),  llamacpp+: g.mtp_spec,
// Q4_0 (~1.4GB) is the only precision published, and Unsloth documents no
// recommended draft depth for 3.8, so mtp_spec's 2 is the conservative choice.
//   local m = d.models['27b'];  m.gguf("Q8_0") / m.hf() / m.quants
local defaults = import 'defaults.libsonnet';
local g = import 'gguf.libsonnet';

// Recommended sampling settings per mode.
local thinking_general = {
  sampling: {
    temperature: 1.0,
    top_p: 0.95,
    top_k: 20,
    min_p: 0.0,
    presence_penalty: 0.0,
    repetition_penalty: 1.0
  }
};
local nonthinking_general = {
  sampling: {
    temperature: 0.7,
    top_p: 0.8,
    top_k: 20,
    min_p: 0.0,
    presence_penalty: 1.5,
    repetition_penalty: 1.0
  }
};

// Thinking-mode templates. chat_template_kwargs is backend-agnostic (both vllm
// and llama-server take --chat-template-kwargs), so it lives in backend_options.
local thinking_template = { backend_options+: { chat_template_kwargs: { enable_thinking: true } } };
local nonthinking_template = { backend_options+: { chat_template_kwargs: { enable_thinking: false } } };
// Thinking template with the reasoning depth pinned: "xhigh" | "medium" | "low".
local effort(level) = thinking_template {
  backend_options+: { chat_template_kwargs+: { reasoning_effort: level } },
};

local common = defaults {
  backend_options: { ctx_size: 262144 },  // native context length
  vllm: {
    tool_call_parser: "qwen3_coder",
    extra_args: ["--reasoning-parser", "qwen3"],
  },
};

{
  thinking_general: common + thinking_general + thinking_template,
  thinking_xhigh: common + thinking_general + effort("xhigh"),
  thinking_medium: common + thinking_general + effort("medium"),
  thinking_low: common + thinking_general + effort("low"),
  nonthinking_general: common + nonthinking_general + nonthinking_template,

  models: {
    "27b": g.model("Qwen3.8-27B", [
      { label: "BF16", shards: 2 },
      "Q4_0", "Q4_1", "Q8_0", "UD-IQ1_M",
      "UD-IQ1_S", "UD-IQ2_S", "UD-IQ2_XXS", "UD-IQ3_S",
      "UD-IQ3_XXS", "UD-IQ4_XS", "UD-Q2_K_XL", "UD-Q3_K_XL",
      "UD-Q4_K_M", "UD-Q4_K_S", "UD-Q4_K_XL", "UD-Q5_K_M",
      "UD-Q5_K_S", "UD-Q5_K_XL", "UD-Q6_K", "UD-Q6_K_L",
      "UD-Q6_K_M", "UD-Q6_K_XL", "UD-Q8_K_L", "UD-Q8_K_XL",
    ], { drafts: ["Q4_0"] }),
  },
}
