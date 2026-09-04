// Qwen 3.6 defaults: general SWE-bench defaults + Qwen 3.6 sampling.
// Use: local d = import 'lib/qwen36.libsonnet'; d { name: "...", model: {...} }
//
// Sampling presets (Qwen's recommended settings, see
// https://huggingface.co/Qwen/Qwen3.6-27B and https://unsloth.ai/docs/models/qwen3.6)
// are exposed as fields so a config can pick a mode:
//   d.thinking_coding { name: "...", model: {...} }
// Each preset also sets `enable_thinking` (via --chat-template-kwargs on both
// backends) to match its mode: thinking_* enable it, nonthinking_* disable it.
//
// Note: unlike qwen35.libsonnet, thinking mode uses presence_penalty 0.0 (not
// 1.5) -- that is what Qwen recommends for 3.6, not a typo. Qwen documents no
// non-thinking reasoning mode for 3.6, so there is no nonthinking_reasoning
// preset here.
//
// `models` is the family's model catalog (see gguf.libsonnet): repo ids and
// every quant published in each Unsloth GGUF repo. The separate `-MTP-GGUF`
// repos (multi-token-prediction draft models) are not catalogued.
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
local thinking_coding = thinking_general {
  sampling+: {
    temperature: 0.6
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

local common = defaults {
  backend_options: { ctx_size: 262144 },  // native context length
  vllm: {
    tool_call_parser: "qwen3_coder",
    extra_args: ["--reasoning-parser", "qwen3"],
  },
};

{
  thinking_general: common + thinking_general + thinking_template,
  thinking_coding: common + thinking_coding + thinking_template,
  nonthinking_general: common + nonthinking_general + nonthinking_template,

  models: {
    "27b": g.model("Qwen3.6-27B", [
      { label: "BF16", shards: 2 },
      "IQ4_NL", "IQ4_XS", "Q3_K_M", "Q3_K_S",
      "Q4_0", "Q4_1", "Q4_K_M", "Q4_K_S",
      "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0",
      "UD-IQ2_M", "UD-IQ2_XXS", "UD-IQ3_XXS", "UD-Q2_K_XL",
      "UD-Q3_K_XL", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL",
      "UD-Q8_K_XL",
    ]),
    "35b-a3b": g.model("Qwen3.6-35B-A3B", [
      { label: "BF16", shards: 2 },
      "MXFP4_MOE", "Q8_0", "UD-IQ1_M", "UD-IQ2_M",
      "UD-IQ2_XXS", "UD-IQ3_S", "UD-IQ3_XXS", "UD-IQ4_NL",
      "UD-IQ4_NL_XL", "UD-IQ4_XS", "UD-Q2_K_XL", "UD-Q3_K_M",
      "UD-Q3_K_S", "UD-Q3_K_XL", "UD-Q4_K_M", "UD-Q4_K_S",
      "UD-Q4_K_XL", "UD-Q5_K_M", "UD-Q5_K_S", "UD-Q5_K_XL",
      "UD-Q6_K", "UD-Q6_K_XL", "UD-Q8_K_XL",
    ]),
  },
}
