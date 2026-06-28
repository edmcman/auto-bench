// Qwen 3.5 defaults: general SWE-bench defaults + Qwen 3.5 sampling.
// Use: local d = import 'lib/qwen35.libsonnet'; d { name: "...", model: {...} }
//
// Sampling presets (Unsloth recommended settings, see
// https://unsloth.ai/docs/models/qwen3.5) are exposed as fields so a config
// can pick a mode:  d.thinking_coding { name: "...", model: {...} }
// Each preset also sets `enable_thinking` (via --chat-template-kwargs on both
// backends) to match its mode: thinking_* enable it, nonthinking_* disable it.
local defaults = import 'defaults.libsonnet';

// Recommended sampling settings per mode.
local thinking_general = {
  sampling: {
    temperature: 1.0,
    top_p: 0.95,
    top_k: 20,
    min_p: 0.0,
    presence_penalty: 1.5,
    repetition_penalty: 1.0
  }
};
local thinking_coding = thinking_general {
  sampling+: {
    temperature: 0.6,
    presence_penalty: 0.0
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
local nonthinking_reasoning = nonthinking_general {
  sampling+: {
    temperature: 1.0,
    top_p: 0.95
  }
};

// Thinking-mode templates. chat_template_kwargs is backend-agnostic (both vllm
// and llama-server take --chat-template-kwargs), so it lives in backend_options.
local thinking_template = { backend_options+: { chat_template_kwargs: { enable_thinking: true } } };
local nonthinking_template = { backend_options+: { chat_template_kwargs: { enable_thinking: false } } };

local common = defaults {
  backend_options: { ctx_size: 262144 },
  vllm: { tool_call_parser: "qwen3_coder" },
};

{
  thinking_general: common + thinking_general + thinking_template,
  thinking_coding: common + thinking_coding + thinking_template,
  nonthinking_general: common + nonthinking_general + nonthinking_template,
  nonthinking_reasoning: common + nonthinking_reasoning + nonthinking_template
}
