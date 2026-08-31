// Qwen3.5-35B-A3B BF16, nonthinking-reasoning, on all SWE-bench Verified instances,
// swept across multiple Harbor agents.
//
// Note: claude-code and codex are vendor CLI wrappers (Anthropic/OpenAI's own
// CLIs) and may not honor OPENAI_BASE_URL/a custom "openai/<model>" the way
// the other agents here do -- worth a smoke-test run before committing to the
// full 500-instance sweep for those two.
local base = import 'qwen-35b-a3b-bf16-nonthinking-reasoning.jsonnet';
local agent_sweep = import 'lib/agent_sweep.libsonnet';

agent_sweep.apply(base, [
  "openhands",
  "opencode",
  "mini-swe-agent",
  "hermes",
  "pi",
  "claude-code",
  "codex",
])
