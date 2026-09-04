// Quick smoke-test on 1 instance with Gemma 4 26B-A4B.
//
// Mainly a check that Gemma 4 drives the agent at all: that llama.cpp's chat
// template produces tool calls OpenHands can use, and that thinking stays off.
// Two deliberate departures from the model's defaults:
//
//   * UD-Q4_K_XL (~17GB) rather than the smaller UD-Q2_K_XL (~10.5GB). At 2 bits
//     a failure to emit valid tool calls would be indistinguishable from the
//     plumbing being broken, which is the thing under test.
//   * ctx_size 32768 instead of the family's native 262144. 256K of KV cache on
//     a 26B MoE is a lot to ask of a smoke test; --fit on may cope, but this
//     removes the variable.
//
// Usage: auto-bench run configs/smoke-test-gemma4.jsonnet
local d = import 'lib/gemma4.libsonnet';

d.nonthinking {
  name: "smoke-test-gemma4",
  instance_ids: ["swe-bench/sympy__sympy-22914"],
  model: d.models["26b-a4b"].gguf("UD-Q4_K_XL"),
  backend_options+: { ctx_size: 32768 },
  agent+: { attempts: 1, limit: 1 },
}
