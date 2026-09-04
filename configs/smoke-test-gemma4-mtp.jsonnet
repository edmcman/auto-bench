// Smoke-test for Gemma 4 MTP speculative decoding.
//
// Identical to smoke-test-gemma4.jsonnet except that llama-server also loads the
// repo's MTP drafter (--spec-draft-model) and is told to draft with it. Checks
// that the drafter downloads and loads next to the target, that the server
// accepts the spec-decoding flags, and that an instance still runs end to end.
//
// Unlike Qwen 3.5/3.6, where the MTP layer is baked into a `-MTP-GGUF` build of
// the weights, Gemma 4 keeps the drafter in a separate ~462MB file, so the model
// block names both.
//
// Run it next to `smoke-test-gemma4` for the wall-clock comparison -- the two
// differ only in the drafter and those flags, so the "Agent done in ...s" lines
// are directly comparable. Resolve rates should not move: speculative decoding
// is distribution-preserving, it only changes throughput.
//
// Usage: auto-bench run configs/smoke-test-gemma4-mtp.jsonnet
local d = import 'lib/gemma4.libsonnet';
local g = import 'lib/gguf.libsonnet';
local base = import 'smoke-test-gemma4.jsonnet';

base {
  name: "smoke-test-gemma4-mtp",
  model: d.models["26b-a4b"].gguf("UD-Q4_K_XL", draft="Q8_0"),
  llamacpp+: g.mtp_spec_n(4),  // Unsloth's recommended draft depth for Gemma 4
}
