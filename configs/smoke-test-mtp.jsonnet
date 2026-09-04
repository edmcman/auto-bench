// Smoke-test for MTP self-speculative decoding.
//
// Identical to smoke-test.jsonnet except that the model comes from the
// -MTP-GGUF repo (same weights, multi-token-prediction layer kept) and
// llama-server is told to use it via --spec-type draft-mtp. Checks that the
// MTP GGUF loads, that the server accepts the spec-decoding flags, and that
// an instance still runs end to end.
//
// Run it next to `smoke-test` for the wall-clock comparison -- the two differ
// only in the model repo and those flags, so the "Agent done in ...s" lines
// are directly comparable. Resolve rates should not move: speculative decoding
// is distribution-preserving, it only changes throughput.
//
// Usage: auto-bench run configs/smoke-test-mtp.jsonnet
local d = import 'lib/qwen35.libsonnet';
local g = import 'lib/gguf.libsonnet';
local base = import 'smoke-test.jsonnet';

base {
  name: "smoke-test-mtp",
  model: d.models["2b-mtp"].gguf("UD-Q5_K_XL"),
  llamacpp+: g.mtp_spec,
}
