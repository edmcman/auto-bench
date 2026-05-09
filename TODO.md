# TODO

## Perplexity / KL divergence for non-llamacpp backends

Currently PPL and KL divergence are only computed for `type: llamacpp` (via the
`llama-perplexity` binary). vLLM and OpenAI backends silently skip this step.

The OpenAI-compatible `/v1/completions` endpoint supports `echo: true` +
`logprobs: N` on vLLM (and real OpenAI), returning per-token logprobs for the
full prompt. This is enough to compute both PPL and a top-N KL divergence
estimate.

Challenges:
- llama-server does not implement `echo` (silently ignored; only the single
  generated token's logprob is returned).
- KL divergence via top-N is approximate; tokens outside the top-N are floored
  (e.g. log P = −50). With N=500 the missing rate is ~6% for Qwen3.5-2B on
  wikitext; higher N reduces this but the response payload grows proportionally.
- vLLM caps logprobs at 20 by default; requires `--max-logprobs N` at startup.
  This should be auto-injected when KL is enabled (was done before, removed
  along with the HTTP path).
- The reference logits can't easily be shared across backends (binary format is
  llama-perplexity-specific); cross-backend KL would require computing and
  storing the top-N vocab distributions in a portable format (e.g. gzip JSON).
