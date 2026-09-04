// Helpers for describing an Unsloth GGUF repo and the upstream repo it quantizes.
// Used by the per-family libs (qwen35/qwen36/qwen38/gemma4.libsonnet) to declare
// a `models` catalog, so configs name a model and a quant label instead of
// spelling out repo ids and .gguf filenames.
//
//   local d = import 'lib/qwen35.libsonnet';
//   local m = d.models['27b'];
//   m.gguf("Q8_0")     // -> model block for llamacpp (repo_id + filename)
//   m.gguf("BF16")     // -> model block with `filenames` for sharded quants
//   m.hf()             // -> model block for vllm (upstream repo)
//   m.quants           // -> every quant in the repo, as {label, filename[s]}
//   m.pick(["BF16", "Q8_0"])  // -> that subset, in the order given
//
// MTP (multi-token prediction, for self-speculative decoding) comes in two
// shapes, and a family lib uses whichever its vendor publishes:
//
//   * the layer baked into the weights, in a parallel `-MTP-GGUF` repo -- a
//     second catalog entry ('27b-mtp') built with `mtp()`, run with
//     `llamacpp+: g.mtp_spec`.  (Qwen 3.5 / 3.6)
//   * a separate drafter GGUF sitting in the *base* repo under `MTP/` -- no
//     extra entry; declare `drafts:` on the model and ask for one alongside the
//     target quant:  m.gguf("UD-Q4_K_XL", draft="Q8_0").  (Gemma 4, Qwen 3.8)
//
// Quants are declared compactly; `model()` expands them:
//   "Q8_0"                     -> Qwen3.5-27B-Q8_0.gguf
//   { label: "BF16", shards: 2 }  -> BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf, ...
//   { label: "X", filename: "..." } / { label: "X", filenames: [...] }  -> verbatim
// A sharded quant lives in a directory named after its label; pass `dir` to
// override. Catalogs list only the quants themselves and (via `drafts`) the MTP
// drafters -- mmproj and imatrix files are deliberately left out.
{
  // shards("BF16", "Qwen3.5-27B", "BF16", 2) ->
  //   ["BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf", ".../00002-of-00002.gguf"]
  shards(dir, model, quant, n):: [
    "%s/%s-%s-%05d-of-%05d.gguf" % [dir, model, quant, i, n]
    for i in std.range(1, n)
  ],

  // Expand one compact quant spec against a model name.
  quant(model, spec)::
    if std.isString(spec) then
      { label: spec, filename: model + "-" + spec + ".gguf" }
    else if std.objectHas(spec, "shards") then
      {
        label: spec.label,
        filenames: $.shards(std.get(spec, "dir", spec.label), model, spec.label, spec.shards),
      }
    else
      spec,

  // A catalog entry: an Unsloth GGUF repo plus the upstream repo vLLM uses.
  //
  // Both repo ids default to the Qwen layout -- `unsloth/<name>-GGUF` and
  // `Qwen/<name>` -- and `opts` overrides either. Other vendors keep the first
  // (Unsloth names its repos the same way) but need the second, e.g.
  //   g.model("gemma-4-26B-A4B-it", quants, { vllm_repo_id: "google/gemma-4-26B-A4B-it" })
  //
  // `opts.drafts` lists the precisions of the MTP drafter published under `MTP/`
  // in the same repo, whose files are `MTP/mtp-<name>-<label>.gguf`. Some repos
  // name the drafter after a *different* model than the target -- the Gemma 4 qat
  // repo ships the plain `mtp-gemma-4-26B-A4B-it-*.gguf` -- so `opts.draft_name`
  // overrides that half of the path.
  model(name, quants, opts={}):: {
    local this = self,

    name: name,
    gguf_repo_id: std.get(opts, "gguf_repo_id", "unsloth/" + name + "-GGUF"),
    vllm_repo_id: std.get(opts, "vllm_repo_id", "Qwen/" + name),
    quants: [$.quant(name, q) for q in quants],
    labels: [q.label for q in self.quants],
    draft_labels: std.get(opts, "drafts", []),

    // Quant entry by label; errors (rather than silently 404ing at download
    // time) when the label isn't in the repo.
    quant(label)::
      local hits = std.filter(function(q) q.label == label, this.quants);
      if std.length(hits) == 0 then
        error "unknown quant %s for %s (have: %s)" % [label, this.name, std.join(", ", this.labels)]
      else
        hits[0],

    pick(labels):: [this.quant(l) for l in labels],

    // The MTP drafter at one precision, as a fragment to merge into a model
    // block. Errors on an unpublished precision, as quant() does.
    draft(label)::
      if !std.member(this.draft_labels, label) then
        if std.length(this.draft_labels) == 0 then
          error "%s has no MTP drafter in its repo" % this.name
        else
          error "unknown MTP drafter %s for %s (have: %s)"
                % [label, this.name, std.join(", ", this.draft_labels)]
      else
        // `name` (the parameter), not this.name: mtp() renames the entry.
        { draft_filename: "MTP/mtp-%s-%s.gguf" % [std.get(opts, "draft_name", name), label] },

    // Model block for the llamacpp backend, from a quant label ...
    gguf(label, draft=null)::
      this.gguf_of(this.quant(label)) + (if draft == null then {} else this.draft(draft)),

    // ... or from an expanded {label, filename[s]} entry, so callers can also
    // use a GGUF that isn't in the catalog.
    gguf_of(q):: {
      name: this.name + "-" + q.label,
      source: "huggingface",
      repo_id: this.gguf_repo_id,
    } + (if std.objectHas(q, "filenames") then { filenames: q.filenames } else { filename: q.filename }),

    // Model block for the vllm backend (unquantized upstream weights).
    hf():: {
      name: this.name,
      source: "huggingface",
      repo_id: this.vllm_repo_id,
    },
  },

  // The same model from its parallel `unsloth/<name>-MTP-GGUF` repo: identical
  // weights with the multi-token-prediction layer kept (~2% larger), which
  // llama.cpp can use for self-speculative decoding. The .gguf files are named
  // exactly as in the base repo, so only the repo id and the entry's display
  // name change -- but the published quant list differs per repo, so it is
  // passed in rather than inherited.
  //
  // Pair with `llamacpp+: g.mtp_spec` to actually turn speculative decoding on;
  // without it the model simply runs as normal and the MTP layer is unused.
  mtp(name, quants, opts={}):: $.model(name, quants, opts) {
    name: name + "-MTP",
    gguf_repo_id: std.get(opts, "gguf_repo_id", "unsloth/" + name + "-MTP-GGUF"),
  },

  // llama.cpp flags enabling MTP speculative decoding, drafting `n` tokens ahead.
  // Requires a build with `--spec-type draft-mtp` support (merged 2026-06-07,
  // ggml-org/llama.cpp#23398). Use with either MTP shape: a `-MTP-GGUF` model,
  // or a target whose model block names a drafter via `gguf(label, draft=...)`.
  mtp_spec_n(n):: { extra_args+: ["--spec-type", "draft-mtp", "--spec-draft-n-max", std.toString(n)] },
  mtp_spec:: $.mtp_spec_n(2),
}
