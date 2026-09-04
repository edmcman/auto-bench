// Helpers for describing an Unsloth GGUF repo and its upstream Qwen repo.
// Used by the per-family libs (qwen35/qwen36/qwen38.libsonnet) to declare a
// `models` catalog, so configs name a model and a quant label instead of
// spelling out repo ids and .gguf filenames.
//
//   local d = import 'lib/qwen35.libsonnet';
//   local m = d.models['27b'];
//   m.gguf("Q8_0")     // -> model block for llamacpp (repo_id + filename)
//   m.gguf("BF16")     // -> model block with `filenames` for sharded quants
//   m.hf()             // -> model block for vllm (upstream Qwen repo)
//   m.quants           // -> every quant in the repo, as {label, filename[s]}
//   m.pick(["BF16", "Q8_0"])  // -> that subset, in the order given
//
// Quants are declared compactly; `model()` expands them:
//   "Q8_0"                     -> Qwen3.5-27B-Q8_0.gguf
//   { label: "BF16", shards: 2 }  -> BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf, ...
//   { label: "X", filename: "..." } / { label: "X", filenames: [...] }  -> verbatim
// A sharded quant lives in a directory named after its label; pass `dir` to
// override. Catalogs list only the quants themselves -- mmproj, imatrix and
// MTP draft files are deliberately left out.
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

  // A catalog entry: an Unsloth GGUF repo plus the upstream Qwen repo vLLM uses.
  model(name, quants):: {
    local this = self,

    name: name,
    gguf_repo_id: "unsloth/" + name + "-GGUF",
    vllm_repo_id: "Qwen/" + name,
    quants: [$.quant(name, q) for q in quants],
    labels: [q.label for q in self.quants],

    // Quant entry by label; errors (rather than silently 404ing at download
    // time) when the label isn't in the repo.
    quant(label)::
      local hits = std.filter(function(q) q.label == label, this.quants);
      if std.length(hits) == 0 then
        error "unknown quant %s for %s (have: %s)" % [label, name, std.join(", ", this.labels)]
      else
        hits[0],

    pick(labels):: [this.quant(l) for l in labels],

    // Model block for the llamacpp backend, from a quant label ...
    gguf(label):: this.gguf_of(this.quant(label)),

    // ... or from an expanded {label, filename[s]} entry, so callers can also
    // use a GGUF that isn't in the catalog.
    gguf_of(q):: {
      name: name + "-" + q.label,
      source: "huggingface",
      repo_id: this.gguf_repo_id,
    } + (if std.objectHas(q, "filenames") then { filenames: q.filenames } else { filename: q.filename }),

    // Model block for the vllm backend (unquantized upstream weights).
    hf():: {
      name: name,
      source: "huggingface",
      repo_id: this.vllm_repo_id,
    },
  },
}
