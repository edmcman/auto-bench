// Benchmark all 22 quants of Qwen3.5-2B on 10 SWE-bench instances
// Usage: auto-bench run configs/qwen-2b-quant-sweep-10inst.jsonnet
local d = import 'lib/qwen35.libsonnet';
local quants = import 'lib/quants.libsonnet';
local entries = quants.qwen_2b;

local base = d {
  name: "qwen-2b-quant-sweep-10inst",
  instance_ids: [
    "swe-bench/pylint-dev__pylint-7080",
    "swe-bench/django__django-11734",
    "swe-bench/astropy__astropy-14508",
    "swe-bench/scikit-learn__scikit-learn-26323",
    "swe-bench/django__django-14034",
    "swe-bench/django__django-13741",
    "swe-bench/django__django-13417",
    "swe-bench/django__django-12125",
    "swe-bench/scikit-learn__scikit-learn-25973",
    "swe-bench/django__django-11532",
  ],
  model: {
    name: "Qwen3.5-2B",
    source: "huggingface",
    repo_id: "unsloth/Qwen3.5-2B-GGUF",
    filename: "Qwen3.5-2B-UD-Q5_K_XL.gguf",
  },
  sampling+: { max_tokens: 32768 },
  agent+: { attempts: 1, limit: 10 },
  remove_downloaded_models: false,
};

std.map(
  function(q)
    base {
      name+: "-" + q.label,
      model+: { filename: q.filename },
    },
  entries
)