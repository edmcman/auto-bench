// Benchmark all 22 quants of Qwen3.5-2B on 10 SWE-bench instances
// Usage: auto-bench run configs/qwen-2b-quant-sweep-10inst.jsonnet
local base_sweep = import 'qwen-2b-quant-sweep.jsonnet';
local instance_ids = [
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
];
std.map(
  function(entry) entry {
    name: std.strReplace(entry.name, "qwen-2b-quant-sweep", "qwen-2b-quant-sweep-10inst"),
    instance_ids: instance_ids,
    sampling+: { max_tokens: 32768 },
    agent+: { attempts: 1, limit: 10 },
  },
  base_sweep
)
