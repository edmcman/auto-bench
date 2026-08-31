// Qwen3.5-35B-A3B BF16, thinking-general only, on all SWE-bench Verified instances.
// Extracted from qwen-35b-a3b-quant-sweep.jsonnet's full sweep.
local full_sweep = import 'qwen-35b-a3b-quant-sweep.jsonnet';

local matches = std.filter(
  function(c) c.name == "qwen-35b-a3b-quant-sweep-thinking-general-BF16",
  full_sweep
);

assert std.length(matches) == 1 : "expected exactly one matching entry in qwen-35b-a3b-quant-sweep.jsonnet";

matches[0]
