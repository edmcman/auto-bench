// Smoke-test with 4x parallelism (4 parallel slots, 4 parallel Harbor trials)
// Usage: auto-bench run configs/smoke-test-4x.jsonnet
local parallel = import 'lib/parallel.libsonnet';
local base = import 'smoke-test.jsonnet';
parallel.apply(base { name: "smoke-test-4x", agent+: { attempts: 4 } }, 4)
