// Parallelism modifier for experiment configs.
// Increases parallel Harbor runs and backend parallel slots.
// ctx_size is the per-slot size; the llamacpp backend passes parallel*ctx_size.
//
// Usage:
//   local parallel = import 'lib/parallel.libsonnet';
//   parallel.apply(base_config, 4)  // 4x parallelism
//
// apply(config, n):
//   - agent.trials = n
//   - backend_options.parallel = n
{
  apply(config, n):
    config {
      agent+: { trials: n },
      backend_options+: { parallel: n },
    },
}