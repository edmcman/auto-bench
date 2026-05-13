// General defaults for SWE-bench runs
// Import: local d = import 'lib/defaults.libsonnet';
{
  backend_type: "llamacpp",
  dataset: "SWE-bench/SWE-bench_Verified",
  agent: {
    agent: "openhands",
    env: "docker",
    setup_multiplier: 10.0,
  },
  evaluation: { run_evaluation: true },
}