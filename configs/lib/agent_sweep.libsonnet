// Sweep a single base config across multiple Harbor agents.
// Each entry in `agents` is either a plain agent name string, or an object
// {agent: "...", ...other AgentConfig overrides} for per-agent tuning (e.g.
// {agent: "openhands", max_iterations: 50}). The base config's own `agent`
// fields (env, attempts, setup_multiplier, ...) are preserved unless
// overridden here.
//
// Usage:
//   local agent_sweep = import 'lib/agent_sweep.libsonnet';
//   agent_sweep.apply(base_config, ["openhands", "opencode", "mini-swe-agent"])
{
  apply(config, agents):
    std.map(
      function(a)
        local spec = if std.isString(a) then { agent: a } else a;
        config {
          name: config.name + "-" + spec.agent,
          agent+: spec,
        },
      agents
    ),
}
