# Examples

## 01 — open Settings, with an A2A agent

The bundle (`agent-env run mobilerun`) seats a model on the phone directly. This is the other
way: deploy the phone, deploy an A2A agent against it, prompt it, then collect the recording.

**You need an A2A agent registered first.** The `mobilerun-deploy-agent` step below carries no
`a2a_agent_id`, so it deploys whatever `[agents] default_a2a_agent_id` names (built-in default:
`a2a-default`). agent-env does not ship an agent of its own — the in-core agent gateway and its
`claude_code` / `claude_cua` harnesses were retired in favour of native A2A agents, so the agent
is yours to supply. The protocol package's `a2a_agent` SDK
(`pip install 'agentenv-framework-protocol[agent]'`) is one way to build one; register it with
`agent-env a2a-agent put --id a2a-default --dockerfile path/to/its/Dockerfile`, or point the step at
your own with `"a2a_agent_id": "<id>"`.

This plugin supplies the *tools*, not the agent: the 18 `mobilerun_*` MCP tools are what any
agent drives the phone through.

```bash
export MOBILERUN_API_KEY=dr_sk_...

agent-env mobilerun doctor --skip-model     # this example needs an agent, not a model endpoint
agent-env mobilerun setup --id mobilerun --platform ios --record

agent-env task create examples/tasks/01-open-settings.json \
  --id mobilerun-demo-01 --project-id <your-project-id>
agent-env task run --id mobilerun-demo-01
```

Drop the `mobilerun-collect` step (and the `--record` flag) if you do not want a recording.

### Scoring

This example stops at "the agent did something and it was recorded"; it does not score the run.
To grade it, add the plugin's `mobilerun_check_screen` step after the prompt, the way the bundle's
`maps-walking` task does: it reads the final screen's `ui_state` and scores 1 when the text you
name is on it.

### Notes on the prompt

Two things in the prompt above matter more than they look:

- **"Take a screenshot first and look at it before every action."** A mobile UI animates.
  Acting on a stale screen is the most common failure mode, and `mobilerun_wait` exists so an
  agent can let a screen settle instead of poking it again.
- **"Coordinates ... do not rescale them."** The screenshot is the device's own pixels and the
  tap tool takes the same space. Models that have been trained around downscaled desktop
  screenshots will sometimes try to do scale maths anyway; saying so plainly prevents it.
