# Examples

## 01 — open Settings

A minimal end-to-end run: deploy the phone, deploy an agent against it, prompt it, then
collect the recording.

**You need an A2A agent registered first.** The `mobilerun-deploy-agent` step below carries no
`a2a_agent_id`, so it deploys whatever `[agents] default_a2a_agent_id` names (built-in default:
`a2a-default`). agent-env does not ship an agent of its own — the in-core agent gateway and its
`claude_code` / `claude_cua` harnesses were retired in favour of native A2A agents, so the agent
is yours to supply. Build one on `agentenv-protocol`'s `a2a_agent` framework (see
`packages/agentenv-protocol/README.md`, "A2A agent framework") and register it with
`A2AAgent.put(id="a2a-default", docker_image_artifact=...)`, or point the step at your own with
`"a2a_agent_id": "<id>"`.

This plugin supplies the *tools*, not the agent: the 18 `mobilerun_*` MCP tools are what any
agent drives the phone through.

```bash
export MOBILERUN_API_KEY=dr_sk_...

agent-env mobilerun doctor
agent-env mobilerun setup --id mobilerun --platform android --record

agent-env task create examples/tasks/01-open-settings.json \
  --id mobilerun-demo-01 --project-id <your-project-id>
agent-env task run --id mobilerun-demo-01
```

Drop the `mobilerun-collect` step (and the `--record` flag) if you do not want a recording.

### Scoring

This example deliberately stops at "the agent did something and it was recorded" — it does
not score the run. Add agent-env's own `rubrics_verifier` step to grade it; that step needs a
stored rubric prompt artifact, which is agent-env configuration rather than anything this
plugin owns, so see the agent-env docs for its schema rather than copying a guess from here.

### Notes on the prompt

Two things in the prompt above matter more than they look:

- **"Take a screenshot first and look at it before every action."** A mobile UI animates.
  Acting on a stale screen is the most common failure mode, and `mobilerun_wait` exists so an
  agent can let a screen settle instead of poking it again.
- **"Coordinates ... do not rescale them."** The screenshot is the device's own pixels and the
  tap tool takes the same space. Models that have been trained around downscaled desktop
  screenshots will sometimes try to do scale maths anyway; saying so plainly prevents it.
