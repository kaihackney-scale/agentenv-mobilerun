Drive a real MobileRun cloud phone: deploy the env, hand it to an agent, collect the recording.

Needs `MOBILERUN_API_KEY` and one ready device on the account, `agent-env mobilerun setup` once
to build and register the MCP server image, and an A2A agent: the task deploys your configured
default (`[agents] default_a2a_agent_id`, else `a2a-default`), and agent-env ships none, so on a
clean install register one with `agent-env a2a-agent put --id a2a-default --dockerfile ...`.
`agent-env mobilerun doctor` checks all of it and exits non-zero if the run would fail.
