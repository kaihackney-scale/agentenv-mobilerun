Seat a model on a real MobileRun phone, have it get walking directions in Maps, and grade the screen it leaves.

Needs `MOBILERUN_API_KEY`, one ready device on the account, `agent-env mobilerun setup --record` once to build and
register the MCP server image, and a model endpoint: `[model]` in `.agentenv/config.toml`, or `LITELLM_BASE_URL` and
`LITELLM_API_KEY`. `--model` picks the model; the task's default is a LiteLLM id, so change it to one your endpoint
serves. `agent-env mobilerun doctor` checks all of it and exits non-zero if the run would fail.
