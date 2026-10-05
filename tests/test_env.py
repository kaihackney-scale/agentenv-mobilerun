"""MobileRunEnv: serialisation, key resolution, and what reaches the container."""

from __future__ import annotations

import asyncio
import json

import pytest

# The framework half of the plugin. Skipped rather than failed when agentenv-framework
# is absent: it is not on PyPI yet, and the REST client and MCP server tests do not need
# it. A skip is visible in the run; an ImportError at collection takes the whole suite
# down and hides the coverage that does apply.
pytest.importorskip("agent_env", reason="agentenv-framework is not installed")

from agentenv_mobilerun.client import MobileRunError
from agentenv_mobilerun.env import MobileRunEnv, resolve_api_key


class FakeArtifact:
    """Stands in for a DockerImageArtifact without touching the artifact store."""

    id = "mcp-server-mobilerun"
    version = 3
    type = "docker_image"
    image_name = "mcp-server-mobilerun:v3"


def _env(**kwargs) -> MobileRunEnv:
    defaults = dict(id="mobilerun", version=1, docker_image_artifact=FakeArtifact())
    defaults.update(kwargs)
    return MobileRunEnv(**defaults)


def test_type_is_the_portable_identity():
    # The [envs] seam registers a custom env under its own `type`, and a class that
    # inherits the base default is rejected at registration.
    assert MobileRunEnv.type == "mobilerun"
    from agent_env.env.env import Env

    assert MobileRunEnv.type != Env.type


def test_unknown_platform_is_rejected_at_construction():
    with pytest.raises(ValueError, match="platform must be"):
        _env(platform="windows-phone")


def test_platform_is_normalised():
    assert _env(platform="iOS").platform == "ios"


def test_to_dict_omits_an_unset_device_id():
    # Writing device_id=None would be harmless, but writing a *stale* one pins every
    # future deploy to a phone the account may no longer have.
    assert "device_id" not in _env().to_dict()
    assert _env(device_id="dev-7").to_dict()["device_id"] == "dev-7"


def test_to_dict_carries_the_fields_from_dict_needs():
    data = _env(platform="ios", device_id="dev-7", record=True).to_dict()
    assert data["type"] == "mobilerun"
    assert data["platform"] == "ios"
    assert data["record"] is True
    assert data["docker_image_artifact"]["id"] == "mcp-server-mobilerun"


def test_from_dict_round_trips(monkeypatch):
    artifact = FakeArtifact()
    monkeypatch.setattr("agent_env.artifact.Artifact.get", classmethod(lambda cls, i, version=None: artifact))

    original = _env(platform="ios", device_id="dev-7", record=True, metadata={"team": "env-pod"})
    restored = MobileRunEnv.from_dict(original.to_dict())

    assert (restored.platform, restored.device_id, restored.record) == ("ios", "dev-7", True)
    assert restored.metadata == {"team": "env-pod"}


def test_construction_does_not_require_a_compute_config():
    # from_dict runs wherever an env document is read, including processes that will never
    # deploy. Building a GatewayProvider eagerly would make reading a document fail there.
    env = _env()
    assert env._gateway_provider_instance is None


def test_resolve_api_key_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("MOBILERUN_API_KEY", "dr_sk_from_env")
    assert resolve_api_key() == "dr_sk_from_env"


def test_resolve_api_key_ignores_a_blank_variable(monkeypatch):
    # An exported-but-empty variable is a very common shell mistake; treating it as a key
    # produces a 401 far from its cause.
    monkeypatch.setenv("MOBILERUN_API_KEY", "   ")
    monkeypatch.setattr(
        "agent_env.config.get_config", lambda: _StubConfig("dr_sk_from_store")
    )
    assert resolve_api_key() == "dr_sk_from_store"


def test_resolve_api_key_falls_back_to_the_secret_store(monkeypatch):
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    monkeypatch.setattr("agent_env.config.get_config", lambda: _StubConfig("dr_sk_from_store"))
    assert resolve_api_key() == "dr_sk_from_store"


def test_resolve_api_key_error_names_both_ways_to_fix_it(monkeypatch):
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    monkeypatch.setattr("agent_env.config.get_config", lambda: _StubConfig(None))
    with pytest.raises(MobileRunError) as excinfo:
        resolve_api_key()
    message = str(excinfo.value)
    assert "MOBILERUN_API_KEY" in message and "secret store" in message


def test_resolve_api_key_survives_a_broken_secret_store(monkeypatch):
    # A misconfigured store must not mask the actionable advice with a stack trace about
    # the store itself.
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)

    def boom():
        raise RuntimeError("no config.toml anywhere")

    monkeypatch.setattr("agent_env.config.get_config", boom)
    with pytest.raises(MobileRunError, match="MOBILERUN_API_KEY"):
        resolve_api_key()


def test_mcp_env_passes_the_key_device_and_capabilities():
    env = _env()
    injected = env._mcp_env("dr_sk_x", "dev-1", {"clipboard": True, "esim": False, "recording": True})
    assert injected["MOBILERUN_API_KEY"] == "dr_sk_x"
    assert injected["MOBILERUN_DEVICE_ID"] == "dev-1"
    assert injected["MOBILERUN_CAPABILITIES"] == "clipboard=true,esim=false,recording=true"


def test_mcp_env_sends_the_false_entries_too():
    # The server treats an absent capability as present (so a name this plugin has not
    # heard of does not silently remove a working tool). That makes the `false` entries
    # the only thing the gate can act on — dropping them would make it inert.
    assert _env()._mcp_env("k", "d", {"clipboard": False})["MOBILERUN_CAPABILITIES"] == "clipboard=false"


def test_every_injected_value_is_safe_in_a_compose_environment_block():
    """The regression that broke a real deploy.

    The gateway renders these pairs into a docker-compose ``environment:`` list. A YAML
    plain scalar may not contain ``": "``, so a JSON capability map made ``docker compose
    up`` fail with "mapping values are not allowed in this context" — a parse error with
    no connection to the env var that caused it.
    """
    yaml = pytest.importorskip("yaml")

    injected = _env()._mcp_env(
        "dr_sk_x", "dev-1", {"accessibility": True, "clipboard": False, "recording": True}
    )
    lines = "\n".join(f"      - {k}={v}" for k, v in injected.items())
    document = f"services:\n  mcp:\n    environment:\n{lines}\n"
    parsed = yaml.safe_load(document)

    rendered = parsed["services"]["mcp"]["environment"]
    assert len(rendered) == len(injected)
    for entry in rendered:
        # Each entry must survive as a single plain scalar, not be re-read as a mapping.
        assert isinstance(entry, str), f"{entry!r} parsed as {type(entry).__name__}, not a string"
        assert ": " not in entry


def test_mcp_env_omits_capabilities_when_the_map_is_empty():
    assert "MOBILERUN_CAPABILITIES" not in _env()._mcp_env("k", "d", {})


def test_mcp_env_forwards_a_custom_base_url(monkeypatch):
    monkeypatch.setenv("MOBILERUN_BASE_URL", "https://staging.example/v1")
    assert _env()._mcp_env("k", "d", {})["MOBILERUN_BASE_URL"] == "https://staging.example/v1"


class _StubConfig:
    def __init__(self, value):
        self._value = value

    def get_secret_store(self):
        return self

    def get(self, name):
        return self._value


@pytest.mark.asyncio
async def test_from_deployed_env_restores_enough_to_tear_down(monkeypatch):
    """Reattach must not touch GatewayProvider.sandbox.

    It is a read-only property, so assigning it raises AttributeError — which took out
    reattach and, with it, teardown. Found by actually closing a real deployment.
    """
    from agent_env.env.env import DeployedGatewayEnv

    artifact = FakeArtifact()
    monkeypatch.setattr("agent_env.artifact.Artifact.get", classmethod(lambda cls, i, version=None: artifact))
    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(lambda cls, i, v=None: _env(device_id="dev-7")))

    terminated = []

    class StubSandbox:
        sandbox_id = "sb-1"
        type = "local"

        async def terminate(self):
            terminated.append(True)

    class StubProvider:
        async def get_sandbox(self, sandbox_id):
            return StubSandbox()

    monkeypatch.setattr("agent_env.providers.build_sandbox_provider", lambda t: StubProvider())

    deployed = DeployedGatewayEnv(
        env_id="mobilerun",
        env_version=1,
        gateway_url="http://gw",
        mcp_url="http://gw/mcp",
        db_web_url=None,
        sandbox_id="sb-1",
        sandbox_type="local",
        instance_id="mobilerun-abc",
        metadata={"mobilerun_device_id": "dev-7"},
    )

    env = await MobileRunEnv.from_deployed_env(deployed)
    assert env._resolved_device_id == "dev-7"
    assert env._instance_id == "mobilerun-abc"

    await env.close()
    assert terminated == [True], "close() must terminate the reattached sandbox"


# --------------------------------------------------------------- deploy

# These drive deploy() through the REAL EnvironmentGatewayProvider, replacing only its
# network-facing create_gateway. They used to hand deploy() a MagicMock provider, which
# answers to any name and accepts any kwargs -- so when agent-env deleted the
# `GatewayProvider` name, every real deploy raised ImportError and all of these passed.


def test_the_gateway_provider_is_the_framework_class():
    """No mocks: the property deploy() reads must resolve against the installed agent-env."""
    from agent_env.providers import EnvironmentGatewayProvider

    provider = _env()._gateway_provider
    assert isinstance(provider, EnvironmentGatewayProvider)
    # The private name deploy() records sandbox_ids from. If agent-env renames it, this
    # fails here instead of container-mode deploys quietly leaking their sidecars.
    assert callable(getattr(provider, "_sandbox_ids", None))


class _StubSandbox:
    sandbox_id = "sb-gw"
    type = "modal"


_CARD = {
    "name": "mobilerun",
    "protocolVersion": "1.0",
    "url": "/agentenv",
    "preferredTransport": "JSONRPC",
    "additionalInterfaces": [{"url": "/mcp", "transport": "mcp"}],
    "capabilities": {"extensions": [], "operations": []},
}


def _deployable(monkeypatch, *, card=_CARD, extra_sandboxes=None):
    """An env whose deploy() reaches a real EnvironmentGatewayProvider, with only the
    MobileRun API and create_gateway's network stood in for. Returns (env, calls)."""
    from unittest.mock import AsyncMock, MagicMock

    from agent_env.providers import DeployedGateway

    monkeypatch.setenv("MOBILERUN_API_KEY", "dr_sk_test")
    env = _env()

    device = MagicMock(id="dev-1", platform="android", state="ready")
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.resolve_device = AsyncMock(return_value=device)
    client.capabilities = AsyncMock(return_value={"clipboard": True})
    monkeypatch.setattr("agentenv_mobilerun.env.MobileRunClient", lambda *a, **k: client)
    monkeypatch.setattr("agent_env.providers.get_env_sandbox_provider", lambda: MagicMock())
    monkeypatch.setattr("agentenv_mobilerun.env.register_env_instance", lambda d, t: d)

    provider = env._gateway_provider  # the real class
    calls: list[dict] = []

    async def fake_create_gateway(**kwargs):
        calls.append(kwargs)
        provider._sandbox = _StubSandbox()
        provider._environment_sandboxes = dict(extra_sandboxes or {})
        return DeployedGateway(
            gateway_url="https://gw",
            mcp_url="https://gw/mcp",
            db_web_url=None,
            env_state_instance_ids=["state-1"],
            environment_card=card,
            environment_card_read_at_utc="2026-10-05T00:00:00Z" if card else None,
        )

    monkeypatch.setattr(provider, "create_gateway", fake_create_gateway)
    return env, calls


@pytest.mark.asyncio
async def test_deploy_passes_only_arguments_create_gateway_takes(monkeypatch):
    """The fake accepts **kwargs; the real method does not. Checked against its signature."""
    import inspect

    from agent_env.providers import EnvironmentGatewayProvider

    env, calls = _deployable(monkeypatch)
    await env.deploy()
    accepted = inspect.signature(EnvironmentGatewayProvider.create_gateway).parameters
    assert set(calls[0]) <= set(accepted), f"not parameters of create_gateway: {set(calls[0]) - set(accepted)}"
    # No external store handle: the gateway builds a local store this env never reads,
    # until it moves onto the store-less EnvironmentServerProvider.
    assert "state_instance" not in calls[0]


@pytest.mark.asyncio
async def test_deploy_records_what_the_frameworks_own_deploy_records(monkeypatch):
    """The record mirrors EnvironmentGatewayProvider.deploy(): provider type, the env card
    and its address (which is where the agent learns the MCP server's name), and the
    store instances."""
    env, _ = _deployable(monkeypatch)
    deployed = await env.deploy()

    assert deployed.env_provider_type == "gateway"
    assert deployed.environment_card == _CARD
    assert deployed.environment_card_url == "https://gw/.well-known/agent-env.json"
    assert deployed.mcp_server_name == "mobilerun"
    assert deployed.mcp_url == "https://gw/mcp"
    assert deployed.env_state_instance_ids == ["state-1"]


@pytest.mark.asyncio
async def test_deploy_names_every_sandbox_so_teardown_can_reach_it(monkeypatch):
    """Run-end teardown terminates exactly the sandboxes the record names. On a container
    backend the MCP server is a sandbox of its own, and a record that only named the
    gateway left it running until its TTL."""
    from agent_env.task_step.task_steps.teardown_sandboxes import _env_sandbox_ids

    class McpSandbox:
        sandbox_id = "sb-mcp"
        type = "modal"

    env, _ = _deployable(monkeypatch, extra_sandboxes={"mobilerun": McpSandbox()})
    deployed = await env.deploy()

    reached = {sandbox_id for sandbox_id, _ in _env_sandbox_ids(deployed)}
    assert reached == {"sb-gw", "sb-mcp"}


@pytest.mark.asyncio
async def test_a_deploy_whose_card_read_failed_still_has_an_mcp_address(monkeypatch):
    """Card and card URL come as a pair or not at all; without them, mcp_url carries it."""
    env, _ = _deployable(monkeypatch, card=None)
    deployed = await env.deploy()

    assert deployed.environment_card is None and deployed.environment_card_url is None
    assert deployed.mcp_url == "https://gw/mcp"


@pytest.mark.asyncio
async def test_a_failed_stop_recording_keeps_the_id_for_a_retry(monkeypatch):
    """Clearing the id in a `finally` turned a retryable API error into a phone that
    records until its own limits end it: the one handle a later close() could retry with
    was erased on the way out."""
    from agentenv_mobilerun.client import MobileRunError

    env = _env()
    env._resolved_device_id = "dev-7"
    env._recording_id = "rec-1"

    class Boom:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def stop_recording(self, device_id, recording_id):
            raise MobileRunError("503 from the device plane")

    monkeypatch.setattr("agentenv_mobilerun.env.resolve_api_key", lambda name: "dr_sk_x")
    monkeypatch.setattr("agentenv_mobilerun.env.MobileRunClient", lambda *a, **k: Boom())

    await env.close()
    assert env._recording_id == "rec-1", "a failed stop must leave something to retry"


@pytest.mark.asyncio
async def test_a_successful_stop_recording_forgets_the_id(monkeypatch):
    """The control: the id must not be retained after a confirmed stop, or close() would
    keep trying to stop a recording that already ended."""
    env = _env()
    env._resolved_device_id = "dev-7"
    env._recording_id = "rec-1"

    stopped = []

    class Ok:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def stop_recording(self, device_id, recording_id):
            stopped.append(recording_id)

    monkeypatch.setattr("agentenv_mobilerun.env.resolve_api_key", lambda name: "dr_sk_x")
    monkeypatch.setattr("agentenv_mobilerun.env.MobileRunClient", lambda *a, **k: Ok())

    await env.close()
    assert stopped == ["rec-1"] and env._recording_id is None


@pytest.mark.asyncio
async def test_cancellation_during_terminate_is_not_swallowed():
    """BaseException is deliberate there -- a failed terminate leaks a VM, so nothing may
    pass silently -- but cancellation is not an error to log and drop. A timeout or
    shutdown mid-terminate has to propagate."""
    env = _env()

    class Cancelling:
        sandbox_id = "sb-1"

        async def terminate(self):
            raise asyncio.CancelledError()

    env._sandbox = Cancelling()
    with pytest.raises(asyncio.CancelledError):
        await env.close()
    assert env._sandbox is not None, "the handle stays for a later retry"
