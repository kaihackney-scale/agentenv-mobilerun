"""The collect step: finding the run's recording and storing it durably."""

from __future__ import annotations

import httpx
import pytest

# The framework half of the plugin. Skipped rather than failed when agentenv-framework
# is absent: it is not on PyPI yet, and the REST client and MCP server tests do not need
# it. A skip is visible in the run; an ImportError at collection takes the whole suite
# down and hides the coverage that does apply.
pytest.importorskip("agent_env", reason="agentenv-framework is not installed")

from agent_env.env.env import DeployedGatewayEnv
from agent_env.task_step import TaskStepContext

from agentenv_mobilerun.steps.collect_recording import CollectRecordingStep


def deployed(**metadata) -> DeployedGatewayEnv:
    return DeployedGatewayEnv(
        env_id="mobilerun",
        env_version=1,
        gateway_url="http://gw",
        mcp_url="http://gw/mcp",
        db_web_url=None,
        sandbox_id="sb-1",
        metadata=metadata or None,
    )


def _step(**kwargs) -> CollectRecordingStep:
    return CollectRecordingStep(id="collect", version=1, **kwargs)


def test_type_is_the_portable_identity():
    assert CollectRecordingStep.type == "mobilerun_collect_recording"


def test_defaults_to_non_fatal():
    # A missing video must not fail an otherwise good run: the demonstration is still
    # scoreable, so the default has to be non-fatal.
    assert _step().fail_task_on_error is False


def test_to_dict_from_dict_round_trip():
    original = _step(key_prefix="runs/mobile", timeout_seconds=90.0)
    restored = CollectRecordingStep.from_dict(original.to_dict())
    assert restored.key_prefix == "runs/mobile"
    assert restored.timeout_seconds == 90.0
    assert restored.fail_task_on_error is False


def test_key_prefix_is_normalised():
    assert _step(key_prefix="/runs/mobile/").key_prefix == "runs/mobile"


def test_find_recording_returns_none_without_metadata():
    context = TaskStepContext(deployed_envs=[deployed(), deployed(other="x")])
    assert _step()._find_recording(context) is None


def test_find_recording_prefers_the_most_recent_deployment():
    # A task that deploys twice must collect the recording the agent was just prompted
    # against, not the first one.
    context = TaskStepContext(
        deployed_envs=[
            deployed(mobilerun_device_id="dev-1", mobilerun_recording_id="rec-1"),
            deployed(mobilerun_device_id="dev-2", mobilerun_recording_id="rec-2"),
        ]
    )
    assert _step()._find_recording(context) == ("dev-2", "rec-2")


def test_find_recording_ignores_a_half_stamped_deployment():
    context = TaskStepContext(deployed_envs=[deployed(mobilerun_device_id="dev-1")])
    assert _step()._find_recording(context) is None


@pytest.mark.asyncio
async def test_execute_is_a_no_op_without_a_recording():
    context = TaskStepContext(deployed_envs=[deployed()])
    result = await _step().execute(context)
    assert result is context
    assert "mobilerun" not in result.metadata


@pytest.mark.asyncio
async def test_execute_stores_both_artifacts_and_records_where(monkeypatch):
    stored: dict[str, bytes] = {}

    class StubStore:
        def put(self, key, data, content_type="application/octet-stream", allow_overwrite=False):
            stored[key] = data
            return f"local://{key}"

    class StubConfig:
        def get_object_store(self):
            return StubStore()

    monkeypatch.setenv("MOBILERUN_API_KEY", "dr_sk_test")
    monkeypatch.setattr("agent_env.config.get_config", lambda: StubConfig())

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/video"):
            return httpx.Response(200, content=b"mp4")
        if path.endswith("/trajectory"):
            return httpx.Response(200, content=b'{"kind":"header"}\n')
        if request.method == "POST":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"status": "completed", "actions": 4})

    monkeypatch.setattr(
        "agentenv_mobilerun.steps.collect_recording.MobileRunClient",
        _client_factory(handler),
    )

    context = TaskStepContext(
        deployed_envs=[deployed(mobilerun_device_id="dev-1", mobilerun_recording_id="rec-1")]
    )
    result = await _step(key_prefix="recordings").execute(context)

    assert stored["recordings/rec-1/session.mp4"] == b"mp4"
    assert stored["recordings/rec-1/trajectory.jsonl"] == b'{"kind":"header"}\n'
    entry = result.metadata["mobilerun"]["rec-1"]
    assert entry["device_id"] == "dev-1"
    assert entry["actions"] == 4
    assert entry["video_url"] == "local://recordings/rec-1/session.mp4"


@pytest.mark.asyncio
async def test_execute_continues_when_the_recording_was_already_stopped(monkeypatch):
    """Stopping an already-stopped recording is the normal case once the env has closed."""
    stored = {}

    class StubStore:
        def put(self, key, data, content_type="application/octet-stream", allow_overwrite=False):
            stored[key] = data
            return f"local://{key}"

    monkeypatch.setenv("MOBILERUN_API_KEY", "dr_sk_test")
    monkeypatch.setattr(
        "agent_env.config.get_config", lambda: type("C", (), {"get_object_store": lambda self: StubStore()})()
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(409, json={"detail": "recording already stopped"})
        if request.url.path.endswith("/video"):
            return httpx.Response(200, content=b"mp4")
        if request.url.path.endswith("/trajectory"):
            return httpx.Response(200, content=b"{}")
        return httpx.Response(200, json={"status": "completed"})

    monkeypatch.setattr(
        "agentenv_mobilerun.steps.collect_recording.MobileRunClient", _client_factory(handler)
    )

    context = TaskStepContext(
        deployed_envs=[deployed(mobilerun_device_id="dev-1", mobilerun_recording_id="rec-1")]
    )
    await _step().execute(context)
    assert stored  # the 409 on stop did not abort collection


def test_preflight_reports_a_missing_key(monkeypatch):
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    monkeypatch.setattr(
        "agent_env.config.get_config",
        lambda: type("C", (), {"get_secret_store": lambda self: type("S", (), {"get": lambda s, n: None})()})(),
    )
    problems = _step().preflight()
    # Found before a run provisions anything, rather than after an agent has already
    # driven a phone for ten minutes.
    assert problems and "MOBILERUN_API_KEY" in problems[0]


def test_preflight_is_quiet_when_configured(monkeypatch):
    monkeypatch.setenv("MOBILERUN_API_KEY", "dr_sk_test")
    assert _step().preflight() == []


def _client_factory(handler):
    """A drop-in MobileRunClient bound to a mock transport."""
    from agentenv_mobilerun.client import MobileRunClient

    def factory(api_key, **kwargs):
        kwargs.pop("transport", None)
        return MobileRunClient(api_key, transport=httpx.MockTransport(handler), **kwargs)

    return factory


@pytest.mark.asyncio
async def test_storing_the_same_recording_twice_succeeds(monkeypatch):
    """A retried step must be able to re-store.

    The object key is derived from the recording id alone, so a retry writes the same
    key. Without allow_overwrite the second attempt raises ObjectAlreadyExistsError and
    the step can never recover the recording it exists to preserve.
    """
    written = {}

    class StubStore:
        def put(self, key, data, content_type="application/octet-stream", allow_overwrite=False):
            if key in written and not allow_overwrite:
                raise FileExistsError(f"{key} already exists")
            written[key] = data
            return f"local://{key}"

    monkeypatch.setenv("MOBILERUN_API_KEY", "dr_sk_test")
    monkeypatch.setattr(
        "agent_env.config.get_config",
        lambda: type("C", (), {"get_object_store": lambda self: StubStore()})(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/video"):
            return httpx.Response(200, content=b"mp4")
        if request.url.path.endswith("/trajectory"):
            return httpx.Response(200, content=b"{}")
        if request.method == "POST":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"status": "completed"})

    monkeypatch.setattr(
        "agentenv_mobilerun.steps.collect_recording.MobileRunClient", _client_factory(handler)
    )

    def fresh_context():
        return TaskStepContext(
            deployed_envs=[deployed(mobilerun_device_id="dev-1", mobilerun_recording_id="rec-1")]
        )

    await _step().execute(fresh_context())
    await _step().execute(fresh_context())
    assert written["mobilerun-recordings/rec-1/session.mp4"] == b"mp4"
