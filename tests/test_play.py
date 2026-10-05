"""``mobilerun_play``: a model seated on the phone, driven by function calling.

The episode loop runs against a fake MCP session and a scripted model, so every branch is
deterministic; the HTTP client runs against ``httpx.MockTransport``, so the request a real
endpoint receives is checked byte for byte.
"""

from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("agent_env", reason="agentenv-framework is not installed")

from agentenv_mobilerun.steps.play import (  # noqa: E402
    HEADERS_ENV,
    PlayStep,
    deployed_env,
    http_chat,
    model_headers,
    play,
    resolve_model,
)


def _png(width=40, height=80) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeSession:
    """The two MCP calls the loop makes, recorded."""

    def __init__(self):
        self.calls = []

    async def list_tools(self):
        tool = lambda name: SimpleNamespace(name=name, description=f"{name} tool", inputSchema={"type": "object", "properties": {}})
        return SimpleNamespace(tools=[tool("mobilerun_screenshot"), tool("mobilerun_tap")])

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "mobilerun_screenshot":
            return SimpleNamespace(content=[SimpleNamespace(type="image", data=_png())])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps({"ok": True, "action": "tap"}))])


def scripted(*replies):
    """A model that answers with these assistant messages in order, and records what it was sent."""
    seen = []

    async def chat(messages, tools):
        seen.append({"messages": [dict(m) for m in messages], "tools": tools})
        return replies[len(seen) - 1]

    chat.seen = seen
    return chat


def call(name, args=None, id="c1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args or {})}}


@pytest.mark.asyncio
async def test_an_episode_ends_when_the_model_answers_without_a_tool_call():
    session = FakeSession()
    chat = scripted(
        {"role": "assistant", "content": "The home screen; I'll tap Maps.", "tool_calls": [call("mobilerun_tap", {"x": 1, "y": 2})]},
        {"role": "assistant", "content": "DONE. Directions are on screen."},
    )
    transcript = await play(session, chat, "get directions", max_steps=10)
    assert transcript["ended"] == "done" and transcript["tool_calls"] == 1
    assert session.calls == [("mobilerun_tap", {"x": 1, "y": 2})]
    assert [e["kind"] for e in transcript["events"]] == ["thought", "call", "result", "final"]
    assert transcript["events"][-1]["text"].startswith("DONE")


@pytest.mark.asyncio
async def test_one_action_per_turn_and_every_tool_call_still_gets_an_answer():
    """A second call in the same turn is not run -- each action is seen on screen before the next
    is chosen -- but it must be answered, or the next request to the model is malformed."""
    session = FakeSession()
    chat = scripted(
        {"role": "assistant", "content": "", "tool_calls": [call("mobilerun_tap", {"x": 1}, "a"), call("mobilerun_tap", {"x": 2}, "b")]},
        {"role": "assistant", "content": "DONE"},
    )
    await play(session, chat, "t", max_steps=5)
    assert session.calls == [("mobilerun_tap", {"x": 1})]
    answers = {m["tool_call_id"]: m["content"] for m in chat.seen[1]["messages"] if m["role"] == "tool"}
    assert set(answers) == {"a", "b"} and answers["b"].startswith("Not run")


@pytest.mark.asyncio
async def test_a_screenshot_reaches_the_model_as_a_half_size_image():
    session = FakeSession()
    chat = scripted({"role": "assistant", "content": "", "tool_calls": [call("mobilerun_screenshot")]},
                    {"role": "assistant", "content": "DONE"})
    transcript = await play(session, chat, "t", max_steps=5)
    image_turn = chat.seen[1]["messages"][-1]
    assert image_turn["role"] == "user"
    url = image_turn["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    from PIL import Image
    assert Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size == (20, 40)
    # The transcript does not keep the image; the env's own recording is the record of the screen.
    assert all("data:image" not in json.dumps(e) for e in transcript["events"])


@pytest.mark.asyncio
async def test_an_episode_stops_at_max_steps():
    session = FakeSession()
    loop = {"role": "assistant", "content": "", "tool_calls": [call("mobilerun_tap")]}
    transcript = await play(session, scripted(*[loop] * 3), "t", max_steps=3)
    assert transcript["ended"] == "max_steps" and transcript["tool_calls"] == 3


@pytest.mark.asyncio
async def test_the_model_request_carries_the_key_the_headers_and_the_tools():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"], seen["body"] = str(request.url), request.headers, json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "DONE"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = http_chat("https://models.example", "sk-test", "some-model", headers={"x-project": "p1"},
                         max_tokens=512, client=client)
        message = await chat([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "t"}}])
    assert message["content"] == "DONE"
    assert seen["url"] == "https://models.example/v1/chat/completions"     # /v1 added once
    assert seen["headers"]["authorization"] == "Bearer sk-test" and seen["headers"]["x-project"] == "p1"
    assert seen["body"]["model"] == "some-model" and seen["body"]["tools"] and "temperature" not in seen["body"]


@pytest.mark.asyncio
async def test_an_endpoint_error_says_what_the_endpoint_said():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(400, text="no project attribution"))) as client:
        chat = http_chat("https://models.example/v1", "k", "m", headers={}, max_tokens=1, client=client)
        with pytest.raises(RuntimeError, match="400: no project attribution"):
            await chat([], [])


def test_headers_from_the_environment_extend_the_steps_own(monkeypatch):
    monkeypatch.setenv(HEADERS_ENV, json.dumps({"x-litellm-customer-id": "p1"}))
    assert model_headers({"x-a": "1"}) == {"x-a": "1", "x-litellm-customer-id": "p1"}
    monkeypatch.setenv(HEADERS_ENV, "[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        model_headers(None)


def test_the_run_model_beats_the_steps_and_a_missing_model_is_an_error():
    assert resolve_model(SimpleNamespace(agent_model="from-run", default_agent_model=None), "from-step") == "from-run"
    assert resolve_model(SimpleNamespace(agent_model=None, default_agent_model="default"), "from-step") == "from-step"
    assert resolve_model(SimpleNamespace(agent_model=None, default_agent_model="default"), None) == "default"
    with pytest.raises(ValueError, match="--model"):
        resolve_model(SimpleNamespace(agent_model=None, default_agent_model=None), None)


def test_the_step_round_trips_and_preflights():
    step = PlayStep("play", 1, prompt="do it", env_step_id="deploy", model="m", max_steps=7, headers={"x": "y"})
    again = PlayStep.from_dict(step.to_dict())
    assert (again.prompt, again.env_step_id, again.model, again.max_steps, again.headers) == ("do it", "deploy", "m", 7, {"x": "y"})
    assert again.preflight() == []
    assert any("prompt" in p for p in PlayStep("p", 1, prompt=" ", max_steps=0).preflight())


def test_the_env_is_found_by_its_deploy_step():
    a = SimpleNamespace(env_id="a", mcp_url="http://a/mcp", metadata={"deploy_step_id": "one"})
    b = SimpleNamespace(env_id="b", mcp_url="http://b/mcp", metadata={"deploy_step_id": "two"})
    context = SimpleNamespace(deployed_envs=[a, b])
    assert deployed_env(context, "one") is a
    assert deployed_env(context, None) is b
    with pytest.raises(RuntimeError, match="deploy_env"):
        deployed_env(context, "three")


class FakeClient:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    async def stop_app(self, device_id, app, clear_data=False):
        from agentenv_mobilerun.client import MobileRunError
        self.calls.append(("stop_app", device_id, app, clear_data))
        if app in self.fail:
            raise MobileRunError("not running", status_code=409)

    async def press_home(self, device_id):
        self.calls.append(("press_home", device_id))


@pytest.mark.asyncio
async def test_a_clean_start_stops_the_named_apps_without_clearing_data_then_goes_home():
    """An app reopens where it was left, so a phone that already ran the task would hand the next model its
    answer. A rented phone's data is not ours to clear."""
    from agentenv_mobilerun.steps.play import start_clean

    deployed = SimpleNamespace(env_id="mobilerun", metadata={"mobilerun_device_id": "dev-1"})
    client = FakeClient(fail={"com.apple.Notes"})
    await start_clean(deployed, ["com.apple.Maps", "com.apple.Notes"], client=client)
    assert client.calls == [("stop_app", "dev-1", "com.apple.Maps", False), ("stop_app", "dev-1", "com.apple.Notes", False),
                            ("press_home", "dev-1")]
