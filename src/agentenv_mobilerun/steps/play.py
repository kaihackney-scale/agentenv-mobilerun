"""``mobilerun_play`` — seat a model on the phone and let it work the task.

The model drives the deployed env's ``mobilerun_*`` MCP tools by function calling, one tool per
turn, until it answers without a tool call (it is done) or runs out of turns. No A2A agent is
involved: any model behind an OpenAI-compatible ``/v1/chat/completions`` endpoint will do, which
is the framework's standard model endpoint (``[model] base_url``/``api_key`` in
``.agentenv/config.toml``, or ``LITELLM_BASE_URL``/``LITELLM_API_KEY``).

The model: ``agent-env run --model`` if given, else the step's ``model``. Proxies that need
headers on every request (attribution, routing) get them from the step's ``headers`` or the
``MOBILERUN_MODEL_HEADERS`` environment variable, a JSON object, so a bundle need not carry
them.

A clean start: an app reopens on whatever screen it was left on, so a phone that already ran
the task can hand the next model its answer. ``stop_apps`` names apps to force-quit before the
episode (MobileRun's stop, which clears no data), and the episode then starts from the home
screen.

What it records: ``context.metadata["mobilerun_play"]``, the transcript a run leaves behind,
with what the model wrote before each action, every tool call and its result, wall-clock times,
and how the episode ended. Screenshots are shown to the model but not stored; the env's own
recording is the record of the screen. Pair it with ``mobilerun_check_screen`` to grade the run.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from typing import Any, Awaitable, Callable, Optional

import httpx

from agent_env.task_step import TaskStep, TaskStepContext

logger = logging.getLogger(__name__)

HEADERS_ENV = "MOBILERUN_MODEL_HEADERS"
METADATA_KEY = "mobilerun_play"

SYSTEM_PROMPT = (
    "You are an agent operating a real phone through the mobilerun_* tools. Before each action, write one short "
    "sentence saying what you see and what you will do next, then call exactly one tool.\n"
    "Coordinates: take tap and swipe targets from mobilerun_ui_state element centres, which are in the phone's "
    "screenshot pixel space. The screenshots you are shown are scaled to half size, so if you read a point off an "
    "image, double it.\n"
    "Open apps by tapping their home-screen icon. Use mobilerun_press_key with key 'home' to go to the home screen. "
    "When the task is done, reply with a final message starting with DONE and no tool call."
)

#: (messages, tools) -> the assistant message; the seam the tests drive.
Chat = Callable[[list[dict], list[dict]], Awaitable[dict]]


def openai_tool(tool: Any) -> dict:
    return {"type": "function", "function": {"name": tool.name, "description": (tool.description or "")[:1000],
            "parameters": tool.inputSchema or {"type": "object", "properties": {}}}}


def half_size_jpeg(data_b64: str) -> str:
    """A screenshot at half size, for the model's eyes: a full 1179x2556 frame costs tokens and buys
    nothing ``ui_state`` doesn't already give in exact coordinates."""
    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(data_b64))).convert("RGB")
    image = image.resize((max(1, image.width // 2), max(1, image.height // 2)))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


async def play(session: Any, chat: Chat, prompt: str, *, max_steps: int, system_prompt: str = SYSTEM_PROMPT) -> dict:
    """Run one episode against an initialised MCP ``session``; return its transcript."""
    tools = [openai_tool(t) for t in (await session.list_tools()).tools]
    messages: list[dict] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    events: list[dict] = []
    tool_calls, ended = 0, "max_steps"

    def log(kind: str, **fields: Any) -> None:
        events.append({"t": time.time(), "kind": kind, **fields})

    for _ in range(max_steps):
        message = await chat(messages, tools)
        text = (message.get("content") or "").strip()
        calls = message.get("tool_calls") or []
        messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls")})
        if not calls:
            log("final", text=text)
            ended = "done"
            break
        if text:
            log("thought", text=text)
        call, extras = calls[0], calls[1:]
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except ValueError:
            args = {}
        log("call", name=name, args=args)
        tool_calls += 1
        result = await session.call_tool(name, args)
        image = next((c for c in result.content if getattr(c, "type", "") == "image"), None)
        if image is not None:
            log("result", name=name, text="screenshot")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": "Screenshot captured, shown below at half size."})
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + half_size_jpeg(image.data)}}]})
        else:
            text = "".join(getattr(c, "text", "") for c in result.content)
            log("result", name=name, text=text[:2000])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": text[:12000]})
        # One action per turn, so each one is seen on screen before the next is chosen. Every tool call the
        # model made still needs an answer, or the next request is malformed.
        for extra in extras:
            messages.append({"role": "tool", "tool_call_id": extra["id"], "content": "Not run: one tool call per turn."})
    return {"prompt": prompt, "ended": ended, "tool_calls": tool_calls, "events": events}


def model_headers(step_headers: Optional[dict]) -> dict[str, str]:
    headers = dict(step_headers or {})
    raw = (os.environ.get(HEADERS_ENV) or "").strip()
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"{HEADERS_ENV} must be a JSON object of header names to values")
        headers.update({str(k): str(v) for k, v in parsed.items()})
    return headers


def http_chat(base_url: str, api_key: str, model: str, *, headers: dict[str, str], max_tokens: int,
              client: httpx.AsyncClient) -> Chat:
    url = base_url.rstrip("/")
    url = url if url.endswith("/v1") else url + "/v1"

    async def chat(messages: list[dict], tools: list[dict]) -> dict:
        response = await client.post(
            f"{url}/chat/completions",
            json={"model": model, "messages": messages, "tools": tools, "tool_choice": "auto", "max_tokens": max_tokens},
            headers={"Authorization": f"Bearer {api_key}", **headers},
            timeout=300,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"model endpoint answered {response.status_code}: {response.text[:500]}")
        return response.json()["choices"][0]["message"]

    return chat


class PlayStep(TaskStep):
    """A model works a task on the phone the env's deploy step put up."""

    type = "mobilerun_play"

    def __init__(self, id, version, *, prompt: str, env_step_id: Optional[str] = None, model: Optional[str] = None,
                 max_steps: int = 40, max_tokens: int = 1024, headers: Optional[dict] = None,
                 stop_apps: Optional[list] = None, depends_on=None, fail_task_on_error: bool = True) -> None:
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.prompt, self.env_step_id, self.model = prompt, env_step_id, model
        self.max_steps, self.max_tokens, self.headers = int(max_steps), int(max_tokens), dict(headers or {})
        self.stop_apps = [str(a) for a in (stop_apps or [])]

    def to_dict(self) -> dict:
        data = {**super().to_dict(), "prompt": self.prompt, "max_steps": self.max_steps, "max_tokens": self.max_tokens}
        for key in ("env_step_id", "model"):
            if getattr(self, key):
                data[key] = getattr(self, key)
        if self.headers:
            data["headers"] = self.headers
        if self.stop_apps:
            data["stop_apps"] = self.stop_apps
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PlayStep":
        return cls(**cls._base_from_dict(data), prompt=data["prompt"], env_step_id=data.get("env_step_id"),
                   model=data.get("model"), max_steps=data.get("max_steps", 40), max_tokens=data.get("max_tokens", 1024),
                   headers=data.get("headers"), stop_apps=data.get("stop_apps"))

    def preflight(self) -> list[str]:
        problems = []
        if not (self.prompt or "").strip():
            problems.append("prompt: give the model a task")
        if self.max_steps < 1:
            problems.append("max_steps must be at least 1")
        try:
            model_headers(self.headers)
        except ValueError as exc:
            problems.append(str(exc))
        return problems

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.config import get_config
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        model = resolve_model(context, self.model)
        deployed = deployed_env(context, self.env_step_id)
        await start_clean(deployed, self.stop_apps)
        config = get_config()
        async with httpx.AsyncClient() as client:
            chat = http_chat(config.get_litellm_base_url(), config.get_litellm_api_key(), model,
                             headers=model_headers(self.headers), max_tokens=self.max_tokens, client=client)
            async with streamablehttp_client(deployed.mcp_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info("mobilerun_play: %s working on %s (at most %d turns)", model, deployed.env_id, self.max_steps)
                    transcript = await play(session, chat, self.prompt, max_steps=self.max_steps)
        context.metadata[METADATA_KEY] = {"model": model, "env_id": deployed.env_id, **transcript}
        logger.info("mobilerun_play: %s after %d tool calls", transcript["ended"], transcript["tool_calls"])
        return context


async def start_clean(deployed: Any, stop_apps: list[str], *, client: Any = None) -> None:
    """Force-quit ``stop_apps`` on the deployed phone, then go to the home screen."""
    from agentenv_mobilerun.api_key import resolve_api_key
    from agentenv_mobilerun.client import MobileRunClient, MobileRunError

    device_id = (deployed.metadata or {}).get("mobilerun_device_id")
    if not device_id:
        raise RuntimeError(f"env {deployed.env_id!r} has no mobilerun_device_id; is it a MobileRun env?")
    own = client is None
    client = client or MobileRunClient(resolve_api_key())
    try:
        for app in stop_apps:
            try:
                await client.stop_app(device_id, app)          # clear_data stays False: a rented phone's data is not ours
            except MobileRunError as exc:                     # not running is fine; anything else is worth a line
                logger.info("mobilerun_play: stopping %s: %s", app, exc)
        await client.press_home(device_id)
    finally:
        if own:
            await client.aclose()


def resolve_model(context: TaskStepContext, configured: Optional[str]) -> str:
    """``agent-env run --model`` wins, then the step's own ``model``, then the run's default agent model."""
    model = getattr(context, "agent_model", None) or configured or getattr(context, "default_agent_model", None)
    if not model:
        raise ValueError("no model: pass --model to agent-env run, or set model on the mobilerun_play step")
    return model


def deployed_env(context: TaskStepContext, env_step_id: Optional[str]) -> Any:
    """The env a ``deploy_env`` step put up in this run: the one ``env_step_id`` names, else the latest."""
    matches = [d for d in context.deployed_envs
               if env_step_id is None or (d.metadata or {}).get("deploy_step_id") == env_step_id]
    if not matches:
        which = f"from step {env_step_id!r}" if env_step_id else "at all"
        raise RuntimeError(f"no env deployed {which} in this run; put a deploy_env step before this one")
    deployed = matches[-1]
    if not deployed.mcp_url:
        raise RuntimeError(f"env {deployed.env_id!r} was deployed without an MCP endpoint")
    return deployed
