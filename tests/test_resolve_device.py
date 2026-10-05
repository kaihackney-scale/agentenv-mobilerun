"""Device resolution — the part that decides which phone a run drives.

The behaviour under test is deliberately suspicious of MobileRun's ``state`` field: a
device that reports ``state: "ready"`` with an unchanged ``stateMessage`` can be
completely undrivable, so resolution health-probes before handing a device back.
"""

from __future__ import annotations

import httpx
import pytest

from agentenv_mobilerun.client import DeviceUnavailable
from tests.helpers import png_bytes


def _fleet(*devices: dict) -> httpx.Response:
    return httpx.Response(200, json={"items": list(devices)})


def device(id_: str, *, platform="android", state="ready", active=None) -> dict:
    return {"id": id_, "platform": platform, "state": state, "activeTask": active}


@pytest.mark.asyncio
async def test_pinned_device_is_used_and_probed(make_client):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/screenshot"):
            return httpx.Response(200, content=png_bytes())
        return httpx.Response(200, json=device("pinned", platform="ios"))

    async with make_client(handler) as client:
        resolved = await client.resolve_device(device_id="pinned")

    assert resolved.id == "pinned"
    # The fleet listing is never fetched when a device is pinned.
    assert not any(path.endswith("/devices") for path in calls)


@pytest.mark.asyncio
async def test_pinned_but_undrivable_device_is_refused_with_state_in_the_message(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/screenshot"):
            return httpx.Response(500, text="Failed connecting to Lockdown with error code:2")
        return httpx.Response(200, json=device("sick", state="ready"))

    async with make_client(handler) as client:
        with pytest.raises(DeviceUnavailable) as excinfo:
            await client.resolve_device(device_id="sick")

    message = str(excinfo.value)
    # The message has to name the contradiction, because the dashboard will say "ready"
    # and the user will otherwise assume the plugin is at fault.
    assert "ready" in message and "health probe" in message


@pytest.mark.asyncio
async def test_no_devices_points_at_provisioning(make_client):
    async with make_client(lambda r: _fleet()) as client:
        with pytest.raises(DeviceUnavailable, match="no devices"):
            await client.resolve_device(platform="android")


@pytest.mark.asyncio
async def test_busy_devices_are_reported_with_their_states(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return _fleet(device("a", state="running"), device("b", state="rebooting"))

    async with make_client(handler) as client:
        with pytest.raises(DeviceUnavailable) as excinfo:
            await client.resolve_device(platform="android")

    message = str(excinfo.value)
    assert "a=running" in message and "b=rebooting" in message


@pytest.mark.asyncio
async def test_a_device_with_an_active_task_is_not_idle(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return _fleet(device("busy", active="task-9"))

    async with make_client(handler) as client:
        with pytest.raises(DeviceUnavailable, match="no idle ready device"):
            await client.resolve_device()


@pytest.mark.asyncio
async def test_platform_filter_excludes_other_platforms(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/screenshot"):
            return httpx.Response(200, content=png_bytes())
        return _fleet(device("droid", platform="android"), device("phone", platform="ios"))

    async with make_client(handler) as client:
        assert (await client.resolve_device(platform="ios")).id == "phone"
        assert (await client.resolve_device(platform="android")).id == "droid"


@pytest.mark.asyncio
async def test_unhealthy_candidates_are_skipped_for_a_healthy_one(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        # base_url carries a /v1 prefix, so match a suffix rather than the whole path.
        if request.url.path.endswith("/devices/bad/screenshot"):
            return httpx.Response(502, text="tap failed")
        if request.url.path.endswith("/screenshot"):
            return httpx.Response(200, content=png_bytes())
        return _fleet(device("bad"), device("good"))

    async with make_client(handler) as client:
        assert (await client.resolve_device()).id == "good"


@pytest.mark.asyncio
async def test_all_unhealthy_says_state_is_not_health(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/screenshot"):
            return httpx.Response(500, text="broken")
        return _fleet(device("a"), device("b"))

    async with make_client(handler) as client:
        with pytest.raises(DeviceUnavailable) as excinfo:
            await client.resolve_device()

    assert "not a health signal" in str(excinfo.value)


@pytest.mark.asyncio
async def test_require_healthy_false_skips_the_probe(make_client):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _fleet(device("a"))

    async with make_client(handler) as client:
        assert (await client.resolve_device(require_healthy=False)).id == "a"
    assert not any(path.endswith("/screenshot") for path in calls)
