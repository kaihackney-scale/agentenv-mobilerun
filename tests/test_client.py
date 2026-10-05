from __future__ import annotations

import json

import httpx
import pytest

from agentenv_mobilerun.client import (
    Device,
    DeviceUnavailable,
    MobileRunClient,
    MobileRunError,
    _items,
)
from tests.helpers import png_bytes


def test_items_accepts_both_collection_shapes():
    # GET /devices answers {"items": [...]}; GET /recordings answers a bare list. Code
    # written against only the first shape dies on the second.
    assert _items({"items": [{"id": "a"}]}) == [{"id": "a"}]
    assert _items([{"id": "b"}]) == [{"id": "b"}]
    assert _items({"no_items_key": 1}) == []
    assert _items(None) == []


def test_device_accepts_either_active_task_spelling():
    assert Device.from_api({"id": "d", "activeTask": "t1"}).active_task_id == "t1"
    assert Device.from_api({"id": "d", "active_task_id": "t2"}).active_task_id == "t2"
    assert Device.from_api({"id": "d"}).idle is True


@pytest.mark.asyncio
async def test_api_key_is_sent_as_bearer(make_client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"items": []})

    async with make_client(handler) as client:
        await client.list_devices()
    assert seen["auth"] == "Bearer dr_sk_test"


def test_empty_api_key_is_rejected_at_construction():
    with pytest.raises(MobileRunError, match="API key is required"):
        MobileRunClient("")


@pytest.mark.asyncio
async def test_get_device_strips_the_stream_credential(make_client):
    # GET /devices/{id} returns a live device-scoped JWT. It must not survive into a
    # Device.raw that a CLI or a log line might print.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "dev-1",
                "platform": "ios",
                "state": "ready",
                "streamToken": "ey.SECRET.jwt",
                "streamUrl": "wss://api.mobilerun.ai/v1/devices/dev-1/stream",
            },
        )

    async with make_client(handler) as client:
        device = await client.get_device("dev-1")

    assert device.id == "dev-1"
    assert "streamToken" not in device.raw
    assert "streamUrl" not in device.raw
    assert "SECRET" not in json.dumps(device.raw)


@pytest.mark.asyncio
async def test_transport_failure_has_no_status_code(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("no route")

    async with make_client(handler) as client:
        with pytest.raises(MobileRunError) as excinfo:
            await client.list_devices()
    # None means "we never got an answer", which callers must not read as a rejection.
    assert excinfo.value.status_code is None


@pytest.mark.asyncio
async def test_http_error_carries_status_and_detail(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "key revoked"})

    async with make_client(handler) as client:
        with pytest.raises(MobileRunError) as excinfo:
            await client.list_devices()
    assert excinfo.value.status_code == 403
    assert "key revoked" in str(excinfo.value)


@pytest.mark.asyncio
async def test_swipe_clamps_duration_below_the_api_floor(make_client):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={})

    async with make_client(handler) as client:
        await client.swipe("d", 1, 2, 3, 4, duration_ms=0)
    # The API rejects duration < 10, so forwarding 0 would be a guaranteed 400.
    assert bodies[0]["duration"] == 10


@pytest.mark.asyncio
async def test_input_text_defaults_to_committed(make_client):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={})

    async with make_client(handler) as client:
        await client.input_text("d", "hello")
    # "committed" waits for the text to actually land in the focused field; "accepted"
    # only proves the keystrokes were taken.
    assert bodies[0]["completionMode"] == "committed"


@pytest.mark.asyncio
async def test_input_text_rejects_an_unknown_completion_mode(make_client):
    async with make_client(lambda r: httpx.Response(200, json={})) as client:
        with pytest.raises(MobileRunError, match="completion_mode"):
            await client.input_text("d", "x", completion_mode="eventually")


@pytest.mark.asyncio
async def test_health_probe_is_false_when_the_screenshot_fails(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="portal rpc call 'screenshot' failed")

    async with make_client(handler) as client:
        assert await client.health_probe("d") is False


@pytest.mark.asyncio
async def test_capabilities_handles_both_envelopes(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        if "wrapped" in str(request.url):
            return httpx.Response(200, json={"capabilities": {"stream": True, "esim": False}})
        return httpx.Response(200, json={"stream": True, "esim": False, "deviceType": "device_slot"})

    async with make_client(handler) as client:
        assert await client.capabilities("wrapped") == {"stream": True, "esim": False}
        # Non-bool values (deviceType) are dropped rather than coerced to True.
        assert await client.capabilities("flat") == {"stream": True, "esim": False}


@pytest.mark.asyncio
async def test_start_app_always_sends_a_body(make_client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["body"] = request.content
        return httpx.Response(200, json={})

    async with make_client(handler) as client:
        await client.start_app("d", "com.apple.mobilesafari")

    assert seen["method"] == "PUT"
    # The API answers `400 request body is required` to a bodyless PUT, so an empty JSON
    # object is not optional. Verified against a real device.
    assert seen["body"] == b"{}"


@pytest.mark.asyncio
async def test_start_app_forwards_an_activity(make_client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={})

    async with make_client(handler) as client:
        await client.start_app("d", "pkg", activity=".MainActivity")
    assert seen["body"] == {"activity": ".MainActivity"}


@pytest.mark.asyncio
async def test_stop_app_patches_and_never_deletes(make_client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={})

    async with make_client(handler) as client:
        await client.stop_app("d", "pkg")

    # DELETE on this route is `delete-app` — it UNINSTALLS. A stop that deletes the app is
    # the kind of mistake you only make once, on somebody's real phone.
    assert seen["method"] == "PATCH"
    assert seen["body"] == {"clearData": False}


@pytest.mark.asyncio
async def test_stop_app_does_not_clear_data_by_default(make_client):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={})

    async with make_client(handler) as client:
        await client.stop_app("d", "pkg")
        await client.stop_app("d", "pkg", clear_data=True)
    assert [b["clearData"] for b in bodies] == [False, True]


@pytest.mark.asyncio
async def test_the_destructive_route_is_named_for_what_it_does(make_client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        return httpx.Response(200, json={})

    async with make_client(handler) as client:
        await client.uninstall_app("d", "pkg")
    assert seen["method"] == "DELETE"


# ------------------------------------------------------------------- retries

# The SDK retries 408/409/429/5xx and every transport failure, two attempts deep by
# default, without distinguishing read from write and without an idempotency key on the
# wire. On this device plane a replayed write is a second real gesture, so the client
# keeps the retries for reads and takes them away from writes (see MobileRunClient
# __init__). These two tests are the only thing holding that apart.


@pytest.fixture
def no_backoff(monkeypatch):
    """Collapse the SDK's exponential backoff so a retry test costs no wall clock."""
    import anyio

    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(anyio, "sleep", instant)


def _counting_handler(counter: list) -> "object":
    def handler(request: httpx.Request) -> httpx.Response:
        counter.append(f"{request.method} {request.url.path}")
        return httpx.Response(500, json={"detail": "boom"})

    return handler


@pytest.mark.asyncio
async def test_a_write_is_not_retried(no_backoff):
    """A tap whose response is lost must not become two taps on whatever is now under
    the finger. Nothing on the device plane deduplicates it."""
    attempts: list = []
    client = MobileRunClient("dr_sk_test", transport=httpx.MockTransport(_counting_handler(attempts)))
    async with client:
        with pytest.raises(MobileRunError):
            await client.tap("dev-1", 10, 20)
    assert len(attempts) == 1, f"the write was sent {len(attempts)} times: {attempts}"


@pytest.mark.asyncio
async def test_a_read_is_retried(no_backoff):
    """The other half of the same decision: replaying a screenshot costs nothing, and a
    transient 500 on a read is what the SDK's retries are for."""
    attempts: list = []
    client = MobileRunClient("dr_sk_test", transport=httpx.MockTransport(_counting_handler(attempts)))
    async with client:
        with pytest.raises(MobileRunError):
            await client.screenshot("dev-1")
    assert len(attempts) == 3, f"expected 1 attempt + 2 retries, got {attempts}"


@pytest.mark.asyncio
async def test_both_clients_share_the_injected_transport(no_backoff):
    """The no-retry client is a view of the same connection pool, not a second one: an
    injected transport (which is how every test here drives it) has to reach both."""
    attempts: list = []
    client = MobileRunClient("dr_sk_test", transport=httpx.MockTransport(_counting_handler(attempts)))
    async with client:
        with pytest.raises(MobileRunError):
            await client.press_home("dev-1")
    assert attempts, "the write path bypassed the injected transport"


@pytest.mark.asyncio
async def test_wait_ready_strips_the_stream_credential(make_client):
    """Measured on a live device: GET /devices/{id}/wait answers the full device record,
    live streamToken included, and the MCP wait_ready tool hands it to the agent."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/devices/dev-1/wait")
        return httpx.Response(200, json={
            "id": "dev-1", "state": "ready", "platform": "ios",
            "streamUrl": "wss://api.mobilerun.ai/v1/devices/dev-1/stream", "streamToken": "eyJ.live.jwt",
        })

    async with make_client(handler) as client:
        ready = await client.wait_ready("dev-1")
    assert ready["state"] == "ready"
    assert "streamToken" not in ready and "streamUrl" not in ready
    assert "eyJ.live.jwt" not in json.dumps(ready)
