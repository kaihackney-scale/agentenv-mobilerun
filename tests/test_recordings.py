"""Recording lifecycle: the terminal status, the 302, and the bare-list envelope."""

from __future__ import annotations

import httpx
import pytest

from agentenv_mobilerun.client import MobileRunError


@pytest.mark.asyncio
async def test_start_recording_returns_the_id(make_client):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        bodies.append(json.loads(request.content))
        return httpx.Response(201, json={"id": "rec-1", "status": "recording"})

    async with make_client(handler) as client:
        assert await client.start_recording("d", name="run-7") == "rec-1"
    assert bodies[0]["types"] == ["video", "trajectory"]
    assert bodies[0]["name"] == "run-7"


@pytest.mark.asyncio
async def test_start_recording_without_an_id_fails_loudly(make_client):
    async with make_client(lambda r: httpx.Response(201, json={"status": "recording"})) as client:
        with pytest.raises(MobileRunError, match="no recording id"):
            await client.start_recording("d")


@pytest.mark.asyncio
async def test_await_recording_waits_for_completed_not_ready(make_client):
    # The lifecycle is recording -> uploading -> completed. A poll that waits for "ready"
    # never returns on a recording that has already succeeded.
    statuses = iter(["recording", "uploading", "completed"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "rec-1", "status": next(statuses), "actions": 3})

    async with make_client(handler) as client:
        info = await client.await_recording("d", "rec-1", poll_seconds=0)
    assert info["status"] == "completed"
    assert info["actions"] == 3


@pytest.mark.asyncio
async def test_await_recording_raises_on_a_failed_recording(make_client):
    async with make_client(lambda r: httpx.Response(200, json={"status": "failed"})) as client:
        with pytest.raises(MobileRunError, match="status 'failed'"):
            await client.await_recording("d", "rec-1", poll_seconds=0)


@pytest.mark.asyncio
async def test_await_recording_times_out_naming_the_last_status(make_client):
    async with make_client(lambda r: httpx.Response(200, json={"status": "uploading"})) as client:
        with pytest.raises(MobileRunError, match="still 'uploading'"):
            await client.await_recording("d", "rec-1", timeout_seconds=0, poll_seconds=0)


@pytest.mark.asyncio
async def test_video_download_follows_the_302_to_the_presigned_url(make_client):
    # The route answers 302 to an R2 URL whose signature expires in 900s, so the client
    # must follow it and return bytes rather than hand back a link.
    def handler(request: httpx.Request) -> httpx.Response:
        if "r2.example" in str(request.url):
            return httpx.Response(200, content=b"mp4-bytes")
        return httpx.Response(
            302, headers={"Location": "https://r2.example/session.mp4?X-Amz-Expires=900"}
        )

    async with make_client(handler) as client:
        assert await client.download_recording_video("d", "rec-1") == b"mp4-bytes"


@pytest.mark.asyncio
async def test_list_recordings_accepts_the_bare_list(make_client):
    # Unlike most collections, this route returns a list rather than {"items": [...]}.
    async with make_client(lambda r: httpx.Response(200, json=[{"id": "rec-1"}])) as client:
        assert await client.list_recordings("d") == [{"id": "rec-1"}]


@pytest.mark.asyncio
async def test_expired_is_terminal_not_something_to_wait_through(make_client):
    # Seen live: a real recording sat at status "expired" with an `error` field. Treating
    # it as non-terminal means polling a dead recording until the timeout and then
    # reporting a timeout instead of the actual reason.
    async with make_client(lambda r: httpx.Response(200, json={"status": "expired", "error": "retention lapsed"})) as client:
        with pytest.raises(MobileRunError, match="status 'expired'"):
            await client.await_recording("d", "rec-1", poll_seconds=0)


@pytest.mark.asyncio
async def test_recording_info_fields_seen_live_survive_the_round_trip(make_client):
    # Field names copied from a live RecordingInfo, not the spec.
    body = {
        "id": "rec-1",
        "deviceId": "dev-1",
        "types": ["video", "trajectory"],
        "status": "completed",
        "display": {"width": 393, "height": 852, "rotation": 0},
        "actions": 3,
        "expiresAt": "2026-09-25T16:43:42.702418Z",
        "video": {"durationMs": 38306, "sizeBytes": 2031656, "format": "mp4"},
    }
    async with make_client(lambda r: httpx.Response(200, json=body)) as client:
        info = await client.await_recording("d", "rec-1", poll_seconds=0)
    assert info["actions"] == 3
    assert info["display"]["width"] == 393
    assert info["expiresAt"].startswith("2026-09-25")
