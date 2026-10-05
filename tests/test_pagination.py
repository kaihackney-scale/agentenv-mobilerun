"""Device listing: pagination and the API's default state filter.

Both behaviours were found against the live API, and both fail *silently*: an unpaged read
returns a truncated fleet and a default read hides every device that is not ready. The
second one actually misled this package's first validation run into reporting an
all-iPhone account when an Android phone was sitting in ``maintenance``.
"""

from __future__ import annotations

import httpx
import pytest


def page(items, *, has_next, page_no=1, page_size=100):
    return httpx.Response(
        200,
        json={
            "items": items,
            "pagination": {
                "hasNext": has_next,
                "hasPrev": page_no > 1,
                "page": page_no,
                "pageSize": page_size,
                "total": len(items),
                "pages": 2 if has_next else page_no,
            },
        },
    )


def dev(id_, *, platform="ios", state="ready"):
    return {"id": id_, "platform": platform, "state": state, "type": "device_slot"}


@pytest.mark.asyncio
async def test_every_page_is_followed(make_client):
    seen_pages = []

    def handler(request: httpx.Request) -> httpx.Response:
        n = int(request.url.params.get("page", 1))
        seen_pages.append(n)
        if n == 1:
            return page([dev("a"), dev("b")], has_next=True, page_no=1)
        return page([dev("c")], has_next=False, page_no=2)

    async with make_client(handler) as client:
        devices = await client.list_devices()

    # A fleet larger than one page must not come back truncated.
    assert [d.id for d in devices] == ["a", "b", "c"]
    assert seen_pages == [1, 2]


@pytest.mark.asyncio
async def test_a_single_page_does_not_fetch_a_second(make_client):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params.get("page"))
        return page([dev("a")], has_next=False)

    async with make_client(handler) as client:
        await client.list_devices()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_server_that_always_reports_another_page_cannot_spin_forever(make_client):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return page([dev(f"d{len(calls)}")], has_next=True)

    async with make_client(handler) as client:
        devices = await client.list_devices(max_pages=4)
    assert len(calls) == 4 and len(devices) == 4


@pytest.mark.asyncio
async def test_a_missing_pagination_object_stops_after_one_page(make_client):
    # The bare-list routes have no pagination block at all; treat its absence as "done"
    # rather than crashing or looping.
    async with make_client(lambda r: httpx.Response(200, json={"items": [dev("a")]})) as client:
        assert len(await client.list_devices()) == 1


@pytest.mark.asyncio
async def test_ready_is_requested_explicitly_by_default(make_client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["state"] = request.url.params.get_list("state")
        return page([dev("a")], has_next=False)

    async with make_client(handler) as client:
        await client.list_devices()
    # Sent explicitly rather than relying on the server default, so the filter in force is
    # visible in the request instead of being an invisible server-side assumption.
    assert seen["state"] == ["ready"]


@pytest.mark.asyncio
async def test_multiple_states_go_in_one_comma_joined_parameter(make_client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["state"] = request.url.params.get_list("state")
        return page([], has_next=False)

    async with make_client(handler) as client:
        await client.list_devices(states=("ready", "maintenance"))
    # NOT ["ready", "maintenance"] — as repeated keys this server silently returns the
    # wrong rows (measured: the maintenance device vanished, and ten repeated keys
    # returned nothing at all).
    assert seen["state"] == ["ready,maintenance"]


@pytest.mark.asyncio
async def test_states_none_names_every_state_rather_than_omitting_the_parameter(make_client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["state"] = request.url.params.get_list("state")
        return page([dev("a", state="maintenance")], has_next=False)

    async with make_client(handler) as client:
        devices = await client.list_devices(states=None)

    # Two subtleties, each of which defeated an earlier attempt at this fix:
    # omitting `state` does NOT disable the filter (it selects the server's ["ready"]
    # default), and the values must be comma-joined in ONE parameter — sent as repeated
    # keys the server answers 200 with the wrong rows.
    from agentenv_mobilerun.client import ALL_DEVICE_STATES

    assert seen["state"] == [",".join(ALL_DEVICE_STATES)]
    assert devices[0].state == "maintenance"


@pytest.mark.asyncio
async def test_the_state_parameter_is_never_absent(make_client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["state"] = request.url.params.get_list("state")
        return page([], has_next=False)

    async with make_client(handler) as client:
        await client.list_devices()
    assert seen["state"], "an absent state parameter silently means ready-only"


@pytest.mark.asyncio
async def test_device_type_filter_is_forwarded(make_client):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["type"] = request.url.params.get("type")
        return page([], has_next=False)

    async with make_client(handler) as client:
        await client.list_devices(device_type="android_cloud_phone")
    assert seen["type"] == "android_cloud_phone"


@pytest.mark.asyncio
async def test_platform_filter_still_applies_across_pages(make_client):
    def handler(request: httpx.Request) -> httpx.Response:
        n = int(request.url.params.get("page", 1))
        if n == 1:
            return page([dev("i1", platform="ios")], has_next=True, page_no=1)
        return page([dev("a1", platform="android")], has_next=False, page_no=2)

    async with make_client(handler) as client:
        found = await client.list_devices(platform="android")
    # The filter must be applied after the pages are joined, not to the first page only.
    assert [d.id for d in found] == ["a1"]


@pytest.mark.asyncio
async def test_fleet_summary_reports_what_the_listing_hides(make_client):
    body = {
        "total": 13,
        "byState": {"ready": 10, "maintenance": 1, "terminated": 2},
        "byType": {"android_cloud_phone": 1, "device_slot": 12},
    }
    async with make_client(lambda r: httpx.Response(200, json=body)) as client:
        assert (await client.fleet_summary())["byState"]["maintenance"] == 1


@pytest.mark.asyncio
async def test_fleet_summary_tolerates_an_unexpected_body(make_client):
    async with make_client(lambda r: httpx.Response(200, json=[])) as client:
        assert await client.fleet_summary() == {}
