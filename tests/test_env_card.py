"""The environment card and the core data plane this server publishes.

An env deployed WITHOUT a gateway has to serve its own card -- the gateway used to do it
on a server's behalf, and `EnvironmentServerProvider` refuses the deploy unless the card
appears, under the env's own `environment_name`, in time.

Every assertion here goes through HTTP against the built app, never against a function
that returns a card. The first version of this card was hand-written and advertised
JSON-RPC at a path the server did not route; a test that read the same literal agreed
with it. Only a request can tell you what is served.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agentenv_mobilerun.server.main import (
    MCP_PATH,
    MCP_TRANSPORT,
    PROTOCOL_VERSION,
    RPC_PATH,
    WELL_KNOWN_PATH,
    build_server,
)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("MOBILERUN_API_KEY", "dr_sk_test")
    monkeypatch.setenv("MOBILERUN_DEVICE_ID", "dev-1")
    # Declared, so building the server makes no network call.
    monkeypatch.setenv("MOBILERUN_CAPABILITIES", "accessibility=true,recording=true,clipboard=false")
    monkeypatch.setenv("ENVIRONMENT_NAME", "mobilerun-phone")
    return build_server()


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def run() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://srv") as c:
            return await c.request(method, path, **kwargs)

    return asyncio.run(run())


def _card(server) -> dict:
    response = _request(server.streamable_http_app(), "GET", WELL_KNOWN_PATH)
    assert response.status_code == 200
    return response.json()


def test_the_card_is_served_at_the_well_known_path(server):
    assert _card(server)["name"] == "mobilerun-phone"


def test_the_card_is_named_for_the_environment_not_the_server(server):
    """The provider compares `card["name"]` to the env's `environment_name` and fails the
    deploy on a mismatch. The FastMCP app is called "MobileRun"; the card must not be."""
    assert _card(server)["name"] == "mobilerun-phone"


def test_the_card_declares_the_mcp_interface_explicitly(server):
    """`mcp_path()` falls back to /mcp by convention, but a reader should not have to infer
    the transport from a missing field."""
    assert {"url": MCP_PATH, "transport": MCP_TRANSPORT} in _card(server)["additionalInterfaces"]


def test_operations_is_empty_rather_than_absent(server):
    """The load-bearing distinction. An ABSENT `operations` means "pre-advertisement
    card", which a reader takes as the full add/get/reset data trio. This server
    implements none of them, so absence would be a false claim about its wire surface."""
    capabilities = _card(server)["capabilities"]
    assert capabilities.get("operations") == [], "absent/None operations claims the full data trio"


def test_the_card_does_not_freeze_a_tool_list(server):
    """The tool surface is gated per device, so a static list in the card would drift from
    what the process actually serves. The provider does a live tools/list, which cannot."""
    assert not _card(server)["capabilities"].get("tools")


def test_the_card_matches_the_protocol_constants(server):
    card = _card(server)
    assert card["protocolVersion"] == PROTOCOL_VERSION
    assert card["url"] == RPC_PATH


def test_the_card_is_json_serialisable_as_sent(server):
    """It goes out over HTTP, so anything unserialisable is a 500 at deploy time."""
    assert json.loads(json.dumps(_card(server)))["name"] == "mobilerun-phone"


def test_every_address_the_card_advertises_is_routed(server):
    """What the first card got wrong: it named /agentenv as its primary address with
    nothing serving there. Anything the card names has to answer."""
    paths = {route.path for route in server.streamable_http_app().routes if hasattr(route, "path")}
    card = _card(server)
    advertised = {card["url"]} | {i["url"] for i in card["additionalInterfaces"]}
    assert advertised <= paths, f"card advertises {advertised - paths}, which nothing routes"


def test_the_data_plane_answers_method_not_found(server):
    """This env registers no data operations, so the dispatcher's whole job is to say so
    -- at the RPC level, over a successful HTTP exchange, not as a transport failure."""
    response = _request(
        server.streamable_http_app(),
        "POST",
        RPC_PATH,
        json={"jsonrpc": "2.0", "id": 7, "method": "data/reset", "params": {}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 7
    assert body["error"]["code"] == -32601


def test_the_data_plane_rejects_a_malformed_request(server):
    response = _request(server.streamable_http_app(), "POST", RPC_PATH, content=b"{not json")
    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32700


def test_the_server_serves_at_least_one_tool_with_no_duplicates(server):
    """The provider's other gate: an env without a gateway must list >=1 tool and no name
    twice, or the deploy is refused."""
    names = [t.name for t in asyncio.run(server.list_tools())]
    assert names
    assert len(names) == len(set(names))
    # And the capability gate still applies: clipboard was declared false.
    assert "mobilerun_set_clipboard" not in names


def test_a_standalone_run_still_serves_a_card(monkeypatch):
    """Without ENVIRONMENT_NAME -- agent-env injects it, a local `python -m ...` does not --
    the server must still come up rather than refuse or serve a nameless card."""
    monkeypatch.setenv("MOBILERUN_API_KEY", "dr_sk_test")
    monkeypatch.setenv("MOBILERUN_DEVICE_ID", "dev-1")
    monkeypatch.setenv("MOBILERUN_CAPABILITIES", "accessibility=true")
    monkeypatch.delenv("ENVIRONMENT_NAME", raising=False)
    served = _request(build_server().streamable_http_app(), "GET", WELL_KNOWN_PATH)
    assert served.json()["name"] == "mobilerun"
