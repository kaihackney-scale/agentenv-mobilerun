"""The MCP tool layer: coordinate space, composed gestures, and capability gating."""

from __future__ import annotations

import io
import json

import httpx
import pytest

from agentenv_mobilerun.server.main import (
    CAPABILITY_GATED,
    CORE_TOOLS,
    MobileRunTools,
    _declared_capabilities,
    png_size,
    select_tools,
)
from tests.helpers import png_bytes


def test_png_size_reads_the_ihdr():
    assert png_size(png_bytes(1170, 2532)) == (1170, 2532)


def test_png_size_rejects_non_png_bytes():
    assert png_size(b"not an image") is None
    assert png_size(b"") is None
    # A correct signature with a truncated header is still unmeasurable.
    assert png_size(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4) is None


def test_select_tools_drops_a_tool_the_device_cannot_support():
    tools = select_tools({"clipboard": False, "accessibility": True})
    assert "get_clipboard" not in tools and "set_clipboard" not in tools
    assert "ui_state" in tools
    # The core surface is never gated.
    assert set(CORE_TOOLS).issubset(set(tools))


def test_select_tools_keeps_a_tool_when_the_capability_is_unknown():
    # An absent capability means "this plugin has not heard of it", not "unsupported".
    # Gating on absence would silently remove working tools as MobileRun adds names.
    tools = select_tools({})
    for name in CAPABILITY_GATED:
        assert name in tools


def test_declared_capabilities_reads_the_pair_list(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CAPABILITIES", "clipboard=false,recording=true")
    assert _declared_capabilities() == {"clipboard": False, "recording": True}


def test_the_writer_and_reader_agree(monkeypatch):
    # One round trip across the process boundary, so the two halves cannot drift.
    from agentenv_mobilerun.capabilities import encode_capabilities

    original = {"accessibility": True, "clipboard": False, "esim": False, "recording": True}
    monkeypatch.setenv("MOBILERUN_CAPABILITIES", encode_capabilities(original))
    assert _declared_capabilities() == original


def test_declared_capabilities_gates_the_tool_it_names(monkeypatch):
    # The regression this pins: an earlier encoding sent only the supported names, so the
    # server could not tell "unsupported" from "unknown", had to assume the latter, and
    # registered the clipboard tools on a phone that cannot do clipboards.
    monkeypatch.setenv("MOBILERUN_CAPABILITIES", "clipboard=false,accessibility=true")
    tools = select_tools(_declared_capabilities())
    assert "get_clipboard" not in tools and "set_clipboard" not in tools
    assert "ui_state" in tools


def test_declared_capabilities_falls_back_on_unusable_input(monkeypatch):
    # None means "discover it yourself", which is strictly better than gating on garbage.
    monkeypatch.setenv("MOBILERUN_CAPABILITIES", "clipboard,recording")
    assert _declared_capabilities() is None
    monkeypatch.setenv("MOBILERUN_CAPABILITIES", "   ")
    assert _declared_capabilities() is None
    monkeypatch.delenv("MOBILERUN_CAPABILITIES")
    assert _declared_capabilities() is None


def test_declared_capabilities_skips_only_the_malformed_entries(monkeypatch):
    # A single bad pair must not discard the good ones alongside it.
    monkeypatch.setenv("MOBILERUN_CAPABILITIES", "clipboard=false,garbage,recording=true,=true")
    assert _declared_capabilities() == {"clipboard": False, "recording": True}


def _geometry_handler(sent, *, pixels=(1179, 2556), points=(393, 852)):
    """Serve the two geometry probes, and record every gesture body.

    Defaults are a real iPhone 15 Pro: 1179x2556 physical pixels over a 393x852 logical
    point screen, i.e. exactly 3x.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/screenshot"):
            return httpx.Response(200, content=png_bytes(*pixels))
        if path.endswith("/ui-state"):
            return httpx.Response(
                200,
                json={
                    "a11y_tree": {},
                    "phone_state": {},
                    "device_context": {"screen_bounds": {"width": points[0], "height": points[1]}},
                },
            )
        sent.append((path, json.loads(request.content)))
        return httpx.Response(200, json={})

    return handler


def _tools(handler) -> MobileRunTools:
    from agentenv_mobilerun.client import MobileRunClient

    client = MobileRunClient(
        "dr_sk_test", transport=httpx.MockTransport(handler), max_retries=0
    )
    return MobileRunTools(client, "dev-1")


@pytest.mark.asyncio
async def test_screenshot_is_returned_unscaled():
    raw = png_bytes(1170, 2532)

    tools = _tools(lambda r: httpx.Response(200, content=raw))
    image = await tools.screenshot()
    # The whole coordinate contract rests on this: the agent sees the device's own
    # pixels, so a coordinate read off the image is the coordinate the device expects.
    # Any resize here would silently offset every tap.
    assert image.data == raw

    size = json.loads(await tools.screen_size())
    assert (size["width"], size["height"]) == (1170, 2532)


@pytest.mark.asyncio
async def test_tap_converts_screenshot_pixels_to_device_points():
    sent = []
    result = json.loads(await _tools(_geometry_handler(sent)).tap(589, 1278))

    path, body = sent[0]
    assert path.endswith("/tap")
    # THE central behaviour of this package. The caller passes the pixel centre of a
    # 1179x2556 screenshot; the device plane takes logical points, so 589,1278 must arrive
    # as 196,426 — the centre of a 393x852 screen. Sending the pixel values straight
    # through would aim at a point three times too far right and down, which on most of
    # the screen is out of bounds entirely.
    assert (body["x"], body["y"]) == (196, 426)
    # The reply echoes the caller's space, and reports what was actually sent.
    assert (result["x"], result["y"]) == (589, 1278)
    assert result["device_point"] == [196, 426]


@pytest.mark.asyncio
async def test_long_press_is_a_held_zero_distance_swipe_in_point_space():
    sent = []
    await _tools(_geometry_handler(sent)).long_press(300, 600, duration_ms=900)

    path, body = sent[0]
    assert path.endswith("/swipe")
    # Zero distance: a held touch, not a drag.
    assert (body["startX"], body["startY"]) == (body["endX"], body["endY"]) == (100, 200)
    assert body["duration"] == 900


@pytest.mark.asyncio
async def test_double_tap_sends_two_taps_at_the_same_converted_point():
    sent = []
    await _tools(_geometry_handler(sent)).double_tap(30, 60)

    assert [p.endswith("/tap") for p, _ in sent] == [True, True]
    assert [(b["x"], b["y"]) for _, b in sent] == [(10, 20), (10, 20)]


@pytest.mark.asyncio
async def test_scroll_down_drags_the_finger_upward():
    sent = []
    await _tools(_geometry_handler(sent, pixels=(400, 800), points=(400, 800))).scroll("down", amount=0.5)

    body = sent[0][1]
    # "down" means reveal what is below the fold, which requires dragging up. Getting this
    # inversion wrong scrolls the wrong way on every single call.
    assert body["startY"] > body["endY"]
    assert body["startX"] == body["endX"] == 200


@pytest.mark.asyncio
async def test_scroll_right_drags_leftward():
    sent = []
    await _tools(_geometry_handler(sent, pixels=(400, 800), points=(400, 800))).scroll("right")
    assert sent[0][1]["startX"] > sent[0][1]["endX"]


@pytest.mark.asyncio
async def test_scroll_geometry_is_also_converted():
    sent = []
    # 3x display: a scroll computed in pixels must still be sent in points, so every
    # coordinate stays under the 393x852 bound.
    await _tools(_geometry_handler(sent)).scroll("down", amount=0.8)
    body = sent[0][1]
    assert max(body["startX"], body["endX"]) <= 393
    assert max(body["startY"], body["endY"]) <= 852


@pytest.mark.asyncio
async def test_the_scale_probe_happens_once_and_is_then_cached():
    sent = []
    probes = {"screenshot": 0, "ui-state": 0}

    inner = _geometry_handler(sent)

    def counting(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/screenshot"):
            probes["screenshot"] += 1
        if request.url.path.endswith("/ui-state"):
            probes["ui-state"] += 1
        return inner(request)

    tools = _tools(counting)
    await tools.tap(10, 10)
    await tools.tap(20, 20)
    await tools.swipe(1, 2, 3, 4)
    # /ui-state is ~183 KB on a real device, so reading it per gesture would be a serious
    # cost. The ratio is display density and does not change, so once is enough.
    assert probes["ui-state"] == 1
    assert probes["screenshot"] == 1


@pytest.mark.asyncio
async def test_an_unmeasurable_screen_passes_coordinates_through_unchanged():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/screenshot"):
            return httpx.Response(200, content=b"not an image")
        if request.url.path.endswith("/ui-state"):
            return httpx.Response(500, text="unavailable")
        sent.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={})

    await _tools(handler).tap(196, 426)
    # Fail open rather than refuse to act: a pass-through is right for a caller already
    # working in points, and a run should not die because /ui-state blipped. The scale is
    # logged loudly instead.
    assert (sent[0][1]["x"], sent[0][1]["y"]) == (196, 426)


@pytest.mark.asyncio
async def test_screen_size_reports_the_caller_space_and_the_conversion():
    result = json.loads(await _tools(_geometry_handler([])).screen_size())
    assert (result["width"], result["height"]) == (1179, 2556)
    assert result["device_points"] == [393, 852]
    assert result["pixels_per_point"] == 3.0
    assert "screenshot pixels" in result["space"]


@pytest.mark.asyncio
async def test_scroll_rejects_a_bad_direction_without_touching_the_device():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={})

    result = json.loads(await _tools(handler).scroll("sideways"))
    assert result["ok"] is False
    assert calls == []


@pytest.mark.asyncio
async def test_press_key_maps_names_to_codes():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={})

    tools = _tools(handler)
    assert json.loads(await tools.press_key("home"))["code"] == 2
    assert bodies[0]["action"] == 2


@pytest.mark.asyncio
async def test_press_key_lists_what_it_supports_when_given_junk():
    result = json.loads(await _tools(lambda r: httpx.Response(200, json={})).press_key("cmd+q"))
    assert result["ok"] is False
    assert "home" in result["supported"]


@pytest.mark.asyncio
async def test_wait_is_clamped_and_performs_no_action(monkeypatch):
    calls = []
    slept = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={})

    # Patched so the assertion about the clamp does not cost the suite 15 real seconds.
    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr("agentenv_mobilerun.server.main.asyncio.sleep", fake_sleep)

    result = json.loads(await _tools(handler).wait(999))
    assert result["waited_seconds"] == 15
    assert slept == [15]
    # An unbounded wait would let an agent stall a run; and waiting must never act.
    assert calls == []


@pytest.mark.asyncio
async def test_wait_never_sleeps_a_negative_duration(monkeypatch):
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr("agentenv_mobilerun.server.main.asyncio.sleep", fake_sleep)
    assert json.loads(await _tools(lambda r: httpx.Response(200, json={})).wait(-5))["waited_seconds"] == 0
    assert slept == [0]


def test_jpeg_reencoding_preserves_the_dimensions_exactly():
    """The dimensions ARE the coordinate space, so this must never resize.

    The size saving is measured on real screen content (a device screenshot is ~4 MB of
    PNG, ~5.3 MB once base64'd into an MCP response, and ~280 KB as JPEG at the same
    dimensions). It is deliberately NOT asserted here: a synthetic flat-colour image
    compresses better as PNG than as JPEG, so a size assertion on a fixture would either
    fail or force a contrived input. Dimension preservation is the property that actually
    has to hold, and it is the one worth pinning.
    """
    from PIL import Image as PILImage

    from agentenv_mobilerun.server.main import to_jpeg

    original = PILImage.new("RGBA", (1179, 2556), (10, 20, 30, 255))
    buffer = io.BytesIO()
    original.save(buffer, format="PNG")

    data, image_format = to_jpeg(buffer.getvalue())
    assert image_format == "jpeg"
    assert PILImage.open(io.BytesIO(data)).size == (1179, 2556)


def test_jpeg_reencoding_drops_the_alpha_channel_ios_screenshots_carry():
    # JPEG has no alpha; an unconverted RGBA image raises inside Pillow and would fall
    # back to the 14x-larger PNG on every single screenshot.
    from PIL import Image as PILImage

    from agentenv_mobilerun.server.main import to_jpeg

    buffer = io.BytesIO()
    PILImage.new("RGBA", (40, 80), (1, 2, 3, 128)).save(buffer, format="PNG")
    data, image_format = to_jpeg(buffer.getvalue())
    assert image_format == "jpeg"
    assert PILImage.open(io.BytesIO(data)).mode == "RGB"


def test_an_undecodable_image_falls_back_to_the_original_bytes():
    from agentenv_mobilerun.server.main import to_jpeg

    # Better a large correct screenshot than none: the agent cannot act without one.
    data, image_format = to_jpeg(b"not an image")
    assert (data, image_format) == (b"not an image", "png")


@pytest.mark.asyncio
async def test_the_screenshot_tool_returns_jpeg_at_the_device_dimensions():
    from PIL import Image as PILImage

    source = PILImage.new("RGBA", (1179, 2556))
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")

    image = await _tools(lambda r: httpx.Response(200, content=buffer.getvalue())).screenshot()
    assert image._mime_type == "image/jpeg"
    assert PILImage.open(io.BytesIO(image.data)).size == (1179, 2556)


@pytest.mark.asyncio
async def test_a_transient_ui_state_failure_is_not_cached_as_a_scale_of_one():
    """The worst bug this package could ship, and it did for one commit.

    Caching the 1.0 fallback lets a single blip on `/ui-state` silently misplace every
    gesture for the life of the process — on a 3x display that is most of the screen,
    with one warning logged at startup and nothing after.
    """
    sent = []
    calls = {"ui": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/screenshot"):
            return httpx.Response(200, content=png_bytes(1179, 2556))
        if path.endswith("/ui-state"):
            calls["ui"] += 1
            if calls["ui"] == 1:
                return httpx.Response(500, text="transient")
            return httpx.Response(200, json={
                "a11y_tree": {}, "phone_state": {},
                "device_context": {"screen_bounds": {"width": 393, "height": 852}},
            })
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={})

    tools = _tools(handler)

    await tools.tap(589, 1278)
    # First call degrades to a pass-through rather than refusing to act.
    assert (sent[0]["x"], sent[0]["y"]) == (589, 1278)

    await tools.tap(589, 1278)
    # Second call re-probes and gets it right, instead of being stuck on 1.0 forever.
    assert (sent[1]["x"], sent[1]["y"]) == (196, 426)


@pytest.mark.asyncio
async def test_scroll_uses_the_current_screen_size_after_a_rotation():
    """Rotation changes the dimensions but not the ratio.

    Caching the dimensions alongside the ratio makes a post-rotation scroll drag along
    the old axis — a vertical swipe down the middle of a screen that is now landscape.
    """
    sent = []
    orientation = {"landscape": False}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/screenshot"):
            size = (2556, 1179) if orientation["landscape"] else (1179, 2556)
            return httpx.Response(200, content=png_bytes(*size))
        if path.endswith("/ui-state"):
            bounds = {"width": 852, "height": 393} if orientation["landscape"] else {"width": 393, "height": 852}
            return httpx.Response(200, json={
                "a11y_tree": {}, "phone_state": {}, "device_context": {"screen_bounds": bounds},
            })
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={})

    tools = _tools(handler)
    await tools.scroll("down", amount=0.5)
    portrait_x = sent[0]["startX"]

    orientation["landscape"] = True
    await tools.scroll("down", amount=0.5)
    landscape_x = sent[1]["startX"]

    # Centre of a 393-point screen vs a 852-point one. Equal values would mean the
    # rotation was not noticed.
    assert portrait_x == 196
    assert landscape_x == 426


@pytest.mark.asyncio
async def test_a_transient_ui_state_failure_is_retried_instead_of_misplacing_the_gesture():
    """The upside of delegating transport to the vendor SDK.

    Before, nothing retried: one 500 on /ui-state dropped to the 1.0 scale fallback and
    sent the tap to a third of its intended position on a 3x display. The SDK retries, so
    the scale resolves correctly and the fallback is never reached. Constructed WITHOUT
    max_retries=0 on purpose -- the rest of the suite pins that off for determinism, so
    this is the only place the retry itself is asserted.
    """
    sent = []
    calls = {"ui": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/screenshot"):
            return httpx.Response(200, content=png_bytes(1179, 2556))
        if path.endswith("/ui-state"):
            calls["ui"] += 1
            if calls["ui"] == 1:
                return httpx.Response(500, text="transient")
            return httpx.Response(200, json={
                "a11y_tree": {}, "phone_state": {},
                "device_context": {"screen_bounds": {"width": 393, "height": 852}},
            })
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={})

    from agentenv_mobilerun.client import MobileRunClient

    client = MobileRunClient("dr_sk_test", transport=httpx.MockTransport(handler))
    tools = MobileRunTools(client, "dev-1")

    await tools.tap(589, 1278)

    assert calls["ui"] == 2, "the 500 should have been retried, not surfaced"
    assert (sent[0]["x"], sent[0]["y"]) == (196, 426), "scale resolved, so the tap is placed correctly"
