"""MCP server exposing one MobileRun cloud phone as agent tools.

Runs as a single container inside the agent-env gateway and talks straight to
``api.mobilerun.ai`` over HTTPS. There is deliberately no second tier: the
MCP -> computer-server -> driver split that a USB-attached phone needs exists only
because the phone is bolted to a host machine. A cloud phone needs outbound HTTPS,
so the device client lives in this process.

Configuration (all injected by :class:`~agentenv_mobilerun.env.MobileRunEnv`):

===========================  ====================================================
``MOBILERUN_API_KEY``        required; the account key (``dr_sk_...``)
``MOBILERUN_DEVICE_ID``      required; the pinned device this server drives
``MOBILERUN_BASE_URL``       optional; defaults to the public API
``MOBILERUN_CAPABILITIES``   optional; the device's capability map as
                             ``name=true,name2=false``. It must carry the ``false``
                             entries: a list of only the supported names cannot
                             distinguish "unsupported" from "a capability this plugin has
                             not heard of", which would make the gating inert. Not JSON —
                             this value is rendered into docker-compose YAML, where
                             ``": "`` is illegal. When unset, the server reads the
                             capability map itself at startup.
``MCP_HOST`` / ``MCP_PORT``  bind address; defaults ``0.0.0.0:18765``
===========================  ====================================================

**Coordinate contract.** Callers work in **screenshot pixels** — read a coordinate off
the image and pass it in unchanged. This layer converts to the device's logical-point
space before sending, in exactly one place.

The conversion is not optional. Measured on a live iPhone: ``screenshot`` returns
1179x2556 physical pixels while the device plane's ``/tap`` takes logical points on a
393x852 screen — a factor of 3. Proof that ``/tap`` is in points rather than pixels
comes from a recorded trajectory of taps sent through the same endpoint: they land at
``x=196``, the exact centre of a 393-point screen (the pixel centre would be 589).
Passing pixel coordinates straight through therefore misses by 3x and puts most of the
screen out of bounds.

Doing it here rather than asking the caller keeps the rule a model has to follow down to
one sentence, and keeps the scale factor in a single cached place where it cannot be
applied twice or forgotten.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import struct
from typing import Any, Optional

from agentenv_protocol import (
    MCP_PATH,
    MCP_TRANSPORT,
    PROTOCOL_VERSION,
    RPC_PATH,
    WELL_KNOWN_PATH,
    AgentEnvFastMCPApplication,
    EnvironmentCard,
)
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

from agentenv_mobilerun.client import GLOBAL_ACTIONS, MobileRunClient, MobileRunError

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("agentenv_mobilerun.server")


def _require(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise SystemExit(
            f"{name} is not set. This server is normally deployed by MobileRunEnv, which injects it; "
            "to run it by hand, export MOBILERUN_API_KEY and MOBILERUN_DEVICE_ID."
        )
    return value


def png_size(data: bytes) -> Optional[tuple[int, int]]:
    """``(width, height)`` from a PNG's IHDR, or ``None`` if it isn't a PNG.

    Parsed by hand so the image does not need Pillow. The only reason the server cares
    about dimensions is to tell the agent the coordinate space; it never resamples.
    """
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


#: JPEG quality for screenshots. The device returns ~4 MB of PNG, which becomes ~5.3 MB of
#: base64 in an MCP response — measured on a real iPhone. Re-encoding to JPEG at the SAME
#: pixel dimensions is ~14x smaller (374 KB) and leaves the coordinate contract untouched,
#: because nothing is resized. Over a run with dozens of screenshots that is the difference
#: between hundreds of megabytes and a handful.
SCREENSHOT_JPEG_QUALITY = 85


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields})


def to_jpeg(png: bytes, quality: int = SCREENSHOT_JPEG_QUALITY) -> tuple[bytes, str]:
    """``(bytes, format)`` — the screenshot re-encoded as JPEG at unchanged dimensions.

    Returns the original PNG untouched if it cannot be decoded, so an unexpected image
    format degrades to "big but correct" rather than to no screenshot at all.

    ⚠️ Never resize here. The dimensions are the coordinate space the caller reads taps
    off; changing them silently would reintroduce exactly the scale error this plugin
    exists to avoid.
    """
    try:
        image = PILImage.open(io.BytesIO(png))
        if image.mode != "RGB":
            # JPEG has no alpha channel; iOS screenshots arrive as RGBA.
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue(), "jpeg"
    except Exception as exc:
        logger.warning("could not re-encode the screenshot as JPEG (%s); sending the PNG", exc)
        return png, "png"


#: Accessibility-tree fields that are Android-only and always absent/false on iOS, so
#: they are never reported: filtering or branching on them silently drops everything.
#: Measured on a live iPhone — 412 nodes, `isClickable` false on every one.
_ANDROID_ONLY_FLAGS = ("isClickable", "isLongClickable")

#: Flags worth surfacing, and the value that makes them worth mentioning. Anything at its
#: boring default is omitted, which is most of them on most nodes.
_NOTABLE_FLAGS = {
    "isEnabled": False,
    "isFocused": True,
    "isPassword": True,
    "isScrollable": True,
    "isSelected": True,
    "isChecked": True,
}


def _label_of(node: dict) -> str:
    """A node's human-readable label, or "" when it has none.

    ``contentDescription`` is frequently a single space rather than empty, so a plain
    truthiness check counts hundreds of unlabelled container nodes as labelled.
    """
    for key in ("text", "contentDescription"):
        value = (node.get(key) or "").strip()
        if value:
            return value
    return ""


def _walk(node: dict):
    yield node
    for child in node.get("children") or []:
        if isinstance(child, dict):
            yield from _walk(child)


def compact_ui_state(payload: dict, *, max_elements: int = 150, scale: float = 1.0) -> dict:
    """Reduce a raw ``/ui-state`` response to something an agent can actually read.

    The raw response is **not usable as a tool result**: measured on a live iPhone it is
    183 KB / ~46,000 tokens for 412 nodes, of which only 85 carry any label. Returning it
    verbatim would spend most of a context window on empty container nodes.

    So this keeps the labelled nodes with their tappable centre, the handful of flags that
    change what an agent should do, and the app/keyboard state — and drops the tree
    structure, the unlabelled containers, and every Android-only field.

    ``center`` and ``size`` come back in **screenshot pixels** -- the one space the whole
    tool surface speaks, so a centre read here can go straight to ``tap``.

    ``/ui-state`` reports bounds in logical points (``device_context.screen_bounds``), which
    on a 3x phone is a third of the screenshot's pixel space, so ``scale`` converts them.
    Leaving that conversion to the caller was a real defect rather than a documentation
    problem: the gestures divide their input by the same scale, so feeding them a raw
    point-space centre landed the tap at a third of the intended position, and the natural
    thing for an agent to do -- read a centre, tap it -- was the broken thing.
    """
    context = payload.get("device_context") or {}
    bounds = context.get("screen_bounds") or {}
    phone = payload.get("phone_state") or {}

    elements = []
    offscreen: list[str] = []
    total = 0
    tree = payload.get("a11y_tree")
    for node in _walk(tree) if isinstance(tree, dict) else ():
        total += 1
        label = _label_of(node)
        if not label:
            continue
        box = node.get("boundsInScreen") or {}
        left, top = int(box.get("left", 0)), int(box.get("top", 0))
        right, bottom = int(box.get("right", 0)), int(box.get("bottom", 0))
        if right - left <= 0 or bottom - top <= 0:
            # A zero-size element is not on this screen — a real iPhone home screen
            # reports every app on another page, and every app-library entry, with
            # bounds of 0. Listing them with a centre of (0, 0) invites an agent to
            # "tap Amazon" and hit the top-left corner instead. Their labels are still
            # worth reporting: they say the app exists, just not here.
            offscreen.append(label)
            continue
        element = {
            "label": label,
            # XCUIElementTypeButton -> Button; the prefix is on every single node.
            "role": str(node.get("className") or "").replace("XCUIElementType", ""),
            "center": [round((left + right) / 2 * scale), round((top + bottom) / 2 * scale)],
            "size": [round((right - left) * scale), round((bottom - top) * scale)],
        }
        for flag, notable in _NOTABLE_FLAGS.items():
            if node.get(flag) == notable:
                element[flag] = notable
        elements.append(element)

    return {
        # Named for the space it is in, so a mismatch is visible in the payload itself.
        "coordinate_space": "screenshot_pixels",
        "screen_pixels": [round(int(bounds.get("width", 0)) * scale),
                          round(int(bounds.get("height", 0)) * scale)],
        "screen_points": [int(bounds.get("width", 0)), int(bounds.get("height", 0))],
        "app": {
            "package": phone.get("packageName"),
            "keyboard_visible": bool(phone.get("keyboardVisible")),
            "editable": bool(phone.get("isEditable")),
        },
        "elements": elements[:max_elements],
        # Present on the device but not on this screen, so not tappable. Labels only —
        # there is no coordinate that would mean anything.
        "offscreen_labels": offscreen[:max_elements],
        "elements_labelled": len(elements),
        "elements_total": total,
        # Says so explicitly rather than letting a caller assume it saw everything.
        "truncated": len(elements) > max_elements,
    }


class MobileRunTools:
    """Tool implementations bound to one device.

    Kept as a class (rather than module globals) so the tests can drive the tool bodies
    against a stub client without standing up a server or touching the environment.
    """

    def __init__(self, client: MobileRunClient, device_id: str) -> None:
        self.client = client
        self.device_id = device_id
        # Only the RATIO is cached. It is the display's density, so rotation leaves it
        # unchanged (both spaces swap together). The dimensions are deliberately not
        # cached: they DO change with rotation, and anything that needs them reads them
        # from a current screenshot.
        self._scale: Optional[float] = None

    async def _screen_pixels(self) -> Optional[tuple[int, int]]:
        """The screen's current pixel size, read fresh every time.

        Not cached: a rotation changes it, and a stale value would send `scroll` dragging
        along the wrong axis and report the wrong frame from `screen_size`.
        """
        return png_size(await self.client.screenshot(self.device_id))

    async def _pixel_scale(self) -> float:
        """Pixels per logical point for this device, resolved once and then cached.

        Costs one screenshot plus one ``/ui-state`` the first time. ``/ui-state`` is the
        only route reporting the logical-point size and is ~183 KB, which is why the
        result is cached — and why only the ratio is, since that is the display density
        and is rotation-invariant.

        A failure to measure returns 1.0 (a pass-through, correct for a caller already in
        points) **without caching it**. Caching the fallback would let one transient
        ``/ui-state`` blip silently misplace every gesture for the life of the process,
        which on a 3x display is most of the screen.
        """
        if self._scale is not None:
            return self._scale

        pixels = await self._screen_pixels()
        points = None
        try:
            ui = await self.client.ui_state(self.device_id)
            bounds = (ui.get("device_context") or {}).get("screen_bounds") or {}
            width, height = int(bounds.get("width") or 0), int(bounds.get("height") or 0)
            if width and height:
                points = (width, height)
        except MobileRunError as exc:
            logger.warning("could not read the logical-point size from /ui-state: %s", exc)

        if not (pixels and points):
            logger.warning(
                "could not establish the pixel-to-point scale (pixels=%s points=%s); treating "
                "incoming coordinates as device points and retrying on the next call",
                pixels,
                points,
            )
            return 1.0

        self._scale = pixels[0] / points[0]
        logger.info("device scale: %s px / %s pt = %.3f", pixels, points, self._scale)
        return self._scale

    async def _to_points(self, x: float, y: float) -> tuple[int, int]:
        """Convert a screenshot-pixel coordinate to the device-point coordinate /tap wants."""
        scale = await self._pixel_scale() or 1.0
        return round(x / scale), round(y / scale)

    # ------------------------------------------------------------ observation

    async def screenshot(self) -> Image:
        """Capture the screen. Coordinates you read off this image are what the gesture
        tools expect — do not rescale them."""
        png = await self.client.screenshot(self.device_id)
        size = png_size(png)
        data, image_format = to_jpeg(png)
        if size:
            logger.debug("screenshot %dx%d px, %d bytes as %s", *size, len(data), image_format)
        return Image(data=data, format=image_format)

    async def screen_size(self) -> str:
        """Screen dimensions. ``width``/``height`` are the screenshot's own pixels — the
        space every coordinate argument uses."""
        pixels = await self._screen_pixels()
        if not pixels:
            return json.dumps({"ok": False, "error": "the device returned an image this server cannot measure"})
        scale = await self._pixel_scale()
        return _ok(
            width=pixels[0],
            height=pixels[1],
            space="screenshot pixels — pass coordinates in this space",
            # Surfaced for diagnosis, not for the caller to apply: the conversion to the
            # device's logical points happens server-side.
            device_points=[round(pixels[0] / scale), round(pixels[1] / scale)] if scale else None,
            pixels_per_point=round(scale, 4),
        )

    async def ui_state(self) -> str:
        """List the labelled on-screen elements with their tappable centres.

        A compacted view, not the raw accessibility tree: the raw tree is ~46,000 tokens
        on a real phone, almost all of it unlabelled container nodes.

        ``center`` values are in screenshot pixels, the same space the gesture tools take,
        so a centre from here can be passed straight to ``tap``.
        """
        raw = await self.client.ui_state(self.device_id)
        if not isinstance(raw, dict):
            return json.dumps({"ok": False, "error": "unexpected /ui-state payload"})
        return json.dumps(compact_ui_state(raw, scale=await self._pixel_scale() or 1.0))

    async def wait(self, seconds: float = 2.0) -> str:
        """Sleep locally, then let the caller take a fresh screenshot."""
        secs = max(0.0, min(float(seconds), 15.0))
        await asyncio.sleep(secs)
        return _ok(waited_seconds=secs)

    async def wait_ready(self) -> str:
        return json.dumps(await self.client.wait_ready(self.device_id))

    # ---------------------------------------------------------------- gestures

    async def tap(self, x: int, y: int) -> str:
        """Tap a point. Use the coordinate exactly as you read it off the screenshot."""
        px, py = await self._to_points(x, y)
        await self.client.tap(self.device_id, px, py)
        return _ok(action="tap", x=int(x), y=int(y), device_point=[px, py])

    async def double_tap(self, x: int, y: int) -> str:
        """Tap a point twice. Coordinates as read off the screenshot."""
        # Composed: the device plane has no double-tap primitive.
        px, py = await self._to_points(x, y)
        await self.client.tap(self.device_id, px, py)
        await self.client.tap(self.device_id, px, py)
        return _ok(action="double_tap", x=int(x), y=int(y), device_point=[px, py])

    async def long_press(self, x: int, y: int, duration_ms: int = 800) -> str:
        """Press and hold a point. Coordinates as read off the screenshot."""
        # Composed as a zero-distance swipe, which is how a held touch is expressed
        # when the API offers only tap and swipe.
        px, py = await self._to_points(x, y)
        await self.client.swipe(self.device_id, px, py, px, py, duration_ms=duration_ms)
        return _ok(action="long_press", x=int(x), y=int(y), device_point=[px, py], duration_ms=int(duration_ms))

    async def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 300) -> str:
        """Drag from one point to another. Coordinates as read off the screenshot."""
        sx, sy = await self._to_points(start_x, start_y)
        ex, ey = await self._to_points(end_x, end_y)
        await self.client.swipe(self.device_id, sx, sy, ex, ey, duration_ms=duration_ms)
        return _ok(
            action="swipe",
            start=[int(start_x), int(start_y)],
            end=[int(end_x), int(end_y)],
            device_start=[sx, sy],
            device_end=[ex, ey],
        )

    async def scroll(self, direction: str = "down", amount: float = 0.5) -> str:
        """Scroll the screen by swiping across its middle.

        ``direction`` is the direction the *content* moves, i.e. ``down`` reveals what is
        below the fold. ``amount`` is a fraction of the screen (0.1-0.9).
        """
        key = direction.strip().lower()
        if key not in ("up", "down", "left", "right"):
            return json.dumps({"ok": False, "error": "direction must be up, down, left or right"})
        # Fresh, not cached: after a rotation a stale size drags along the wrong axis.
        pixels = await self._screen_pixels()
        if not pixels:
            return json.dumps({"ok": False, "error": "could not determine the screen size to scroll by"})
        width, height = pixels
        frac = max(0.1, min(float(amount), 0.9))
        cx, cy = width // 2, height // 2
        dx = int(width * frac / 2)
        dy = int(height * frac / 2)
        # Revealing content below the fold means dragging the finger UP, hence the inversion.
        moves = {
            "down": (cx, cy + dy, cx, cy - dy),
            "up": (cx, cy - dy, cx, cy + dy),
            "right": (cx + dx, cy, cx - dx, cy),
            "left": (cx - dx, cy, cx + dx, cy),
        }
        start_x, start_y, end_x, end_y = moves[key]
        # Through the same conversion as every other gesture, so there is one scale in play.
        sx, sy = await self._to_points(start_x, start_y)
        ex, ey = await self._to_points(end_x, end_y)
        await self.client.swipe(self.device_id, sx, sy, ex, ey, duration_ms=400)
        return _ok(action="scroll", direction=key, amount=frac)

    # ------------------------------------------------------------------- input

    async def type_text(self, text: str, clear: bool = False) -> str:
        """Type into the currently focused field, waiting for the text to land.

        Uses the device plane's ``committed`` completion mode, so a success here means the
        focused UI actually contains the text rather than merely that the keystrokes were
        accepted. Tap the field first; this does not focus anything.
        """
        await self.client.input_text(self.device_id, text, clear=clear, completion_mode="committed")
        return _ok(action="type_text", characters=len(text), cleared=bool(clear))

    async def clear_text(self) -> str:
        await self.client.clear_text(self.device_id)
        return _ok(action="clear_text")

    async def press_key(self, key: str) -> str:
        """Press a system key: one of the names in :data:`GLOBAL_ACTIONS`.

        ⚠️ These codes are the Android accessibility constants. The endpoint exists for
        iOS as well, but the mapping there is undocumented — on an iPhone, expect only
        ``home`` and ``back`` to be meaningful, and verify before relying on the rest.
        """
        name = key.strip().lower()
        code = GLOBAL_ACTIONS.get(name)
        if code is None:
            return json.dumps(
                {"ok": False, "error": f"unknown key {key!r}", "supported": sorted(GLOBAL_ACTIONS)}
            )
        await self.client.global_action(self.device_id, code)
        return _ok(action="press_key", key=name, code=code)

    # -------------------------------------------------------------------- apps

    async def list_apps(self) -> str:
        return json.dumps(await self.client.list_apps(self.device_id))

    async def launch_app(self, package: str) -> str:
        """Bring an app to the foreground by package name (Android) or bundle id (iOS)."""
        await self.client.start_app(self.device_id, package)
        return _ok(action="launch_app", package=package)

    async def open_deep_link(self, url: str) -> str:
        """Open a URL/deep link directly, skipping navigation through the home screen."""
        await self.client.open_deep_link(self.device_id, url)
        return _ok(action="open_deep_link", url=url)

    # --------------------------------------------------------------- clipboard

    async def get_clipboard(self) -> str:
        return json.dumps(await self.client.get_clipboard(self.device_id))

    async def set_clipboard(self, text: str) -> str:
        await self.client.set_clipboard(self.device_id, text)
        return _ok(action="set_clipboard", characters=len(text))


#: Tools that are only registered when the device advertises the named capability.
#: Everything not listed here is part of the always-available core surface.
CAPABILITY_GATED: dict[str, str] = {
    "get_clipboard": "clipboard",
    "set_clipboard": "clipboard",
    "ui_state": "accessibility",
}

#: The always-available surface, in the order it is advertised.
CORE_TOOLS: tuple[str, ...] = (
    "screenshot",
    "screen_size",
    "wait",
    "wait_ready",
    "tap",
    "double_tap",
    "long_press",
    "swipe",
    "scroll",
    "type_text",
    "clear_text",
    "press_key",
    "list_apps",
    "launch_app",
    "open_deep_link",
)


def select_tools(capabilities: dict[str, bool]) -> list[str]:
    """Tool names to register, given a device's capability map.

    An unknown capability (absent from the map entirely) is treated as **present**: the
    map is authoritative about what a device supports, but a newly-added capability name
    that this plugin has not seen should not silently remove a working tool. Only an
    explicit ``False`` gates a tool out.
    """
    names = list(CORE_TOOLS)
    for tool_name, capability in CAPABILITY_GATED.items():
        if capabilities.get(capability, True):
            names.append(tool_name)
        else:
            logger.info("not registering %s: device reports capability %s=false", tool_name, capability)
    return names


def _declared_capabilities() -> Optional[dict[str, bool]]:
    """Capabilities passed in by the deployer, if any.

    ``MOBILERUN_CAPABILITIES`` lets the env resolve the map once at deploy time (where it
    already talks to the API) instead of making every server instance re-discover it.

    The value is ``name=true,name2=false`` and must include the ``false`` entries.
    :func:`select_tools` treats an absent name as present, so a payload listing only the
    supported names would gate nothing out — the deployer has to say what is *not*
    supported for the gate to do anything.

    Not JSON, because this arrives through a docker-compose ``environment:`` entry and a
    YAML plain scalar may not contain ``": "``.
    """
    raw = (os.environ.get("MOBILERUN_CAPABILITIES") or "").strip()
    if not raw:
        return None
    parsed: dict[str, bool] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, sep, value = pair.partition("=")
        if not sep or not name.strip():
            logger.warning(
                "ignoring malformed MOBILERUN_CAPABILITIES entry %r (expected name=true/false)", pair
            )
            continue
        parsed[name.strip()] = value.strip().lower() in ("1", "true", "yes")
    return parsed or None


# --- the environment card and the core data plane -----------------------------
#
# An env deployed WITHOUT a gateway has to serve its own card: the gateway used to do
# this on a server's behalf, and EnvironmentServerProvider fails the deploy unless the
# card appears here, under the env's own environment_name, within its readiness window.
#
# Served by the protocol's own server SDK rather than hand-written. It publishes to PyPI
# as `agentenv-framework-protocol` -- the package name follows `agentenv-framework`, not
# the repo's directory name, which is why searching for `agentenv-protocol` turns up
# nothing. It is three dependencies deep (httpx, pydantic, starlette), all of which the
# image already has.
#
# Using it is not a convenience. A hand-written card drifts from the thing it describes:
# the first version of this one advertised JSON-RPC at /agentenv, which this server did
# not serve, because copying the card's shape does not copy the routes behind it.
# `add_routes_to_app` mounts the card AND the data plane together, so the card cannot
# advertise an address that 404s.
class NoDataPlane:
    """The RPC handler, with no data operations.

    ``data/add``, ``data/get`` and ``data/reset`` address an environment database, and a
    phone fleet has none. The nearest thing to a reset on this device plane is
    :meth:`MobileRunClient.factory_reset`, which wipes a rented phone's apps and user
    data with no undo -- deliberately not reachable from the protocol.

    Registering nothing is a claim, not an omission: the SDK advertises
    ``capabilities.operations: []``, and an ABSENT ``operations`` would instead read as
    "pre-advertisement card", which a consumer takes as the full add/get/reset trio. The
    dispatcher still answers at :data:`RPC_PATH` -- with ``method not found``, which is
    the honest answer and the one every other zero-operation env gives.
    """


def _protocol_app(name: str) -> AgentEnvFastMCPApplication:
    """The card + data plane for this env, unmounted.

    There is deliberately no ``environment_card()`` accessor to read the card from. The
    served card is only final after :meth:`add_routes_to_app`, which fills in the MCP
    interface from the app it is mounting onto and the operations from the handler; a
    function returning the card before that would describe something the server does not
    serve, which is the mistake this module already made once. Read the card the way a
    consumer does -- GET :data:`WELL_KNOWN_PATH`.

    What is NOT in it: ``capabilities.tools``. The tool surface is gated per device (see
    :func:`select_tools`), so a frozen list would drift from what this process actually
    serves. The provider does a live ``tools/list`` against the MCP path, which cannot.
    """
    return AgentEnvFastMCPApplication(EnvironmentCard(name=name), NoDataPlane())


def build_server() -> FastMCP:
    """Construct the FastMCP app with the tools this device actually supports."""
    api_key = _require("MOBILERUN_API_KEY")
    device_id = _require("MOBILERUN_DEVICE_ID")
    base_url = (os.environ.get("MOBILERUN_BASE_URL") or "").strip()

    client = MobileRunClient(api_key, **({"base_url": base_url} if base_url else {}))
    tools = MobileRunTools(client, device_id)

    capabilities = _declared_capabilities()
    if capabilities is None:
        try:
            capabilities = asyncio.run(client.capabilities(device_id))
        except MobileRunError as exc:
            # Don't refuse to boot over this: the capability map is an optimisation, and a
            # gated tool failing at call time is a better outcome than no server at all.
            logger.warning("could not read the capability map (%s); registering the full surface", exc)
            capabilities = {}

    mcp = FastMCP("MobileRun")
    mcp.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
    mcp.settings.port = int(os.environ.get("MCP_PORT", "18765"))
    # Gateways reach MCP servers by their compose hostname, which mcp's default
    # localhost-only allowlist rejects.
    mcp.settings.transport_security.enable_dns_rebinding_protection = False

    for name in select_tools(capabilities):
        method = getattr(tools, name)
        mcp.add_tool(method, name=f"mobilerun_{name}", description=(method.__doc__ or "").strip() or None)

    # The name the card is served under must match the env's environment_name, or a
    # gateway-less deploy is refused. agent-env injects it; the fallback keeps a bare
    # `python -m agentenv_mobilerun.server.main` usable for local poking.
    environment_name = (os.environ.get("ENVIRONMENT_NAME") or "mobilerun").strip()

    # Mounts the card at WELL_KNOWN_PATH and the data plane at RPC_PATH, and fills the
    # card's MCP interface in from this app's own streamable_http_path.
    _protocol_app(environment_name).add_routes_to_app(mcp)

    logger.info(
        "MobileRun MCP server ready: device=%s tools=%d card=%s name=%s",
        device_id, len(select_tools(capabilities)), WELL_KNOWN_PATH, environment_name,
    )
    return mcp


def main() -> None:
    build_server().run(transport="streamable-http")


if __name__ == "__main__":
    main()
