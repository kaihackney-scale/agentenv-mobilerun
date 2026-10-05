"""Async REST client for the MobileRun (Droidrun Cloud v1) device plane.

Deliberately thin and dependency-light (``httpx`` only) so the same module can be
installed into the MCP server image without pulling agent-env in. Every method maps
to one documented route from ``https://api.mobilerun.ai/v1/openapi.json``; nothing
here knows about agent-env.

Four API behaviours are handled here rather than at every call site, because each
one has already bitten us:

1. ``GET /devices/{id}`` returns a live, device-scoped ``streamToken`` (an EdDSA JWT)
   in the body. :meth:`get_device` strips it before returning, so a device record can
   be logged or echoed by a CLI without leaking a credential.
2. Collection routes disagree about envelopes. ``GET /devices`` returns
   ``{"items": [...], "pagination": {...}}``, while ``GET /devices/{id}/apps`` and
   ``GET /devices/{id}/recordings`` return a **bare list** (both verified live).
   :func:`_items` accepts either shape; code written against only the documented
   envelope dies with ``'list' object has no attribute 'get'``.
3. ``GET /devices`` is **paginated** (``pageSize`` defaults to 20) and **pre-filtered**
   (``state`` defaults to ``["ready"]``). Reading ``items`` alone therefore silently
   truncates a large fleet, and silently hides every device that is provisioning, in
   maintenance, or stopped. :meth:`list_devices` follows the pages and takes an explicit
   ``states`` argument for that reason — a caller asking "where is my device?" must be
   able to see one that is not ready.
3. ``GET /recordings/{id}/video`` is a **302** to a presigned R2 URL that expires in
   900 seconds, so the artifact must be copied promptly and never linked to.
4. A device's ``state`` field is **not** a health signal — a phone that cannot be
   driven at all still reports ``state: "ready"``. :meth:`health_probe` takes a
   screenshot instead, which is the only cheap call that actually exercises the
   device path.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.mobilerun.ai/v1"

#: Read timeout for calls that drive the phone. Device actions routinely take
#: seconds, and a degraded phone can hang a screenshot for the best part of a
#: minute, so the default httpx 5s timeout is far too tight.
DEFAULT_TIMEOUT_SECONDS = 60.0

#: Fields that must never leave this module. ``streamToken`` is a live credential;
#: ``streamUrl`` is only meaningful with it and is equally short-lived.
_REDACTED_DEVICE_FIELDS = ("streamToken", "streamUrl")

#: Every device state the API defines. Needed in full because ``GET /devices`` applies a
#: server-side default of ``["ready"]`` when the ``state`` parameter is absent: **omitting
#: it does not mean "no filter"**, it means "ready only". Asking for the whole fleet
#: requires naming every state. Verified live — leaving the parameter off returned 10 of 13
#: devices and hid an Android phone sitting in ``maintenance``.
#:
#: ⚠️ And it must be sent **comma-joined in one parameter**, not as a repeated key. The
#: spec declares ``state`` as ``type: array``, which conventionally implies
#: ``?state=a&state=b``, but this server reads that form wrongly and — worse — does so
#: *silently*: measured against the same 13-device account, ``?state=ready&state=maintenance``
#: returned 10 (dropping the maintenance device), all ten states as repeated keys returned
#: **0**, and the comma-joined form returned all 13. A wrong answer with HTTP 200 is why
#: this is encoded in one place and pinned by a test.
#: Recording statuses that mean "this will never complete". ``expired`` is in the list
#: because it was observed live on a real recording (with an ``error`` field alongside) —
#: without it, a recording whose retention lapsed is polled until the timeout instead of
#: failing immediately with the reason.
TERMINAL_BAD_RECORDING_STATUSES = ("failed", "error", "cancelled", "expired")

ALL_DEVICE_STATES = (
    "creating",
    "assigned",
    "ready",
    "rebooting",
    "migrating",
    "resetting",
    "terminated",
    "maintenance",
    "stopped",
    "unknown",
)

#: Android ``AccessibilityService.GLOBAL_ACTION_*`` constants. ``POST /devices/{id}/global``
#: takes a bare integer ``action`` and the API documents no enum, so these are the
#: platform constants the endpoint is modelled on.
#:
#: ⚠️ Verified meaning on Android only. The route exists for iOS devices too but the
#: mapping there is undocumented and unprobed — see ``agent-env mobilerun doctor
#: --probe-global``. Treat anything other than HOME/BACK on iOS as unknown.
GLOBAL_ACTIONS: dict[str, int] = {
    "back": 1,
    "home": 2,
    "recents": 3,
    "notifications": 4,
    "quick_settings": 5,
    "power_dialog": 6,
    "toggle_split_screen": 7,
    "lock_screen": 8,
    "take_screenshot": 9,
}


class MobileRunError(RuntimeError):
    """A MobileRun API call failed.

    ``status_code`` is ``None`` for transport-level failures (DNS, TLS, timeout), which
    callers must distinguish from a real rejection: a timeout means "unknown", while a
    403 means "will not work until the key changes".
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class DeviceUnavailable(MobileRunError):
    """No device matched the request, or the matched device failed its health probe."""


@dataclass(frozen=True)
class Device:
    """The subset of a MobileRun device record this plugin relies on.

    ``raw`` keeps everything else for diagnostics, with the stream credential already
    stripped by :meth:`MobileRunClient.get_device`.
    """

    id: str
    platform: str
    state: str
    name: Optional[str] = None
    active_task_id: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Device":
        return cls(
            id=str(data.get("id") or ""),
            platform=str(data.get("platform") or "").lower(),
            state=str(data.get("state") or ""),
            name=data.get("name"),
            # The fleet-listing SDK spells this active_task_id; the REST body spells it
            # activeTask. Accept both rather than depending on which one produced the dict.
            active_task_id=data.get("active_task_id") or data.get("activeTask") or data.get("activeTaskId"),
            raw=data,
        )

    @property
    def idle(self) -> bool:
        return self.active_task_id in (None, "")


def _items(payload: Any) -> list[dict[str, Any]]:
    """Normalise a MobileRun collection response.

    ``/devices`` returns ``{"items": [...]}``; ``/devices/{id}/apps`` and
    ``/devices/{id}/recordings`` return a bare list. Both shapes verified live, so this
    is not a spec-reading guess.
    """
    if isinstance(payload, list):
        return [d for d in payload if isinstance(d, dict)]
    if isinstance(payload, dict):
        inner = payload.get("items")
        if isinstance(inner, list):
            return [d for d in inner if isinstance(d, dict)]
    return []


def _redact_device(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k not in _REDACTED_DEVICE_FIELDS}


class MobileRunClient:
    """Async client over the MobileRun device plane.

    A thin layer over ``mobilerun_sdk.AsyncMobilerun`` -- the vendor's official SDK --
    rather than hand-rolled HTTP. The SDK owns request construction, auth, retries and
    the device plane's quirks; this class keeps what the SDK does not model:

    * :meth:`resolve_device` and :meth:`health_probe`, because ``state`` is not a health
      signal (see their docstrings). This is the whole reason the class still exists.
    * ``Device``, with ``streamToken``/``streamUrl`` redacted before anything logs it.
    * Recording completion polling and its terminal-status set.
    * ``MobileRunError`` / ``DeviceUnavailable``, so callers keep one exception contract.

    Responses come back as the **raw** JSON the API sent, via the SDK's
    ``with_raw_response``, not as its pydantic models. The MCP server reads wire keys
    directly (``boundsInScreen``, ``contentDescription``, ``keyboardVisible``), and a
    model round-trip would rename them; this keeps the payload contract identical while
    still getting the SDK's request layer.

    Owns its ``httpx.AsyncClient`` unless one is injected (which is how the tests drive
    it -- the SDK accepts the same client, so ``MockTransport`` still works). Use as an
    async context manager, or call :meth:`aclose`.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        client: Optional[httpx.AsyncClient] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        if not api_key:
            raise MobileRunError("a MobileRun API key is required (create one on the API Keys page)")
        from mobilerun_sdk import AsyncMobilerun

        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        # The SDK sets Authorization itself; this client carries only transport concerns.
        # follow_redirects: the recording artefact routes answer 302 to a presigned URL.
        self._client = client or httpx.AsyncClient(
            timeout=timeout, transport=transport, follow_redirects=True
        )
        # Retries are the SDK's, and they are a real behaviour change: a transient 500 on
        # /ui-state used to fall through to the 1.0 scale fallback and misplace the
        # gesture, where now it is retried. `max_retries=0` is for tests that inject a
        # failure and need the un-retried path to be deterministic (and fast).
        sdk_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self._base_url,
            "timeout": timeout,
            "http_client": self._client,
        }
        if max_retries is not None:
            sdk_kwargs["max_retries"] = max_retries
        self._sdk = AsyncMobilerun(**sdk_kwargs)
        # ... and a second view of the same connection pool that does NOT retry, for the
        # calls that change the device.
        #
        # The SDK retries 408/409/429/5xx and every transport failure, with no regard for
        # the HTTP method, and sends no idempotency key on the wire (it generates a retry
        # key, but `_idempotency_header` is None, so nothing carries it). A write whose
        # response was lost is therefore simply sent again: a second tap on whatever is
        # now under the finger, a second `input_text` appended to the field, a second
        # recording left running. The device plane offers no deduplication to lean on.
        #
        # Reads keep the retries -- replaying a screenshot or a /ui-state costs nothing
        # and is exactly where a transient 500 used to misplace a gesture. The split is
        # visible at every call site: `_w_*` namespaces do not retry, the rest do.
        self._writes = self._sdk.with_options(max_retries=0)

    async def __aenter__(self) -> "MobileRunClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ---------------------------------------------------------------- transport

    @staticmethod
    def _translate(what: str, exc: BaseException) -> "MobileRunError":
        """The SDK's exception hierarchy mapped onto this module's one contract.

        A status error keeps its code, because callers branch on it (404 vs 409 vs 503).
        A connection or timeout error deliberately carries **no** status: the call never
        reached a decision, so it must not read as "the device said no".
        """
        from mobilerun_sdk import APIStatusError

        if isinstance(exc, APIStatusError):
            return MobileRunError(
                f"{what} -> HTTP {exc.response.status_code}: {_error_detail(exc.response)}",
                status_code=exc.response.status_code,
            )
        return MobileRunError(f"{what} failed in transport: {exc}")

    async def _raw(self, what: str, call: Any) -> httpx.Response:
        """Await an SDK ``with_raw_response`` call and hand back the httpx response."""
        from mobilerun_sdk import MobilerunError as SdkError

        try:
            response = await call
        except SdkError as exc:
            raise self._translate(what, exc) from exc
        except httpx.HTTPError as exc:
            raise self._translate(what, exc) from exc
        return response.http_response

    async def _json(self, what: str, call: Any) -> Any:
        response = await self._raw(what, call)
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise MobileRunError(f"{what} returned non-JSON body: {exc}") from exc

    async def _bytes(self, what: str, call: Any) -> bytes:
        return (await self._raw(what, call)).content

    # Namespaces on the retrying client. Reads only -- see __init__.
    @property
    def _devices(self) -> Any:
        return self._sdk.devices.with_raw_response

    @property
    def _dev_state(self) -> Any:
        return self._sdk.devices.state.with_raw_response

    @property
    def _dev_apps(self) -> Any:
        return self._sdk.devices.apps.with_raw_response

    @property
    def _dev_recordings(self) -> Any:
        return self._sdk.devices.recordings.with_raw_response

    @property
    def _dev_clipboard(self) -> Any:
        return self._sdk.devices.clipboard.with_raw_response

    # The same namespaces on the non-retrying client, for anything that changes the
    # device. A `_w_` prefix at the call site is the whole marker of a write.
    @property
    def _w_devices(self) -> Any:
        return self._writes.devices.with_raw_response

    @property
    def _w_actions(self) -> Any:
        return self._writes.devices.actions.with_raw_response

    @property
    def _w_keyboard(self) -> Any:
        return self._writes.devices.keyboard.with_raw_response

    @property
    def _w_apps(self) -> Any:
        return self._writes.devices.apps.with_raw_response

    @property
    def _w_recordings(self) -> Any:
        return self._writes.devices.recordings.with_raw_response

    @property
    def _w_clipboard(self) -> Any:
        return self._writes.devices.clipboard.with_raw_response

    @property
    def _w_deep_link(self) -> Any:
        return self._writes.devices.deep_link.with_raw_response

    # ------------------------------------------------------------------ devices

    async def list_devices(
        self,
        *,
        platform: Optional[str] = None,
        states: Optional[Iterable[str]] = ("ready",),
        device_type: Optional[str] = None,
        page_size: int = 100,
        max_pages: int = 50,
    ) -> list[Device]:
        """Devices on the account, following pagination to the end.

        ``states`` mirrors the API's own ``state`` filter. Pass ``None`` for the whole
        fleet, which sends :data:`ALL_DEVICE_STATES` explicitly -- **not** an absent
        parameter, because the server treats absence as ``["ready"]``. That distinction is
        the difference between seeing the fleet and seeing a filtered view of it: an
        account can hold an Android phone in ``maintenance`` that a bare ``GET /devices``
        reports as simply not existing.

        The SDK comma-joins the state list (a repeated key returns 200 with the wrong
        rows), so that encoding is now the vendor's contract rather than ours.

        Pages are followed until ``hasNext`` is false, bounded by ``max_pages`` so a
        server that always reports another page cannot spin here forever.
        """
        wanted_states = list(ALL_DEVICE_STATES if states is None else states)
        devices: list[Device] = []
        for page in range(1, max_pages + 1):
            kwargs: dict[str, Any] = {"state": wanted_states, "page_size": page_size, "page": page}
            if device_type:
                kwargs["type"] = device_type
            payload = await self._json("GET /devices", self._devices.list(**kwargs))
            devices.extend(Device.from_api(_redact_device(d)) for d in _items(payload))
            pagination = payload.get("pagination") if isinstance(payload, dict) else None
            if not isinstance(pagination, dict) or not pagination.get("hasNext"):
                break
        else:
            logger.warning(
                "stopped listing devices after %d pages; the fleet may be larger than reported",
                max_pages,
            )

        if platform:
            wanted = platform.lower()
            devices = [d for d in devices if d.platform == wanted]
        return devices

    async def fleet_summary(self) -> dict:
        """Device counts by state and type, including devices ``list_devices`` filters out.

        Answers "what does this account actually have?" in one call, which the paginated,
        state-filtered listing cannot.
        """
        payload = await self._json("GET /devices/count", self._devices.count())
        return payload if isinstance(payload, dict) else {}

    async def get_device(self, device_id: str) -> Device:
        what = f"GET /devices/{device_id}"
        payload = await self._json(what, self._devices.retrieve(device_id))
        if not isinstance(payload, dict):
            raise MobileRunError(f"{what} returned {type(payload).__name__}, expected an object")
        return Device.from_api(_redact_device(payload))

    async def capabilities(self, device_id: str) -> dict[str, bool]:
        """The device's capability map -- the supported way to find out what it can do.

        Prefer this over probing for 404/405/501 responses: it is one cheap call and it
        is authoritative per device (two phones on the same account differ).
        """
        payload = await self._json(
            f"GET /devices/{device_id}/capabilities", self._devices.retrieve_capabilities(device_id)
        )
        if not isinstance(payload, dict):
            return {}
        source = payload.get("capabilities") if isinstance(payload.get("capabilities"), dict) else payload
        return {str(k): bool(v) for k, v in source.items() if isinstance(v, bool)}

    async def health_probe(self, device_id: str) -> bool:
        """True when the device actually responds with a screenshot.

        This exists because ``state`` lies: a phone whose taps 502 and whose screenshots
        time out still reports ``state: "ready"``, with an unchanged ``stateMessage``.
        A screenshot is the cheapest call that traverses the whole device path.
        """
        try:
            await self.screenshot(device_id)
        except MobileRunError as exc:
            logger.warning("device %s failed its health probe: %s", device_id, exc)
            return False
        return True

    async def resolve_device(
        self,
        *,
        platform: Optional[str] = None,
        device_id: Optional[str] = None,
        require_healthy: bool = True,
    ) -> Device:
        """Pick the device to drive: an explicit id, else the first idle healthy match.

        Raises :class:`DeviceUnavailable` with a message naming what was actually found,
        because "no device available" on its own sends people to the wrong dashboard.

        ⚠️ This makes **no reservation** -- MobileRun's device plane has no lease, so two
        concurrent deploys pointed at the same account can resolve the same phone.
        Exclusivity has to come from one job per device upstream.
        """
        if device_id:
            device = await self.get_device(device_id)
            if require_healthy and not await self.health_probe(device.id):
                raise DeviceUnavailable(
                    f"device {device.id} reports state={device.state!r} but failed its screenshot "
                    "health probe, so it cannot be driven. Reboot it from the MobileRun dashboard, or "
                    "pick another device; if a reboot does not clear it, this is a MobileRun support item."
                )
            return device

        devices = await self.list_devices(platform=platform)
        if not devices:
            scope = f" for platform {platform!r}" if platform else ""
            raise DeviceUnavailable(
                f"the MobileRun account has no devices{scope}. Provision one in the dashboard first "
                "(https://mobilerun.ai/pricing — Android and bring-your-own are self-serve; a rented "
                "iPhone currently requires contacting their team)."
            )
        candidates = [d for d in devices if d.state == "ready" and d.idle]
        if not candidates:
            states = ", ".join(sorted({f"{d.id}={d.state}" for d in devices}))
            raise DeviceUnavailable(f"no idle ready device{_scope(platform)}; saw {states}")
        for device in candidates:
            if not require_healthy or await self.health_probe(device.id):
                return device
        raise DeviceUnavailable(
            f"every ready device{_scope(platform)} failed its screenshot health probe "
            f"({', '.join(d.id for d in candidates)}). `state` is not a health signal, so these look "
            "available in the dashboard but cannot be driven."
        )

    # ------------------------------------------------------------------ actions

    async def screenshot(self, device_id: str) -> bytes:
        """Raw PNG bytes of the current screen."""
        return await self._bytes(
            f"GET /devices/{device_id}/screenshot", self._dev_state.screenshot(device_id)
        )

    async def tap(self, device_id: str, x: int, y: int, *, stealth: bool = False) -> Any:
        return await self._json(
            f"POST /devices/{device_id}/tap",
            self._w_actions.tap(device_id=device_id, x=int(x), y=int(y), stealth=stealth),
        )

    async def swipe(
        self,
        device_id: str,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        *,
        duration_ms: int = 300,
        stealth: bool = False,
    ) -> Any:
        return await self._json(
            f"POST /devices/{device_id}/swipe",
            self._w_actions.swipe(
                device_id=device_id,
                start_x=int(start_x),
                start_y=int(start_y),
                end_x=int(end_x),
                end_y=int(end_y),
                duration=max(10, int(duration_ms)),
                stealth=stealth,
            ),
        )

    async def input_text(
        self,
        device_id: str,
        text: str,
        *,
        clear: bool = False,
        completion_mode: str = "committed",
    ) -> Any:
        """Type into the focused field.

        ``completion_mode="committed"`` is the default on purpose: it makes the server
        wait until the focused UI state actually contains the text (or goes quiescent)
        instead of returning as soon as the input provider accepted the keystrokes. That
        is the difference between "typed" and "typed and landed", and it removes the
        class of failure where a field silently drops characters.
        """
        if completion_mode not in ("accepted", "committed"):
            raise MobileRunError(f"completion_mode must be 'accepted' or 'committed', got {completion_mode!r}")
        return await self._json(
            f"POST /devices/{device_id}/keyboard",
            self._w_keyboard.write(
                device_id=device_id, text=text, clear=clear, completion_mode=completion_mode
            ),
        )

    async def clear_text(self, device_id: str) -> Any:
        return await self._json(
            f"DELETE /devices/{device_id}/keyboard", self._w_keyboard.clear(device_id)
        )

    async def global_action(self, device_id: str, action: int) -> Any:
        return await self._json(
            f"POST /devices/{device_id}/global",
            self._w_actions.global_(device_id=device_id, action=int(action)),
        )

    async def ui_state(self, device_id: str) -> Any:
        return await self._json(f"GET /devices/{device_id}/ui-state", self._dev_state.ui(device_id))

    async def wait_ready(self, device_id: str) -> Any:
        """Block until the device plane calls the device ready, and return its record, redacted.

        ``GET /devices/{id}/wait`` answers the full device record -- the same one ``GET
        /devices/{id}`` does, live ``streamToken`` and ``streamUrl`` included. The MCP
        server's ``wait_ready`` tool returns this to the agent, so unredacted it put a
        device-scoped credential in the model's context and in every trajectory recorded
        from it. The hand-written client hit a route that returned no record; the SDK's
        ``wait_ready`` does, and this was the one device-record path left unfiltered.
        """
        payload = await self._json(
            f"GET /devices/{device_id}/wait", self._devices.wait_ready(device_id)
        )
        return _redact_device(payload) if isinstance(payload, dict) else payload

    # --------------------------------------------------------------------- apps

    async def list_apps(self, device_id: str) -> list[dict[str, Any]]:
        return _items(await self._json(f"GET /devices/{device_id}/apps", self._dev_apps.list(device_id)))

    async def start_app(self, device_id: str, package: str, *, activity: Optional[str] = None) -> Any:
        """Launch an app. The body is mandatory even when empty, or the API answers 400.

        That requirement is the SDK's to honour now; it sends ``{}`` when no activity is
        given, which is what this originally had to discover by hand.
        """
        kwargs: dict[str, Any] = {"device_id": device_id}
        if activity:
            kwargs["activity"] = activity
        return await self._json(
            f"PUT /devices/{device_id}/apps/{package}", self._w_apps.start(package, **kwargs)
        )

    async def stop_app(self, device_id: str, package: str, *, clear_data: bool = False) -> Any:
        """Stop an app, optionally clearing its data.

        ⚠️ This is a distinct operation from uninstalling: the same path with ``DELETE``
        removes the app entirely. The SDK models the two as ``apps.stop`` and
        ``apps.delete``, so the distinction is no longer one hand-written verb apart.
        """
        return await self._json(
            f"PATCH /devices/{device_id}/apps/{package}",
            self._w_apps.stop(package, device_id=device_id, clear_data=bool(clear_data)),
        )

    async def uninstall_app(self, device_id: str, package: str) -> Any:
        """Remove an app. Irreversible on a rented phone -- see :meth:`stop_app`."""
        return await self._json(
            f"DELETE /devices/{device_id}/apps/{package}",
            self._w_apps.delete(package, device_id=device_id),
        )

    async def open_deep_link(self, device_id: str, url: str) -> Any:
        return await self._json(
            f"POST /devices/{device_id}/apps/open-deep-link",
            self._w_deep_link.execute_deep_link(
                device_id=device_id, deep_link=url
            ),
        )

    async def get_clipboard(self, device_id: str) -> Any:
        return await self._json(
            f"GET /devices/{device_id}/clipboard",
            self._dev_clipboard.get(device_id),
        )

    async def set_clipboard(self, device_id: str, text: str) -> Any:
        return await self._json(
            f"POST /devices/{device_id}/clipboard",
            self._w_clipboard.set(device_id=device_id, text=text),
        )

    async def press_home(self, device_id: str) -> Any:
        return await self.global_action(device_id, GLOBAL_ACTIONS["home"])

    # --------------------------------------------------------------- recordings

    async def start_recording(
        self,
        device_id: str,
        *,
        name: Optional[str] = None,
        types: Iterable[str] = ("video", "trajectory"),
        quality: Optional[int] = None,
        retention_days: Optional[int] = None,
    ) -> str:
        """Start a server-side recording and return its id.

        The trajectory stream carries exact coordinates, ``seq``, ``at_ms``, gesture
        durations, the display scale, and ``video.timeline.action_zero_to_video_ms`` plus
        an ``uncertainty_ms`` -- i.e. the device plane aligns the action clock to the video
        itself, which is the alignment a client cannot reconstruct after the fact.
        """
        what = f"POST /devices/{device_id}/recordings"
        kwargs: dict[str, Any] = {"types": list(types)}
        if name:
            kwargs["name"] = name
        if quality is not None:
            kwargs["quality"] = quality
        if retention_days is not None:
            kwargs["retention_days"] = retention_days
        payload = await self._json(what, self._w_recordings.start(device_id, **kwargs))
        recording_id = (payload or {}).get("id") if isinstance(payload, dict) else None
        if not recording_id:
            raise MobileRunError(f"{what} returned no recording id: {payload!r}")
        return str(recording_id)

    async def stop_recording(self, device_id: str, recording_id: str) -> Any:
        return await self._json(
            f"POST /devices/{device_id}/recordings/{recording_id}",
            self._w_recordings.stop(recording_id, device_id=device_id),
        )

    async def get_recording(self, device_id: str, recording_id: str) -> dict[str, Any]:
        payload = await self._json(
            f"GET /devices/{device_id}/recordings/{recording_id}",
            self._dev_recordings.status(recording_id, device_id=device_id),
        )
        return payload if isinstance(payload, dict) else {}

    async def await_recording(
        self,
        device_id: str,
        recording_id: str,
        *,
        timeout_seconds: float = 300.0,
        poll_seconds: float = 3.0,
    ) -> dict[str, Any]:
        """Poll until the recording reaches ``completed``.

        The terminal status is ``completed``, **not** ``ready`` -- the lifecycle is
        ``recording -> uploading -> completed``, so a poll that waits for "ready" waits
        forever on a recording that already succeeded.
        """
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        last: dict[str, Any] = {}
        while True:
            last = await self.get_recording(device_id, recording_id)
            status = str(last.get("status") or "")
            if status == "completed":
                return last
            if status in TERMINAL_BAD_RECORDING_STATUSES:
                raise MobileRunError(f"recording {recording_id} ended with status {status!r}: {last!r}")
            if asyncio.get_running_loop().time() >= deadline:
                raise MobileRunError(
                    f"recording {recording_id} was still {status!r} after {timeout_seconds:.0f}s "
                    "(expected 'completed')"
                )
            await asyncio.sleep(poll_seconds)

    async def download_recording_video(self, device_id: str, recording_id: str) -> bytes:
        """Fetch the mp4.

        The route answers **302** to a presigned R2 URL whose signature expires in 900
        seconds, so this follows the redirect (see ``follow_redirects`` in ``__init__``)
        and returns bytes rather than handing back a link. Copy it into durable storage
        immediately: MobileRun applies ``retentionDays`` and then 410s the artifact routes.
        """
        return await self._bytes(
            f"GET /devices/{device_id}/recordings/{recording_id}/video",
            self._dev_recordings.video(recording_id, device_id=device_id),
        )

    async def download_recording_trajectory(self, device_id: str, recording_id: str) -> bytes:
        """Fetch the JSONL trajectory (header record + one record per input action)."""
        return await self._bytes(
            f"GET /devices/{device_id}/recordings/{recording_id}/trajectory",
            self._dev_recordings.trajectory(recording_id, device_id=device_id),
        )

    async def list_recordings(self, device_id: str) -> list[dict[str, Any]]:
        return _items(
            await self._json(
                f"GET /devices/{device_id}/recordings", self._dev_recordings.list(device_id)
            )
        )

    # ------------------------------------------------------------- device state

    async def reboot(self, device_id: str) -> Any:
        """Reboot the phone.

        Survivable but slow, and it can strand a device: a reboot has left one at the
        lockdown pairing prompt, where every subsequent call fails while ``state`` still
        reads ``ready``.
        """
        return await self._json(
            f"POST /devices/{device_id}/reboot", self._w_devices.reboot(device_id)
        )

    async def factory_reset(self, device_id: str) -> Any:
        """⚠️ DESTRUCTIVE. Wipes installed apps and user data on a rented phone.

        There is no undo and no confirmation on the device plane. Nothing in this package
        calls it; it exists so the CLI can expose it behind an explicit flag.
        """
        return await self._json(
            f"POST /devices/{device_id}/reset", self._w_devices.reset(device_id)
        )

def _scope(platform: Optional[str]) -> str:
    return f" for platform {platform!r}" if platform else ""


def _error_detail(response: httpx.Response) -> str:
    """A short, useful error string from a MobileRun failure body."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(payload, dict):
        for key in ("detail", "message", "error", "title"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value[:300]
    return str(payload)[:300]
