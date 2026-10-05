"""``agent-env mobilerun ...``: the four commands, driven through Click's runner.

These went unwritten while the client and server were covered, which left the largest
untested surface in the package sitting behind a command `doctor` that the README
advertises as a CI gate ("Exits non-zero if a deploy would fail"). An untested gate is
worse than no gate, so the exit codes are what most of this file asserts.

No agentenv-framework needed: the commands resolve a key through `api_key.py` and reach
the device plane through `_client()`, both of which are patched here.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pytest
from click.testing import CliRunner

from agentenv_mobilerun import cli as cli_mod
from agentenv_mobilerun.client import Device, DeviceUnavailable, MobileRunError


def _device(device_id: str, *, state: str = "ready", platform: str = "ios", idle: bool = True) -> Device:
    return Device.from_api({
        "id": device_id,
        "platform": platform,
        "state": state,
        "name": f"phone-{device_id}",
        "activeTaskId": None if idle else "task-1",
        "type": "device_slot",
    })


class FakeClient:
    """Stands in for MobileRunClient. Records what the command asked for."""

    def __init__(
        self,
        devices: Optional[list[Device]] = None,
        *,
        healthy: bool = True,
        capabilities: Optional[dict] = None,
        summary: Optional[dict] = None,
        resolve_error: Optional[Exception] = None,
        list_error: Optional[Exception] = None,
        fleet_error: Optional[Exception] = None,
    ) -> None:
        self._devices = devices if devices is not None else [_device("dev-1")]
        self._healthy = healthy
        self._caps = capabilities if capabilities is not None else {"recording": True, "esim": False}
        self._summary = summary if summary is not None else {"device_slot": 1}
        self._resolve_error = resolve_error
        self._list_error = list_error
        # Fails only the whole-fleet report (the states=None listing and the type
        # totals), leaving the ready-only listing that `doctor` actually gates on intact.
        self._fleet_error = fleet_error
        self.calls: list[tuple[str, Any]] = []

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def list_devices(self, *, platform=None, states=("ready",), **_):
        self.calls.append(("list_devices", {"platform": platform, "states": states}))
        if self._list_error:
            raise self._list_error
        if states is None and self._fleet_error:
            raise self._fleet_error
        found = self._devices if states is None else [d for d in self._devices if d.state == "ready"]
        return [d for d in found if not platform or d.platform == platform]

    async def health_probe(self, device_id: str) -> bool:
        self.calls.append(("health_probe", device_id))
        return self._healthy

    async def fleet_summary(self) -> dict:
        if self._fleet_error:
            raise self._fleet_error
        return self._summary

    async def capabilities(self, device_id: str) -> dict:
        return self._caps

    async def resolve_device(self, *, platform=None, device_id=None, require_healthy=True):
        self.calls.append(("resolve_device", {"platform": platform, "device_id": device_id}))
        if self._resolve_error:
            raise self._resolve_error
        return self._devices[0]

    async def screenshot(self, device_id: str) -> bytes:
        return b"png-" + str(len(self.calls)).encode()

    async def global_action(self, device_id: str, action: int):
        self.calls.append(("global_action", action))
        return None


@pytest.fixture
def run(monkeypatch):
    """Invoke a command with a patched client and key."""

    def invoke(
        args: list[str],
        client: Optional[FakeClient] = None,
        key: str = "dr_sk_test",
        agent: Optional[tuple] = ("a2a-default", None),
    ):
        fake = client or FakeClient()
        monkeypatch.setattr(cli_mod, "_client", lambda: fake)
        monkeypatch.setattr(cli_mod, "resolve_api_key", lambda *a, **k: key, raising=False)
        monkeypatch.setenv("MOBILERUN_API_KEY", key)
        # Found, unless a test says otherwise: the real check reads the framework's store,
        # which here would be the developer's own ~/.local/state. It has its own tests below.
        if agent is not None:
            monkeypatch.setattr(cli_mod, "_check_default_agent", lambda: agent)
        result = CliRunner().invoke(cli_mod.mobilerun, args, catch_exceptions=False)
        return result, fake

    return invoke


# ------------------------------------------------------------------- devices


def test_devices_lists_ready_only_by_default(run):
    result, fake = run(["devices"], FakeClient([_device("a"), _device("b", state="maintenance")]))
    assert result.exit_code == 0
    assert "a" in result.stdout and "maintenance" not in result.stdout
    assert ("list_devices", {"platform": None, "states": ("ready",)}) in fake.calls


def test_devices_all_states_asks_for_the_whole_fleet(run):
    """--all-states must send states=None, which the client expands to every state.

    Passing nothing would let the server default to ready and hide the very devices the
    flag exists to reveal.
    """
    result, fake = run(["devices", "--all-states"], FakeClient([_device("a"), _device("b", state="maintenance")]))
    assert result.exit_code == 0
    assert "maintenance" in result.stdout
    assert ("list_devices", {"platform": None, "states": None}) in fake.calls


def test_devices_probe_adds_the_only_column_that_exercises_the_device(run):
    fake = FakeClient([_device("a")], healthy=False)
    result, _ = run(["devices", "--probe"], fake)
    assert "DRIVABLE" in result.stdout and "NO" in result.stdout
    assert ("health_probe", "a") in fake.calls


def test_devices_json_is_machine_readable(run):
    result, _ = run(["devices", "--json"], FakeClient([_device("a")]))
    payload = json.loads(result.stdout)
    assert [d["id"] for d in payload] == ["a"]
    assert payload[0]["state"] == "ready"


def test_devices_says_where_to_get_one_when_the_account_is_empty(run):
    result, _ = run(["devices"], FakeClient([]))
    assert result.exit_code == 0
    assert "mobilerun.ai/pricing" in result.stdout


def test_devices_reports_an_api_failure_as_an_error_not_an_empty_list(run):
    result, _ = run(["devices"], FakeClient(list_error=MobileRunError("503 from the device plane")))
    assert result.exit_code == 1
    assert "503" in result.stderr


# -------------------------------------------------------------------- doctor


def test_doctor_exits_zero_when_a_deploy_would_work(run):
    result, _ = run(["doctor"], FakeClient([_device("a", platform="android")], capabilities={"recording": True}))
    assert result.exit_code == 0
    for step in ("1. API key", "2. Devices", "3. Target device", "4. Health probe"):
        assert step in result.stdout


def test_doctor_never_prints_the_key(run):
    """Only its length. A masked credential still invites a screenshot."""
    result, _ = run(["doctor"], FakeClient([_device("a", platform="android")]), key="dr_sk_supersecret")
    assert "dr_sk_supersecret" not in result.stdout + result.stderr
    assert "17 characters" in result.stdout


def test_doctor_is_a_gate_when_there_is_no_ready_device(run):
    """The property the README sells: non-zero exit when a deploy would fail."""
    result, _ = run(["doctor"], FakeClient([_device("a", state="maintenance")]))
    assert result.exit_code == 1
    assert "no READY device" in result.stderr


def test_doctor_is_a_gate_when_the_device_cannot_be_driven(run):
    """state=ready plus a failing screenshot is the whole reason this command exists."""
    result, _ = run(["doctor"], FakeClient([_device("a", platform="android")], healthy=False))
    assert result.exit_code == 1
    assert "cannot be driven" in result.stderr


def test_doctor_reports_devices_hidden_by_the_ready_only_listing(run):
    """Regression: this read summary["byState"], which the API does not return.

    /devices/count answers a flat {device_type: count} map, so the state breakdown has to
    come from the whole-fleet listing. The effect of getting it wrong was that the section
    meant to stop a maintenance phone looking non-existent printed nothing at all.
    """
    fake = FakeClient(
        [_device("a", platform="android"), _device("b", state="maintenance"), _device("c", state="terminated")],
        summary={"android_cloud_phone": 1, "device_slot": 10},
    )
    result, _ = run(["doctor"], fake)
    assert result.exit_code == 0
    assert "not ready (hidden from the list above)" in result.stdout
    assert "maintenance" in result.stdout and "terminated" in result.stdout
    assert "android_cloud_phone" in result.stdout


def test_doctor_still_passes_when_only_the_fleet_report_fails(run):
    """The gate answers one question: is there a device to deploy onto. The ready-only
    listing answers it, and the whole-fleet colour commentary that follows is best
    effort -- a 500 on it must not turn a deployable account into exit 1."""
    fake = FakeClient(
        [_device("a", platform="android")],
        fleet_error=MobileRunError("GET /devices -> HTTP 503: upstream"),
    )
    result, _ = run(["doctor"], fake)
    assert result.exit_code == 0
    assert "could not read the rest of the fleet" in result.stdout


def test_doctor_is_a_gate_when_the_bundle_has_no_agent(run):
    """`agent-env run mobilerun` deploys the default agent and the framework ships none; on
    a clean install the run only found out at deploy_agent, after the phone and the
    gateway were up. doctor finds out first, and says how to fix it."""
    missing = ("a2a-default", "no A2A agent 'a2a-default' in the store. Register one ...")
    result, _ = run(["doctor"], FakeClient([_device("a", platform="android")]), agent=missing)
    assert result.exit_code == 1
    assert "no A2A agent 'a2a-default'" in result.stderr


def test_doctor_skip_agent_is_a_deploy_only_gate(run):
    """Someone who deploys the env and drives its MCP server directly needs no agent."""
    missing = ("a2a-default", "no A2A agent 'a2a-default' in the store.")
    result, _ = run(["doctor", "--skip-agent"], FakeClient([_device("a", platform="android")]), agent=missing)
    assert result.exit_code == 0
    assert "skipped (--skip-agent)" in result.stdout


def test_doctor_reports_the_agent_it_found(run):
    result, _ = run(["doctor"], FakeClient([_device("a", platform="android")]), agent=("my-agent", None))
    assert result.exit_code == 0
    assert "'my-agent' is registered" in result.stdout


@pytest.fixture
def isolated_framework(monkeypatch, tmp_path):
    """agent-env reading only tmp_path: its own state dir, no config file above it."""
    config = pytest.importorskip("agent_env.config")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    config.reset_config()
    yield tmp_path
    config.reset_config()


def test_the_real_check_finds_no_agent_on_a_clean_install(isolated_framework):
    """Against the framework itself, not a stand-in: a clean store has no 'a2a-default'."""
    agent_id, problem = cli_mod._check_default_agent()
    assert agent_id == "a2a-default"
    assert problem and "agent-env a2a-agent put --id a2a-default" in problem


def test_the_real_check_honours_a_configured_default(isolated_framework, monkeypatch):
    """The reason the bundle's task names no agent: a user's [agents] default_a2a_agent_id
    has to win, exactly as it does in deploy_agent."""
    from agent_env.config import reset_config

    config_file = isolated_framework / "config.toml"
    config_file.write_text('[agents]\ndefault_a2a_agent_id = "my-phone-agent"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config_file))
    reset_config()

    agent_id, problem = cli_mod._check_default_agent()
    assert agent_id == "my-phone-agent"
    assert "'my-phone-agent'" in problem


def test_doctor_flags_a_device_that_cannot_record(run):
    result, _ = run(
        ["doctor"],
        FakeClient([_device("a", platform="android")], capabilities={"recording": False}),
    )
    assert "no recording capability" in result.stdout


def test_doctor_surfaces_an_unresolvable_device(run):
    result, _ = run(
        ["doctor", "--device", "nope"],
        FakeClient(resolve_error=DeviceUnavailable("device nope does not exist")),
    )
    assert result.exit_code == 1
    assert "does not exist" in result.stderr


# --------------------------------------------------------------- probe-global


def test_probe_global_refuses_without_yes(run):
    """It drives the phone, so consent is not a default."""
    result, fake = run(["probe-global", "--device", "a"])
    assert result.exit_code == 1
    assert "--yes" in result.stderr
    assert not any(c[0] == "global_action" for c in fake.calls)


def test_probe_global_rejects_non_integer_codes(run):
    result, _ = run(["probe-global", "--device", "a", "--codes", "1,home", "--yes"])
    assert result.exit_code == 1
    assert "comma-separated integers" in result.stderr


def test_probe_global_reports_whether_each_code_moved_the_screen(run):
    """The point of the command: the API documents no enum, so observe rather than guess."""
    result, fake = run(["probe-global", "--device", "a", "--codes", "1,2", "--yes"])
    assert result.exit_code == 0
    assert [c[1] for c in fake.calls if c[0] == "global_action"] == [1, 2]
    assert result.stdout.count("action=") == 2
    assert "screen changed" in result.stdout


# --------------------------------------------------------------------- setup


def test_setup_skip_build_without_an_artifact_fails_clearly(run, monkeypatch):
    """--skip-build is the one setup path that needs no docker, so it is the one worth
    covering here; the build path shells out and is exercised by hand.

    Unlike the rest of this file, `setup` really does need the framework -- it registers
    an env -- so this is the one CLI test that cannot run without it.
    """
    pytest.importorskip("agent_env", reason="agentenv-framework is not installed")
    from agent_env.artifact import Artifact

    monkeypatch.setattr(Artifact, "get", staticmethod(lambda _id: (_ for _ in ()).throw(KeyError("absent"))))

    result, _ = run(["setup", "--skip-build"])
    assert result.exit_code == 1
    assert "--skip-build needs an existing artifact" in result.stderr
