"""``agent-env mobilerun ...`` — setup and diagnostics for the MobileRun plugin.

Contributed to the agent-env CLI through the ``agent_env.cli_plugins`` entry point, so it
appears as a top-level command group once this package is installed.

The diagnostic commands exist because every failure seen so far is explainable from four
facts: is there a key, are there devices, what does each device claim it can do, and does
it actually answer a screenshot. ``doctor`` prints exactly those four, in that order.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

from agentenv_mobilerun.client import (
    GLOBAL_ACTIONS,
    DeviceUnavailable,
    MobileRunClient,
    MobileRunError,
)

#: Repo-relative paths to the bundled server image source. Resolved off this module so
#: `setup` works from any working directory, including a site-packages install.
_PACKAGE_ROOT = Path(__file__).parent
_SRC_ROOT = _PACKAGE_ROOT.parent
DOCKERFILE = _PACKAGE_ROOT / "server" / "Dockerfile"


@click.group(name="mobilerun")
def mobilerun() -> None:
    """MobileRun cloud-phone environment commands."""


def _client() -> MobileRunClient:
    from agentenv_mobilerun.api_key import resolve_api_key

    return MobileRunClient(resolve_api_key())


def _fail(message: str) -> None:
    click.echo(f"Error: {message}", err=True)
    sys.exit(1)


# --------------------------------------------------------------------- devices


@mobilerun.command(name="devices")
@click.option("--platform", type=click.Choice(["android", "ios"]), help="Only show one platform.")
@click.option("--all-states", is_flag=True, help="Include devices that are not ready (provisioning, maintenance, stopped, terminated).")
@click.option("--probe/--no-probe", default=False, help="Also take a screenshot of each device to prove it is drivable.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def devices(platform: Optional[str], all_states: bool, probe: bool, as_json: bool) -> None:
    """List the account's devices.

    Shows only ready devices by default, mirroring the API. Pass ``--all-states`` to see
    the whole fleet — an account can hold a phone in ``maintenance`` that the default
    listing hides entirely, which makes for a confusing "where is my device?".

    ⚠️ ``state`` is not a health signal — a phone that cannot be driven at all still
    reports ``ready``. Pass ``--probe`` when you need the truth; it is the only column
    here that actually exercises the device.
    """

    async def run() -> list[dict]:
        async with _client() as client:
            found = await client.list_devices(platform=platform, states=None if all_states else ("ready",))
            rows = []
            for device in found:
                row = {
                    "id": device.id,
                    "platform": device.platform,
                    "state": device.state,
                    "name": device.name,
                    "idle": device.idle,
                    "type": device.raw.get("type"),
                }
                if probe:
                    row["drivable"] = await client.health_probe(device.id)
                rows.append(row)
            return rows

    try:
        rows = asyncio.run(run())
    except (MobileRunError, DeviceUnavailable) as exc:
        _fail(str(exc))
        return

    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo("No devices on this account. Provision one at https://mobilerun.ai/pricing first.")
        return
    header = f"{'ID':38} {'PLATFORM':9} {'STATE':10} {'IDLE':5}" + ("  DRIVABLE" if probe else "")
    click.echo(header)
    for row in rows:
        line = f"{row['id']:38} {row['platform']:9} {row['state']:10} {str(row['idle']):5}"
        if probe:
            line += f"  {'yes' if row['drivable'] else 'NO'}"
        click.echo(line)


# ---------------------------------------------------------------------- doctor


@mobilerun.command(name="doctor")
@click.option("--device", "device_id", help="Check this device instead of the first idle one.")
@click.option("--platform", type=click.Choice(["android", "ios"]), default="android", show_default=True)
@click.option(
    "--skip-agent",
    is_flag=True,
    help="Don't check for an agent: you deploy the env and drive its MCP server yourself.",
)
def doctor(device_id: Optional[str], platform: str, skip_agent: bool) -> None:
    """Check the four things that explain every MobileRun failure, and the agent the bundle needs.

    Exits non-zero if `agent-env run mobilerun` would not be able to run, so it is usable
    as a gate in a script or a CI job. With --skip-agent, if the env could not be deployed.
    """
    from agentenv_mobilerun.api_key import resolve_api_key

    click.echo("1. API key")
    try:
        key = resolve_api_key()
    except MobileRunError as exc:
        _fail(str(exc))
        return
    # Never print a credential, even partially-masked ones invite screenshots.
    click.echo(f"   found ({len(key)} characters)")

    async def run() -> int:
        async with _client() as client:
            click.echo("2. Devices")
            try:
                found = await client.list_devices()
            except MobileRunError as exc:
                click.echo(f"   FAILED: {exc}", err=True)
                return 1
            for device in found:
                click.echo(f"   {device.id}  {device.platform:8} state={device.state} idle={device.idle}")
            # The listing above is ready-only, so report the rest of the fleet rather than
            # letting a device in maintenance look like a device that does not exist.
            #
            # From the whole-fleet listing, not from /devices/count: that route answers a
            # flat {device_type: count} map ({"android_cloud_phone": 1, "device_slot": 10}
            # on a real account), with no breakdown by state. This previously read
            # summary["byState"], which does not exist, so the hidden devices this section
            # is FOR were never printed.
            #
            # Best effort, and that is the point: the ready-only listing above has already
            # answered the question `doctor` exists to answer -- is there a device to
            # deploy onto. These two calls only colour that in, so a timeout or a 500 on
            # either must not turn a passing gate into exit 1.
            try:
                hidden = {d.id: d.state for d in await client.list_devices(states=None) if d.state != "ready"}
                if hidden:
                    click.echo(f"   not ready (hidden from the list above): {hidden}")
                totals = await client.fleet_summary()
                if totals:
                    click.echo(f"   fleet by type: {totals}")
            except MobileRunError as exc:
                click.echo(f"   (could not read the rest of the fleet: {exc})")
            if not found:
                click.echo(
                    "   no READY device on this account. Provision one at "
                    "https://mobilerun.ai/pricing, or wait for one of the above to become ready.",
                    err=True,
                )
                return 1

            click.echo("3. Target device")
            try:
                target = await client.resolve_device(
                    platform=None if device_id else platform,
                    device_id=device_id,
                    require_healthy=False,
                )
            except DeviceUnavailable as exc:
                click.echo(f"   FAILED: {exc}", err=True)
                return 1
            click.echo(f"   {target.id} ({target.platform})")
            capabilities = await client.capabilities(target.id)
            enabled = sorted(name for name, value in capabilities.items() if value)
            disabled = sorted(name for name, value in capabilities.items() if not value)
            click.echo(f"   supported ({len(enabled)}): {', '.join(enabled) or '-'}")
            click.echo(f"   unsupported ({len(disabled)}): {', '.join(disabled) or '-'}")
            if capabilities.get("recording") is False:
                click.echo("   note: no recording capability — env record=True will be skipped")

            click.echo("4. Health probe (screenshot)")
            if not await client.health_probe(target.id):
                click.echo(
                    "   FAILED — this device reports a healthy state but cannot be driven. Reboot it "
                    "from the MobileRun dashboard; if that does not clear it, it is a MobileRun "
                    "support item, not a configuration problem.",
                    err=True,
                )
                return 1
            click.echo("   ok — the device returned a screenshot")

            click.echo("5. Default agent (the one `agent-env run mobilerun` deploys)")
            if skip_agent:
                click.echo("   skipped (--skip-agent)")
                return 0
            agent_id, problem = _check_default_agent()
            if problem:
                click.echo(f"   FAILED — {problem}", err=True)
                return 1
            if agent_id is None:
                click.echo("   skipped — agentenv-framework is not installed, so nothing could run the bundle")
                return 0
            click.echo(f"   ok — {agent_id!r} is registered")
            return 0

    try:
        sys.exit(asyncio.run(run()))
    except MobileRunError as exc:
        _fail(str(exc))


def _check_default_agent() -> tuple[Optional[str], Optional[str]]:
    """``(agent id, problem)`` for the agent the bundle's run will deploy.

    The bundle's ``deploy_agent`` step names no agent, so it deploys the configured
    default -- ``[agents] default_a2a_agent_id``, else ``a2a-default`` -- which means a user
    who already has one keeps it. The cost is that the framework's preflight checks only an
    agent a task NAMES, and the framework ships no agent: on a clean install the run
    finds out only at ``deploy_agent``, after the phone has been resolved and the gateway
    stood up. This resolves the id the way ``deploy_agent`` does and reads it, so the gap
    shows up here instead.

    ``(None, None)`` without the framework, where nothing could run the bundle anyway.
    """
    try:
        from agent_env.a2a_agent import A2AAgent
        from agent_env.config import get_config
        from agent_env.store.base import NotFoundError
    except ImportError:
        return None, None
    try:
        agent_id = get_config().get_default_a2a_agent_id()
    except Exception as exc:  # a broken [agents] table: deploy_agent would raise the same
        return None, f"could not resolve the default agent: {exc}"
    try:
        A2AAgent.get(agent_id)
    except NotFoundError:
        return agent_id, (
            f"no A2A agent {agent_id!r} in the store. The bundle deploys the default agent, and "
            f"agent-env ships none. Register one under that id with "
            f"`agent-env a2a-agent put --id {agent_id} --dockerfile <your agent's Dockerfile>`, "
            f"or point [agents] default_a2a_agent_id in .agentenv/config.toml at one you already "
            f"have. Pass --skip-agent if you only deploy the env."
        )
    except Exception as exc:
        return agent_id, f"could not read agent {agent_id!r}: {exc}"
    return agent_id, None


# ----------------------------------------------------------------------- setup


@mobilerun.command(name="setup")
@click.option("--id", "env_id", default="mobilerun", show_default=True, help="Env id to register.")
@click.option("--platform", type=click.Choice(["android", "ios"]), default="android", show_default=True)
@click.option("--device", "device_id", help="Pin this device. Omit to resolve an idle one at each deploy.")
@click.option("--record/--no-record", default=False, show_default=True, help="Record video+trajectory for each deploy.")
@click.option("--build-platform", default="linux/amd64", show_default=True, help="Docker --platform for the image build.")
@click.option("--skip-build", is_flag=True, help="Reuse the already-registered image artifact.")
def setup(
    env_id: str,
    platform: str,
    device_id: Optional[str],
    record: bool,
    build_platform: str,
    skip_build: bool,
) -> None:
    """Build the MCP server image and register the env.

    The image is built locally and registered through agent-env's own image store, so
    there is no public registry to pull from and nothing to ``docker login`` to.
    """
    from agent_env.artifact import Artifact, DockerImageArtifact

    from agentenv_mobilerun.env import MobileRunEnv

    artifact_id = f"mcp-server-{env_id}"
    if skip_build:
        try:
            artifact = Artifact.get(artifact_id)
        except Exception as exc:
            _fail(f"--skip-build needs an existing artifact {artifact_id!r}, but it could not be read: {exc}")
            return
        click.echo(f"Reusing artifact: id={artifact.id} version={artifact.version}")
    else:
        image_tag = f"mcp-server-{env_id}"
        click.echo(f"Building the MobileRun MCP server image ({build_platform})...")
        result = subprocess.run(
            [
                "docker", "build", "--platform", build_platform,
                "-f", str(DOCKERFILE), "-t", image_tag, str(_SRC_ROOT),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # The build log, not just the last line: a failed pip resolve is only
            # diagnosable from the middle of the output.
            click.echo(result.stdout[-4000:], err=True)
            _fail(f"docker build failed:\n{result.stderr[-4000:]}")
            return
        click.echo("Build ok. Registering the image artifact...")
        artifact = DockerImageArtifact.put(
            id=artifact_id,
            description="MobileRun MCP server (agentenv-mobilerun)",
            image_name=image_tag,
            build_context_path=str(_SRC_ROOT),
            dockerfile_path=str(DOCKERFILE),
        )
        click.echo(f"Created artifact: id={artifact.id} version={artifact.version}")

    env = MobileRunEnv.put(
        id=env_id,
        docker_image_artifact=artifact,
        platform=platform,
        device_id=device_id,
        record=record,
    )
    click.echo(f"Registered MobileRunEnv: id={env.id} version={env.version} platform={env.platform}")
    pin = f" (pinned to {device_id})" if device_id else " (resolves an idle device per deploy)"
    click.echo(f"Device selection:{pin}")
    click.echo(f"\nNext: agent-env env deploy --id {env_id}")


# ----------------------------------------------------------------- probe-global


@mobilerun.command(name="probe-global")
@click.option("--device", "device_id", required=True, help="Device to probe.")
@click.option("--codes", default="1,2,3", show_default=True, help="Comma-separated action codes to try.")
@click.option("--yes", is_flag=True, help="Required: this drives the phone.")
def probe_global(device_id: str, codes: str, yes: bool) -> None:
    """Discover what the global-action codes actually do on a device.

    The API takes a bare integer and documents no enum. The names this plugin ships are
    the Android accessibility constants; on iOS the mapping is undocumented. This command
    sends each code and captures a screenshot before and after so you can see for
    yourself, rather than trusting a guess.

    ⚠️ This **acts on the phone** — it navigates, and codes above 3 can open system
    surfaces such as the notification shade or a power dialog. Hence ``--yes``.
    """
    if not yes:
        _fail("probe-global drives the phone; re-run with --yes once you are sure it is free")
        return
    try:
        wanted = [int(part) for part in codes.split(",") if part.strip()]
    except ValueError:
        _fail(f"--codes must be comma-separated integers, got {codes!r}")
        return
    known = {code: name for name, code in GLOBAL_ACTIONS.items()}

    async def run() -> None:
        async with _client() as client:
            for code in wanted:
                label = known.get(code, "unnamed")
                before = await client.screenshot(device_id)
                try:
                    await client.global_action(device_id, code)
                    outcome = "accepted"
                except MobileRunError as exc:
                    outcome = f"rejected ({exc})"
                after = await client.screenshot(device_id)
                changed = "screen changed" if before != after else "screen UNCHANGED"
                click.echo(f"action={code} ({label}): {outcome}; {changed}")

    try:
        asyncio.run(run())
    except (MobileRunError, DeviceUnavailable) as exc:
        _fail(str(exc))
