"""``MobileRunEnv`` — a MobileRun cloud phone as an agent-env environment.

Registered on the ``agent_env.envs`` entry point by installing the package, so a stored
env document with ``type = "mobilerun"`` deserialises here with no fork of agent-env and
no line in anyone's config.toml.

**Shape.** One MCP container in the normal gateway, talking straight to
``api.mobilerun.ai``. There is no second tier and no custom sandbox provider, because a
cloud phone is not a sandbox — it is a remote resource the container dials over HTTPS.
(A USB-attached phone is different: it is bolted to a host machine, which is what forces
a long-lived per-phone server and a device pool. None of that applies here.)

**Exclusivity.** ⚠️ MobileRun's device plane has no lease or reservation primitive, and
neither does this env. Two deploys against one account can resolve the same phone and
both drive it. If you run more than one task at a time, pin distinct devices with
``device_id`` (or run one job per device upstream). This is stated rather than papered
over: a lease that does not actually serialise is worse than no lease.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, ClassVar, Optional

from agentenv_mobilerun.capabilities import encode_capabilities  # noqa: F401  (re-exported)
from agent_env.artifact import Artifact
from agent_env.env.env import Env
from agent_env.env.gateway import GatewayMode
from agent_env.env.store import register_env_instance

from agentenv_mobilerun.client import DeviceUnavailable, MobileRunClient, MobileRunError

if TYPE_CHECKING:
    from agent_env.attribution import Attribution
    from agent_env.artifact import DockerImageArtifact
    from agent_env.env.env import DeployedEnv, DeployedGatewayEnv

logger = logging.getLogger(__name__)

#: Name looked up in the environment and then in the configured secret store. Using one
#: name for both means ``MOBILERUN_API_KEY=...`` works out of the box with the default
#: local secret backend, while an AWS/Vault backend resolves the same name.
#: The container port the bundled MCP server listens on (and agent-env's convention).
MCP_PORT = 18765

SUPPORTED_PLATFORMS = ("android", "ios")

# Re-exported: these moved to a module that does not import agent-env, so the CLI and
# its tests can resolve a key without the framework installed.
from agentenv_mobilerun.api_key import API_KEY_NAME, resolve_api_key  # noqa: E402,F401


class MobileRunEnv(Env):
    """A pinned or auto-resolved MobileRun phone, exposed to an agent over MCP."""

    type: ClassVar[str] = "mobilerun"
    description = (
        "Computer-use environment driving a real MobileRun cloud phone (Android or iOS) "
        "over the MobileRun device API"
    )

    def __init__(
        self,
        id: str,
        version: Optional[int],
        docker_image_artifact: "DockerImageArtifact",
        *,
        platform: str = "android",
        device_id: Optional[str] = None,
        environment_name: str = "mobilerun",
        record: bool = False,
        api_key_secret_name: str = API_KEY_NAME,
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(id, version, metadata=metadata)
        platform_key = (platform or "").strip().lower()
        if platform_key not in SUPPORTED_PLATFORMS:
            raise ValueError(f"platform must be one of {SUPPORTED_PLATFORMS}, got {platform!r}")
        self.docker_image_artifact = docker_image_artifact
        self.platform = platform_key
        #: When set, this exact device is used. When None, deploy() resolves an idle,
        #: healthy device of the right platform — convenient for one-device accounts and
        #: unsafe for concurrent runs (see the module docstring).
        self.device_id = device_id
        self.environment_name = environment_name
        #: Start a server-side video+trajectory recording for the life of the deploy.
        #: Off by default: a recording captures whatever is on screen, which on a
        #: bring-your-own phone includes notifications and personal accounts.
        self.record = bool(record)
        self.api_key_secret_name = api_key_secret_name
        # Built on first use, not in __init__: deserialising an env document must not
        # require a working compute config. `from_dict` runs anywhere a document is read,
        # including in a process that will never deploy anything.
        self._gateway_provider_instance = None
        self._sandbox = None
        self._gateway_url: Optional[str] = None
        self._resolved_device_id: Optional[str] = None
        self._recording_id: Optional[str] = None

    @property
    def _gateway_provider(self):
        # EnvironmentGatewayProvider, the class `GatewayProvider` was an alias of until
        # agent-env#1259 deleted the alias -- on 2026-09-30, the day of the first public
        # release, so no published framework has ever had the old name. Every deploy
        # raised ImportError while the suite stayed green, because the deploy tests inject
        # a fake provider and never reach this import; test_env.py now builds the real one.
        if self._gateway_provider_instance is None:
            from agent_env.providers import EnvironmentGatewayProvider

            self._gateway_provider_instance = EnvironmentGatewayProvider()
        return self._gateway_provider_instance

    # ------------------------------------------------------------ serialisation

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["docker_image_artifact"] = {
            "id": self.docker_image_artifact.id,
            "version": self.docker_image_artifact.version,
            "type": self.docker_image_artifact.type,
        }
        base["platform"] = self.platform
        base["environment_name"] = self.environment_name
        base["record"] = self.record
        base["api_key_secret_name"] = self.api_key_secret_name
        # Omitted when unset so an auto-resolving env document stays free of a stale
        # device id, which would otherwise pin every future deploy to one phone.
        if self.device_id:
            base["device_id"] = self.device_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "MobileRunEnv":
        ref = data["docker_image_artifact"]
        return cls(
            id=data["id"],
            version=data.get("version"),
            docker_image_artifact=Artifact.get(ref["id"], version=ref["version"]),
            platform=data.get("platform", "android"),
            device_id=data.get("device_id"),
            environment_name=data.get("environment_name", "mobilerun"),
            record=bool(data.get("record", False)),
            api_key_secret_name=data.get("api_key_secret_name", API_KEY_NAME),
            metadata=data.get("metadata", {}),
        )

    # ------------------------------------------------------------------ deploy

    async def deploy(
        self,
        ttl_seconds: int = 10800,
        gateway_mode: GatewayMode = GatewayMode.PERFORMANCE,
        sandbox_type: str | None = None,
        cpu: float | None = None,
        memory_mb: int | None = None,
        disk_size_gb: float = 10,
        attribution: "Attribution | None" = None,
        priority: Optional[int] = None,
        env_state_type: str | None = None,
        **_ignored: object,
    ) -> "DeployedEnv":
        """Resolve a device, then deploy the MCP server wired to it.

        The device is resolved and health-probed **before** any compute is provisioned, so
        an unusable phone costs nothing but one API call. ``attribution`` is the mapping
        agent-env now takes; the flat ``product``/``customer``/``team``/``project_id``
        kwargs it replaced are absorbed by ``**_ignored`` rather than silently mis-billed. ``env_state_type`` is accepted and
        ignored: this env keeps no state here -- the phone holds the state, and nothing in
        this plugin writes to a store.

        Gateway-fronted through :meth:`EnvironmentGatewayProvider.create_gateway`, not its
        ``deploy()``: ``deploy()`` derives the MCP server's config from the env and passes
        it no environment variables, and this server needs the API key, device id and
        capability map. ``create_gateway`` is the entry point agent-env documents for a
        caller that builds its own record, so the record below mirrors the one ``deploy()``
        builds, field for field.

        ⚠️ The gateway still builds its local store for this deploy, which this env never
        reads. ``EnvironmentServerProvider`` (agent-env#1206) has no gateway and no store,
        and the server now serves its own env card, which is what that provider requires
        -- so moving onto it is possible, and is the follow-up. It is not done here because
        this change is the minimum that makes a deploy work at all.
        """
        # DeployedGatewayEnv, not DeployedEnv: agent-env split the record by topology
        # (DeployedSandboxEnv / DeployedGatewayEnv) and the base no longer carries the
        # gateway's URLs. This env is gateway-fronted, so it is the gateway variant.
        from agent_env.env.env import DeployedGatewayEnv
        from agent_env.env.gateway.constants import WELL_KNOWN_PATH
        from agent_env.providers import (
            MCPServerConfig,
            build_sandbox_provider,
            get_env_sandbox_provider,
        )
        # agent-env removed the flat product/customer/team/project_id kwargs and
        # resolve_attribution with them (a breaking refactor), so attribution arrives as
        # the mapping it is now. Copied because create_gateway may add pipeline_step/run_id.
        attribution = dict(attribution or {})
        api_key = resolve_api_key(self.api_key_secret_name)

        async with MobileRunClient(api_key, **self._client_kwargs()) as client:
            device = await client.resolve_device(platform=self.platform, device_id=self.device_id)
            capabilities = await client.capabilities(device.id)
            logger.info(
                "MobileRun device %s (%s, state=%s) passed its health probe; %d capabilities advertised",
                device.id,
                device.platform,
                device.state,
                sum(1 for v in capabilities.values() if v),
            )
            if self.record:
                self._recording_id = await self._maybe_start_recording(client, device.id, capabilities)

        self._resolved_device_id = device.id

        try:
            sandbox_provider = (
                build_sandbox_provider(sandbox_type) if sandbox_type else get_env_sandbox_provider()
            )
            mcp_server = MCPServerConfig(
                image=self.docker_image_artifact.image_name,
                environment_name=self.environment_name,
                extra_env_vars=self._mcp_env(api_key, device.id, capabilities),
            )
            result = await self._gateway_provider.create_gateway(
                sandbox_provider=sandbox_provider,
                mcp_servers=[mcp_server],
                mcp_server_images=[self.docker_image_artifact],
                gateway_mode=gateway_mode,
                ttl_seconds=ttl_seconds,
                disk_size_gb=disk_size_gb,
                cpu=cpu,
                memory_mb=memory_mb,
                attribution=attribution,
                priority=priority,
                env_id=self.id,
                # No state_instance: it is a pre-acquired EXTERNAL store handle, and this
                # env has none. The gateway therefore builds its own local store, which
                # this env does not use -- see the note in the docstring.
            )
            self._sandbox = self._gateway_provider.sandbox
            self._gateway_url = result.gateway_url

            metadata = {
                "mobilerun_device_id": device.id,
                "mobilerun_platform": device.platform,
            }
            if self._recording_id:
                metadata["mobilerun_recording_id"] = self._recording_id
            # The env card the gateway composed, and its address. They come as a pair or not
            # at all: with both, the record derives mcp_url and mcp_server_name from the
            # card, which is where the agent learns the server's name. mcp_url is still set,
            # so a deploy whose card read failed keeps a working MCP address.
            card = result.environment_card
            card_fields = (
                {
                    "environment_card_url": f"{result.gateway_url}{WELL_KNOWN_PATH}",
                    "environment_card": card,
                    "environment_card_read_at_utc": result.environment_card_read_at_utc,
                }
                if card
                else {}
            )
            deployed_env = register_env_instance(
                DeployedGatewayEnv(
                    env_id=self.id,
                    env_version=self.version,
                    env_provider_type=self._gateway_provider.type,
                    gateway_url=result.gateway_url,
                    mcp_url=result.mcp_url,
                    db_web_url=result.db_web_url,
                    db_mcp_url=result.db_mcp_url,
                    sandbox_id=self._sandbox.sandbox_id,
                    sandbox_type=self._sandbox.type,
                    # Every sandbox the deploy created besides the gateway's. Run-end
                    # teardown terminates exactly what the record names, so on a container
                    # backend -- where the MCP server and the store are sandboxes of their
                    # own -- leaving this out leaks them until their TTL. Private, but it is
                    # what deploy() itself records; test_env.py fails if it is renamed.
                    sandbox_ids=self._gateway_provider._sandbox_ids(),
                    env_state_instance_ids=result.env_state_instance_ids,
                    gateway_mode=gateway_mode.value,
                    metadata=metadata,
                    **card_fields,
                ),
                ttl_seconds,
            )
            self._instance_id = deployed_env.instance_id
            return deployed_env
        except BaseException:
            await self.close()
            raise

    def _client_kwargs(self) -> dict:
        base_url = (os.environ.get("MOBILERUN_BASE_URL") or "").strip()
        return {"base_url": base_url} if base_url else {}

    def _mcp_env(self, api_key: str, device_id: str, capabilities: dict[str, bool]) -> dict[str, str]:
        """Environment for the MCP container.

        The key travels through the gateway's compose ``environment:`` block, so it never
        lands on a ``docker run`` command line or in the host process table.

        The capability map is passed down rather than re-discovered per container: the env
        has just read it, and doing it here means the server registers exactly the tools the
        phone supports without an extra API round trip at boot.

        It is sent as the **whole map**, ``false`` entries included. Sending only the
        supported names would leave the server unable to tell "unsupported" from "a name
        this plugin has not heard of" — and since it must assume the latter to avoid
        dropping working tools as MobileRun adds capabilities, the gate would be inert.

        ⚠️ Encoded as ``name=true,name2=false``, **not** as JSON. The gateway renders these
        pairs into a docker-compose ``environment:`` list, and a YAML plain scalar may not
        contain ``": "`` — a JSON object therefore makes ``docker compose up`` die with
        "mapping values are not allowed in this context", which surfaces as a deploy
        failure nowhere near this function. No colons, no spaces, no braces.
        """
        env = {
            "MOBILERUN_API_KEY": api_key,
            "MOBILERUN_DEVICE_ID": device_id,
            "MCP_PORT": str(MCP_PORT),
        }
        if capabilities:
            env["MOBILERUN_CAPABILITIES"] = encode_capabilities(capabilities)
        base_url = (os.environ.get("MOBILERUN_BASE_URL") or "").strip()
        if base_url:
            env["MOBILERUN_BASE_URL"] = base_url
        return env

    async def _maybe_start_recording(
        self, client: MobileRunClient, device_id: str, capabilities: dict[str, bool]
    ) -> Optional[str]:
        """Start a recording if the device can, otherwise say so and carry on.

        A run must not die because the phone cannot record — the demonstration is still
        scoreable without video.
        """
        if capabilities.get("recording") is False:
            logger.warning("device %s does not advertise the recording capability; not recording", device_id)
            return None
        try:
            recording_id = await client.start_recording(device_id, name=f"agentenv-{self.id}")
        except MobileRunError as exc:
            logger.warning("could not start a recording on %s (%s); continuing without one", device_id, exc)
            return None
        logger.info("recording %s started on device %s", recording_id, device_id)
        return recording_id

    # ------------------------------------------------------------- reattachment

    @classmethod
    async def from_deployed_env(cls, deployed: "DeployedEnv") -> "MobileRunEnv":
        env = Env.get(deployed.env_id, deployed.env_version)
        if not isinstance(env, MobileRunEnv):
            raise TypeError(f"Expected MobileRunEnv, got {type(env).__name__}")
        env._gateway_url = deployed.gateway_url
        env._instance_id = deployed.instance_id
        metadata = deployed.metadata or {}
        env._resolved_device_id = metadata.get("mobilerun_device_id")
        env._recording_id = metadata.get("mobilerun_recording_id")
        if deployed.sandbox_id:
            from agent_env.providers import build_sandbox_provider, get_env_sandbox_provider

            provider = (
                build_sandbox_provider(deployed.sandbox_type)
                if deployed.sandbox_type
                else get_env_sandbox_provider()
            )
            # Only the env's own handle is restored. EnvironmentGatewayProvider.sandbox is a read-only
            # property, so assigning it raises AttributeError and takes the whole reattach
            # (and therefore teardown) down with it. close() terminates through
            # ``self._sandbox``, so the provider does not need the handle.
            env._sandbox = await provider.get_sandbox(deployed.sandbox_id)
        return env

    # ------------------------------------------------------------------- reset

    async def reset(self, deployed: "DeployedEnv") -> None:
        """Return the phone to a neutral screen between tasks.

        Deliberately **non-destructive**: a home-screen press, nothing more.

        ⚠️ MobileRun also offers ``POST /devices/{id}/reset``, which restores the device to
        a fresh state by clearing installed apps and user data. On a bring-your-own device
        that is somebody's personal phone, so it is never wired in here. It is reachable as
        :meth:`~agentenv_mobilerun.client.MobileRunClient.factory_reset` for an operator who
        means it.
        """
        device_id = (deployed.metadata or {}).get("mobilerun_device_id") or self._resolved_device_id
        if not device_id:
            logger.warning("no device id on the deployment; nothing to reset")
            return
        api_key = resolve_api_key(self.api_key_secret_name)
        async with MobileRunClient(api_key, **self._client_kwargs()) as client:
            try:
                await client.press_home(device_id)
            except MobileRunError as exc:
                # A reset failure should not fail the run that follows it; the next task's
                # first screenshot shows the truth either way.
                logger.warning("could not return device %s to the home screen: %s", device_id, exc)

    # ------------------------------------------------------------------- close

    async def close(self) -> None:
        """Tear down the gateway, then stop any recording this deploy started.

        Order matters. The gateway goes first so that nothing can still be driving the
        phone once recording stops — otherwise the last actions of a run are executed off
        the end of the video, which is exactly the kind of silent truncation that makes a
        trajectory unusable later.

        The recording is then stopped **even if teardown failed**: a recording left running
        keeps billing and eventually expires its own artifact, so an orphaned one costs
        money and loses the deliverable.
        """
        if self._sandbox is not None:
            try:
                await self._sandbox.terminate()
                self._sandbox = None
                self._gateway_url = None
            except asyncio.CancelledError:
                # BaseException below is deliberate -- a failed terminate leaks a VM, so
                # nothing may pass silently -- but cancellation is not an error to log and
                # swallow. Re-raised so a timeout or shutdown during terminate still
                # propagates; the handle is intentionally left set for a later retry.
                raise
            except BaseException as exc:
                # Keep the handle so a later close() can retry; {exc!r} + exc_info because a
                # bare exception renders as nothing under {exc} and hides both type and trace.
                logger.warning(
                    "error terminating gateway sandbox %s: %r", self._sandbox.sandbox_id, exc, exc_info=True
                )

        if self._recording_id and self._resolved_device_id:
            try:
                api_key = resolve_api_key(self.api_key_secret_name)
                async with MobileRunClient(api_key, **self._client_kwargs()) as client:
                    await client.stop_recording(self._resolved_device_id, self._recording_id)
                logger.info("stopped recording %s", self._recording_id)
                # Only a confirmed stop forgets the id. Clearing it in a `finally` meant a
                # transient API error erased the one handle a later close() could retry
                # with, turning a retryable failure into a phone that records until its
                # own limits end it.
                self._recording_id = None
            except (MobileRunError, DeviceUnavailable) as exc:
                logger.warning(
                    "could not stop recording %s on device %s (%s) — retained for a later "
                    "close() to retry; if that never runs, stop it from the MobileRun "
                    "dashboard, or it will keep recording until its own limits end it",
                    self._recording_id,
                    self._resolved_device_id,
                    exc,
                )
