"""``mobilerun_collect_recording`` — copy a run's recording into durable storage.

Registered on the ``agent_env.task_steps`` entry point by installing the package. Place it
**after** the step that drives the phone (``mobilerun_play``, or an agent's ``prompt_agent``). It stops the recording that
:class:`~agentenv_mobilerun.env.MobileRunEnv` started, waits for the server-side upload,
and copies both artifacts into the configured object store.

Why this step exists at all: MobileRun is not durable storage. The video route answers a
302 to a presigned URL that expires in **900 seconds**, and each recording carries a
``retentionDays`` after which the artifact routes return 410 Gone. So the only safe
pattern is "fetch promptly, store yourself, never hand out their link".

What the trajectory buys you: exact coordinates per action, ``seq``, ``at_ms``, gesture
``duration_ms``, the display ``scale``/``rotation``, a ``valid``/``reasons`` contract, and
``video.timeline.action_zero_to_video_ms`` with an ``uncertainty_ms``. That last pair is
the action-clock-to-video-clock offset, measured and bounded by the device that recorded
both — which is the part a client-side recorder cannot know.

⚠️ **Privacy.** A recording captures whatever is on screen, not just the task. On a
bring-your-own phone that includes notifications and personal accounts. This step writes
to the object store *you* configured; it never publishes anything.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent_env.task_step import TaskStep, TaskStepContext

from agentenv_mobilerun.client import MobileRunClient, MobileRunError
from agentenv_mobilerun.env import resolve_api_key

logger = logging.getLogger(__name__)

#: Metadata keys stamped on the deployment by MobileRunEnv.deploy().
DEVICE_KEY = "mobilerun_device_id"
RECORDING_KEY = "mobilerun_recording_id"


class CollectRecordingStep(TaskStep):
    """Stop, await, and store the MobileRun recording belonging to this run."""

    type = "mobilerun_collect_recording"

    def __init__(
        self,
        id,
        version,
        *,
        key_prefix: str = "mobilerun-recordings",
        timeout_seconds: float = 300.0,
        depends_on=None,
        # A missing recording must not fail an otherwise good run: the demonstration is
        # still scoreable without video, so this step defaults to non-fatal.
        fail_task_on_error: bool = False,
    ) -> None:
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.key_prefix = key_prefix.strip("/")
        self.timeout_seconds = float(timeout_seconds)

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "key_prefix": self.key_prefix,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CollectRecordingStep":
        return cls(
            **cls._base_from_dict(data),
            key_prefix=data.get("key_prefix", "mobilerun-recordings"),
            timeout_seconds=float(data.get("timeout_seconds", 300.0)),
        )

    def preflight(self) -> list[str]:
        """Catch the credential problem before a run provisions anything.

        This is the whole point of the seam: without a key the step cannot possibly
        collect, and finding that out after an agent has already driven a phone for ten
        minutes wastes the run and the recording it was meant to keep.
        """
        try:
            resolve_api_key()
        except MobileRunError as exc:
            return [str(exc)]
        return []

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        target = self._find_recording(context)
        if target is None:
            logger.info(
                "no MobileRun recording on this run's deployments; nothing to collect "
                "(set record=True on the env to start one)"
            )
            return context
        device_id, recording_id = target

        api_key = resolve_api_key()
        async with MobileRunClient(api_key) as client:
            # Stopping an already-stopped recording is the normal case when the env was
            # closed first, and it must not abort the collection that follows.
            try:
                await client.stop_recording(device_id, recording_id)
            except MobileRunError as exc:
                logger.info("recording %s was already stopped (%s)", recording_id, exc)

            info = await client.await_recording(
                device_id, recording_id, timeout_seconds=self.timeout_seconds
            )
            video = await client.download_recording_video(device_id, recording_id)
            trajectory = await client.download_recording_trajectory(device_id, recording_id)

        stored = self._store(recording_id, video, trajectory)
        # Append rather than replace: the context's collections are shared across steps and
        # other steps' metadata must survive this one.
        context.metadata.setdefault("mobilerun", {})[recording_id] = {
            "device_id": device_id,
            "status": info.get("status"),
            "actions": info.get("actions"),
            **stored,
        }
        logger.info("stored MobileRun recording %s (%d bytes of video)", recording_id, len(video))
        return context

    def _find_recording(self, context: TaskStepContext) -> Optional[tuple[str, str]]:
        """The (device, recording) pair for this run, from the deployed envs' metadata.

        Scans in reverse so that, if a task deployed more than one MobileRun env, the most
        recent deployment wins — which is the one the agent was just prompted against.
        """
        for deployed in reversed(context.deployed_envs):
            metadata = deployed.metadata or {}
            device_id = metadata.get(DEVICE_KEY)
            recording_id = metadata.get(RECORDING_KEY)
            if device_id and recording_id:
                return str(device_id), str(recording_id)
        return None

    def _store(self, recording_id: str, video: bytes, trajectory: bytes) -> dict[str, str]:
        from agent_env.config import get_config

        store = get_config().get_object_store()
        base = f"{self.key_prefix}/{recording_id}"
        # allow_overwrite because the key is derived from the recording id alone, so it
        # names exactly one immutable artifact. Without it a retried step (retry_config
        # re-dispatches a failed span) hits ObjectAlreadyExistsError on the second attempt
        # and can never recover the recording it exists to preserve.
        return {
            "video_url": store.put(
                f"{base}/session.mp4", video, content_type="video/mp4", allow_overwrite=True
            ),
            "trajectory_url": store.put(
                f"{base}/trajectory.jsonl",
                trajectory,
                content_type="application/x-ndjson",
                allow_overwrite=True,
            ),
        }
