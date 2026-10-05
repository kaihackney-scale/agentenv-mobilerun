"""``mobilerun_check_screen`` — grade a run by what is on the phone's screen when it ends.

Reads the deployed env's ``mobilerun_ui_state`` after the step before it (usually
``mobilerun_play``) and checks the visible text, case-insensitively:

* ``all_of``: every one of these must appear;
* ``any_of``: a list of groups, each of which needs at least one of its strings to appear.

It scores 1.0 when every check holds and 0.0 otherwise, under
``context.metadata["verifications"][verifier_id]``, which is where agent-env reads a run's
scores: ``agent-env run`` reports a task whose scores are all 1 as passed.

A deliberately plain verifier. It is deterministic and needs no model, so a grade cannot
disagree with itself between runs; the cost is that it only sees accessibility labels, not
pixels, so write checks against text the target screen is sure to label.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from agent_env.task_step import TaskStep, TaskStepContext

from agentenv_mobilerun.steps.play import deployed_env

logger = logging.getLogger(__name__)


def labels_of(ui_state: dict) -> list[str]:
    return [str(e.get("label")) for e in ui_state.get("elements") or [] if e.get("label")]


def evaluate(labels: list[str], all_of: list[str], any_of: list[list[str]]) -> dict:
    """Each check and whether it held, against the screen's labels."""
    text = " | ".join(labels).casefold()
    checks = [{"all_of": want, "passed": want.casefold() in text} for want in all_of]
    checks += [{"any_of": group, "passed": any(w.casefold() in text for w in group)} for group in any_of]
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks}


class CheckScreenStep(TaskStep):
    """Score the run 1 or 0 by the text on the phone's final screen."""

    type = "mobilerun_check_screen"

    def __init__(self, id, version, *, all_of: Optional[list] = None, any_of: Optional[list] = None,
                 env_step_id: Optional[str] = None, verifier_id: Optional[str] = None, depends_on=None,
                 fail_task_on_error: bool = True) -> None:
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.all_of = [str(s) for s in (all_of or [])]
        self.any_of = [[str(s) for s in group] for group in (any_of or [])]
        self.env_step_id = env_step_id
        self.verifier_id = verifier_id or id      # what agent-env's run summary maps a score back to

    def to_dict(self) -> dict:
        data = {**super().to_dict(), "all_of": self.all_of, "any_of": self.any_of, "verifier_id": self.verifier_id}
        if self.env_step_id:
            data["env_step_id"] = self.env_step_id
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CheckScreenStep":
        return cls(**cls._base_from_dict(data), all_of=data.get("all_of"), any_of=data.get("any_of"),
                   env_step_id=data.get("env_step_id"), verifier_id=data.get("verifier_id"))

    def preflight(self) -> list[str]:
        problems = []
        if not self.all_of and not self.any_of:
            problems.append("give all_of or any_of: a check with nothing to check passes nothing")
        if any(not group for group in self.any_of):
            problems.append("any_of: every group needs at least one string")
        return problems

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        deployed = deployed_env(context, self.env_step_id)
        async with streamablehttp_client(deployed.mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await self.grade(session, context)

    async def grade(self, session: Any, context: TaskStepContext) -> TaskStepContext:
        """Read the screen through an initialised MCP ``session`` and record the score."""
        result = await session.call_tool("mobilerun_ui_state", {})
        raw = "".join(getattr(c, "text", "") for c in result.content)
        try:
            ui_state = json.loads(raw)
        except ValueError as exc:
            raise RuntimeError(f"mobilerun_ui_state did not return JSON ({raw[:200]!r})") from exc
        labels = labels_of(ui_state)
        verdict = evaluate(labels, self.all_of, self.any_of)
        score = 1.0 if verdict["passed"] else 0.0
        context.metadata.setdefault("verifications", {})[self.verifier_id] = {
            "score": score, **verdict, "labels_seen": len(labels)}
        logger.info("mobilerun_check_screen %s: score %.0f (%s)", self.verifier_id, score,
                    ", ".join(f"{c.get('all_of') or c.get('any_of')}={'ok' if c['passed'] else 'missing'}" for c in verdict["checks"]))
        return context
