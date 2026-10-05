"""The bundle the package ships, checked the way `agent-env run` checks it before writing anything."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("agent_env", reason="agentenv-framework is not installed")

BUNDLE = Path(__file__).resolve().parents[1] / "src" / "agentenv_mobilerun" / "bundles" / "mobilerun"


def test_the_bundle_parses_and_every_task_builds_with_the_registered_steps():
    """check_bundle builds each task's steps through agent-env's step registry, so a step type this package
    forgot to register, or a field a step doesn't take, fails here rather than in someone's first run."""
    from agent_env.bundle import parse_bundle
    from agent_env.bundle.plan import check_bundle
    from agent_env.bundle.resolve import resolve_bundle

    check_bundle(resolve_bundle(parse_bundle(BUNDLE)))


def test_the_task_seats_a_model_grades_the_screen_and_keeps_the_recording():
    from agent_env.task_step.registry import get_task_step_registry

    steps = json.loads((BUNDLE / "tasks" / "maps-walking.json").read_text())
    assert [s["type"] for s in steps] == ["deploy_env", "mobilerun_play", "mobilerun_check_screen", "mobilerun_collect_recording"]
    registry = get_task_step_registry()
    for raw in steps[1:3]:
        built = registry[raw["type"]].from_dict({**raw, "version": None})
        assert built.preflight() == [], raw["id"]
