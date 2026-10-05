"""``mobilerun_check_screen``: a run's grade, from the text on the phone's final screen."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("agent_env", reason="agentenv-framework is not installed")

from agentenv_mobilerun.steps.check_screen import CheckScreenStep, evaluate, labels_of  # noqa: E402

ROUTE = ["Directions", "Brandenburg Gate", "Museum Island", "Walk", "24 min", "8:40 ETA · 1.1 mi"]


def test_every_group_needs_one_of_its_strings_case_insensitively():
    verdict = evaluate(ROUTE, [], [["museum island", "Museumsinsel"], ["WALK", "walking"], [" min"]])
    assert verdict["passed"] and all(c["passed"] for c in verdict["checks"])


def test_one_missing_group_fails_the_screen():
    verdict = evaluate(["Directions", "Museum Island", "Drive", "8 min"], [], [["museum island"], ["walk"]])
    assert not verdict["passed"]
    assert [c["passed"] for c in verdict["checks"]] == [True, False]


def test_all_of_needs_every_string():
    assert evaluate(ROUTE, ["Brandenburg Gate", "Museum Island"], [])["passed"]
    assert not evaluate(ROUTE, ["Brandenburg Gate", "Reichstag"], [])["passed"]


def test_a_check_with_nothing_to_check_does_not_pass():
    assert not evaluate(ROUTE, [], [])["passed"]
    assert CheckScreenStep("c", 1).preflight()


class Screen:
    def __init__(self, labels):
        self.labels = labels

    async def call_tool(self, name, args):
        assert name == "mobilerun_ui_state"
        state = {"elements": [{"label": l, "center": [1, 1]} for l in self.labels]}
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(state))])


@pytest.mark.asyncio
async def test_the_score_goes_where_agent_env_reads_scores():
    """agent-env's run summary reads context.metadata["verifications"][key]["score"] and maps the key back to a
    step through the step's verifier_id; a task whose scores are all >= 1 is reported as passed."""
    step = CheckScreenStep("check", 1, any_of=[["Museum Island"], ["walk"]])
    context = SimpleNamespace(metadata={})
    await step.grade(Screen(ROUTE), context)
    assert step.verifier_id == "check"
    assert context.metadata["verifications"]["check"]["score"] == 1.0

    await step.grade(Screen(["Directions", "Drive"]), context)
    assert context.metadata["verifications"]["check"]["score"] == 0.0


def test_labels_skip_unlabelled_elements():
    assert labels_of({"elements": [{"label": "A"}, {"label": None}, {}]}) == ["A"]


def test_the_step_round_trips():
    step = CheckScreenStep("check", 1, all_of=["x"], any_of=[["a", "b"]], env_step_id="deploy", verifier_id="v")
    again = CheckScreenStep.from_dict(step.to_dict())
    assert (again.all_of, again.any_of, again.env_step_id, again.verifier_id) == (["x"], [["a", "b"]], "deploy", "v")
