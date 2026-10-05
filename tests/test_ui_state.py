"""Compaction of ``/ui-state``.

Field names and shapes here are copied from a live capture against a MobileRun iPhone
(412 nodes, 183 KB, 85 of them labelled), not read off the OpenAPI spec — the spec
declares this route as ``AndroidState`` even on an iPhone, so it is not a reliable guide
to what actually arrives.
"""

from __future__ import annotations

import json

import pytest

from agentenv_mobilerun.server.main import MobileRunTools, compact_ui_state


def node(label="", *, role="XCUIElementTypeOther", bounds=(0, 0, 10, 10), children=(), **flags):
    left, top, right, bottom = bounds
    base = {
        "className": role,
        "text": label,
        "contentDescription": " ",
        "resourceId": "",
        "packageName": "com.apple.springboard",
        "boundsInScreen": {"left": left, "top": top, "right": right, "bottom": bottom},
        "isCheckable": False,
        "isChecked": False,
        "isClickable": False,
        "isEnabled": True,
        "isFocusable": False,
        "isFocused": False,
        "isScrollable": False,
        "isLongClickable": False,
        "isPassword": False,
        "isSelected": False,
        "children": list(children),
    }
    base.update(flags)
    return base


def payload(tree, *, width=393, height=852, **phone):
    state = {"keyboardVisible": False, "packageName": "com.apple.springboard", "isEditable": False}
    state.update(phone)
    return {
        "a11y_tree": tree,
        "device_context": {"screen_bounds": {"width": width, "height": height}},
        "phone_state": state,
    }


def test_unlabelled_container_nodes_are_dropped():
    tree = node(children=[node(), node("Settings", bounds=(10, 20, 30, 40)), node()])
    out = compact_ui_state(payload(tree))
    assert [e["label"] for e in out["elements"]] == ["Settings"]
    # The dropped nodes are still counted, so a caller can see how much was filtered.
    assert out["elements_total"] == 4
    assert out["elements_labelled"] == 1


def test_a_single_space_content_description_is_not_a_label():
    # The real trap: contentDescription is " " on hundreds of container nodes, so a plain
    # truthiness check reports the whole tree as labelled and the compaction does nothing.
    tree = node(children=[node(role="XCUIElementTypeWindow")])
    assert compact_ui_state(payload(tree))["elements"] == []


def test_content_description_is_used_when_text_is_empty():
    tree = node(children=[node(contentDescription="Back")])
    assert compact_ui_state(payload(tree))["elements"][0]["label"] == "Back"


def test_text_wins_over_content_description():
    tree = node(children=[node("Done", contentDescription="Back")])
    assert compact_ui_state(payload(tree))["elements"][0]["label"] == "Done"


def test_role_drops_the_prefix_every_node_carries():
    tree = node(children=[node("Send", role="XCUIElementTypeButton")])
    assert compact_ui_state(payload(tree))["elements"][0]["role"] == "Button"


def test_center_and_size_come_from_the_bounds():
    tree = node(children=[node("Send", bounds=(100, 200, 300, 260))])
    element = compact_ui_state(payload(tree))["elements"][0]
    assert element["center"] == [200, 230]
    assert element["size"] == [200, 60]


def test_center_is_in_screenshot_pixels_so_it_can_be_tapped_directly():
    """The whole point of the scale argument.

    /ui-state reports bounds in logical points; every gesture takes screenshot pixels and
    divides by the same scale. Returning a point-space centre meant the obvious thing --
    read a centre from ui_state, pass it to tap -- landed at a third of the target on a 3x
    phone. One space, so that round-trip is correct by construction.
    """
    tree = node(children=[node("Send", bounds=(100, 200, 300, 260))])
    element = compact_ui_state(payload(tree), scale=3.0)["elements"][0]
    assert element["center"] == [600, 690]   # 3x the point-space [200, 230]
    assert element["size"] == [600, 180]


def test_the_payload_names_the_space_its_coordinates_are_in():
    out = compact_ui_state(payload(tree_one()), scale=3.0)
    assert out["coordinate_space"] == "screenshot_pixels"
    assert out["screen_pixels"] == [1179, 2556]
    assert out["screen_points"] == [393, 852]


def tree_one():
    return node(children=[node("Send", bounds=(100, 200, 300, 260))])


def test_a_scale_of_one_is_the_identity():
    """A device that reports points == pixels must be untouched by the conversion."""
    element = compact_ui_state(payload(tree_one()), scale=1.0)["elements"][0]
    assert element["center"] == [200, 230]


def test_only_notable_flags_are_reported():
    tree = node(
        children=[
            node("Password", isPassword=True, isEnabled=False),
            node("Ordinary"),
        ]
    )
    reported, ordinary = compact_ui_state(payload(tree))["elements"]
    assert reported["isPassword"] is True
    assert reported["isEnabled"] is False
    # A node at every boring default carries no flags at all, which is the common case and
    # the whole reason the output is small.
    assert set(ordinary) == {"label", "role", "center", "size"}


def test_android_only_flags_are_never_reported():
    # isClickable is false on every node of a real iPhone tree, so reporting it would add
    # bulk that says nothing, and filtering on it would drop the entire screen.
    tree = node(children=[node("Send", isClickable=True, isLongClickable=True)])
    element = compact_ui_state(payload(tree))["elements"][0]
    assert "isClickable" not in element and "isLongClickable" not in element


def test_screen_points_and_app_state_are_surfaced():
    tree = node(children=[node("x")])
    out = compact_ui_state(payload(tree, width=393, height=852, keyboardVisible=True, isEditable=True))
    assert out["screen_points"] == [393, 852]
    assert out["app"] == {
        "package": "com.apple.springboard",
        "keyboard_visible": True,
        "editable": True,
    }


def test_truncation_is_declared_not_silent():
    tree = node(children=[node(f"item {i}") for i in range(10)])
    out = compact_ui_state(payload(tree), max_elements=3)
    assert len(out["elements"]) == 3
    assert out["truncated"] is True
    # The true count survives truncation, so a caller knows what it did not see.
    assert out["elements_labelled"] == 10


def test_untruncated_output_says_so():
    tree = node(children=[node("only")])
    assert compact_ui_state(payload(tree), max_elements=3)["truncated"] is False


def test_a_missing_or_malformed_tree_does_not_crash():
    assert compact_ui_state({})["elements"] == []
    assert compact_ui_state({"a11y_tree": None})["elements"] == []
    assert compact_ui_state({"a11y_tree": []})["elements"] == []
    assert compact_ui_state({"a11y_tree": {"children": None}})["elements"] == []


def test_compaction_is_a_large_reduction_on_a_realistic_tree():
    # Proportions taken from the live capture: ~20% of nodes carry a label.
    deep = [node(f"label {i}" if i % 5 == 0 else "") for i in range(400)]
    raw = payload(node(children=deep))
    before, after = len(json.dumps(raw)), len(json.dumps(compact_ui_state(raw)))
    assert after * 10 < before, f"only shrank {before} -> {after}"


@pytest.mark.asyncio
async def test_the_tool_returns_the_compacted_view_not_the_raw_tree():
    import httpx

    from agentenv_mobilerun.client import MobileRunClient

    raw = payload(node(children=[node("Settings", role="XCUIElementTypeButton", bounds=(0, 0, 40, 20))]))
    client = MobileRunClient("dr_sk_test", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=raw)))
    result = json.loads(await MobileRunTools(client, "dev-1").ui_state())
    assert result["elements"] == [
        {"label": "Settings", "role": "Button", "center": [20, 10], "size": [40, 20]}
    ]
    assert "a11y_tree" not in result


@pytest.mark.asyncio
async def test_the_tool_reports_an_unexpected_payload_instead_of_crashing():
    import httpx

    from agentenv_mobilerun.client import MobileRunClient

    client = MobileRunClient("dr_sk_test", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
    assert json.loads(await MobileRunTools(client, "dev-1").ui_state())["ok"] is False


def test_zero_size_elements_are_not_offered_as_tap_targets():
    """A real iPhone home screen reports off-page apps with bounds of all zeros.

    Left in `elements` they arrive as ``{"label": "Amazon", "center": [0, 0]}``, which
    reads as a perfectly good tap target and sends the agent to the top-left corner.
    Measured on a live device: 28 of 32 home-screen icons were zero-bounds.
    """
    tree = node(
        children=[
            node("Amazon", role="XCUIElementTypeIcon", bounds=(0, 0, 0, 0)),
            node("Safari", role="XCUIElementTypeIcon", bounds=(119, 751, 187, 820)),
        ]
    )
    out = compact_ui_state(payload(tree))

    assert [e["label"] for e in out["elements"]] == ["Safari"]
    assert out["offscreen_labels"] == ["Amazon"]
    assert all(e["center"] != [0, 0] for e in out["elements"])


def test_offscreen_labels_are_still_reported_so_the_agent_knows_the_app_exists():
    # Dropping them entirely would hide that the app is installed at all, which is worth
    # knowing — it just is not reachable from this screen without navigating.
    tree = node(children=[node(f"App {i}", bounds=(0, 0, 0, 0)) for i in range(3)])
    out = compact_ui_state(payload(tree))
    assert out["elements"] == []
    assert out["offscreen_labels"] == ["App 0", "App 1", "App 2"]


def test_a_degenerate_bound_in_one_axis_also_counts_as_offscreen():
    tree = node(children=[node("Sliver", bounds=(10, 10, 10, 90)), node("Real", bounds=(0, 0, 10, 10))])
    out = compact_ui_state(payload(tree))
    assert [e["label"] for e in out["elements"]] == ["Real"]
    assert out["offscreen_labels"] == ["Sliver"]
