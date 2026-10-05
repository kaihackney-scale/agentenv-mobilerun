"""MobileRun cloud-phone environment plugin for agent-env.

Import-light on purpose. ``client`` is the only submodule the MCP server image installs,
and the image does not have agent-env, so importing this package must never pull
``env``/``cli``/``steps`` in. The lazy ``__getattr__`` below keeps
``from agentenv_mobilerun import MobileRunEnv`` working for plugin users without making
``import agentenv_mobilerun.client`` fail inside the container.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = ["MobileRunClient", "MobileRunError", "MobileRunEnv", "CollectRecordingStep", "__version__"]

_LAZY = {
    "MobileRunClient": ("agentenv_mobilerun.client", "MobileRunClient"),
    "MobileRunError": ("agentenv_mobilerun.client", "MobileRunError"),
    "MobileRunEnv": ("agentenv_mobilerun.env", "MobileRunEnv"),
    "CollectRecordingStep": ("agentenv_mobilerun.steps.collect_recording", "CollectRecordingStep"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


def __dir__() -> list[str]:
    return sorted(__all__)
