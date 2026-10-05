"""The capability map's wire format, shared by both halves of the plugin.

The env resolves the map once at deploy time and passes it to the container; the server
reads it back. Those two live on opposite sides of a deliberate dependency split -- the
env imports agent-env, the image must not -- so the encoder sits here, in a module that
imports neither. It used to live in ``env.py``, which meant the only test holding the
writer and the reader together could not run without the framework installed.

Deliberately not JSON: the value is rendered into a docker-compose ``environment:``
entry, and ``": "`` is illegal in a YAML plain scalar.
"""

from __future__ import annotations

ENV_VAR = "MOBILERUN_CAPABILITIES"


def encode_capabilities(capabilities: dict[str, bool]) -> str:
    """``{"clipboard": False}`` -> ``"clipboard=false"``.

    Every name is emitted, including the false ones. ``select_tools`` treats an absent
    name as present, so a payload listing only the supported names would gate nothing
    out -- the deployer has to say what is *not* supported for the gate to do anything.
    """
    return ",".join(
        f"{name}={'true' if value else 'false'}" for name, value in sorted(capabilities.items())
    )


def decode_capabilities(raw: str) -> dict[str, bool]:
    """The inverse of :func:`encode_capabilities`; unparsable pairs are dropped."""
    out: dict[str, bool] = {}
    for pair in (raw or "").split(","):
        name, _, value = pair.partition("=")
        name = name.strip()
        if name and _:
            out[name] = value.strip().lower() == "true"
    return out
