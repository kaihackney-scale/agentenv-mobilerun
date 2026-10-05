"""Resolving the MobileRun API key, in a module that does not import agent-env.

The lookup itself never needed the framework -- the secret-store branch imports it
lazily and tolerates its absence -- but it lived in ``env.py``, which imports
``agent_env`` at module scope. That put the CLI's `doctor` (and therefore every test of
it) behind a dependency that is not on PyPI yet, so the CLI went untested in CI.

Same split as ``capabilities.py``: the pieces both halves of the plugin need live where
neither half's dependencies reach them.
"""

from __future__ import annotations

import logging
import os

from agentenv_mobilerun.client import MobileRunError

logger = logging.getLogger(__name__)

API_KEY_NAME = "MOBILERUN_API_KEY"


def resolve_api_key(secret_name: str = API_KEY_NAME) -> str:
    """The MobileRun API key, from the process environment or the secret store.

    Checked in that order so a local run needs nothing but an exported variable, while a
    deployment with a real secret backend resolves the same logical name through it.
    """
    direct = (os.environ.get(secret_name) or "").strip()
    if direct:
        return direct
    try:
        from agent_env.config import get_config

        value = get_config().get_secret_store().get(secret_name)
    except Exception as exc:  # a missing/misconfigured store must not mask the real advice
        logger.debug("secret store lookup for %s failed: %s", secret_name, exc)
        value = None
    if value:
        return value.strip()
    raise MobileRunError(
        f"no MobileRun API key found. Export {secret_name}=dr_sk_... (create one on MobileRun's API "
        f"Keys page), or add it to the secret store your .agentenv/config.toml configures under the "
        f"name {secret_name!r}."
    )
