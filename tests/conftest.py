"""Shared fixtures.

The client is driven through ``httpx.MockTransport`` rather than a mocking library, so
the tests exercise the real request construction (URLs, bodies, headers, redirect
following) with nothing stubbed above the socket.
"""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from agentenv_mobilerun.client import MobileRunClient


@pytest.fixture
def make_client() -> Callable[[Callable[[httpx.Request], httpx.Response]], MobileRunClient]:
    """Build a client whose transport is the given handler."""

    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> MobileRunClient:
        # max_retries=0: these tests inject failures, and the SDK's default retry would
        # both mask the un-retried path and add backoff sleeps to every such test.
        return MobileRunClient(
            "dr_sk_test", transport=httpx.MockTransport(handler), max_retries=0
        )

    return factory
