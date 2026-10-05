"""Test helpers shared across the suite.

A module rather than a conftest fixture so it can be called at import time, and named
explicitly rather than ``helpers`` because pytest imports it as a top-level-ish module:
a generic name would collide with a same-named file in another package's test directory.
"""

from __future__ import annotations

import struct

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_bytes(width: int = 390, height: int = 844) -> bytes:
    """A byte string with a valid PNG signature and IHDR, which is all the code reads."""
    return (
        PNG_SIGNATURE
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x06\x00\x00\x00"
    )
