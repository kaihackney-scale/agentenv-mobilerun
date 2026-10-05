"""Namespaces this package's tests.

Without it, pytest's default prepend import mode would import ``tests/test_client.py``
as the top-level module ``test_client`` — which already exists under
``packages/agentenv-protocol/tests`` — and the run would die on an import-file mismatch.
"""
