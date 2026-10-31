"""Tests for the product-owned provider contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from langchain_google_agent_sandbox.provider import (
    SandboxLease,
    SandboxSessionProvider,
)


def _sandbox() -> SimpleNamespace:
    return SimpleNamespace(name="sandbox")


def test_lease_exposes_opaque_key_and_handle() -> None:
    sandbox = _sandbox()
    lease = SandboxLease(key="opaque-1", sandbox=sandbox)
    assert lease.key == "opaque-1"
    assert lease.sandbox is sandbox
    assert lease.claim_name is None
    assert lease.namespace is None
    assert dict(lease.metadata) == {}


def test_lease_carries_product_correlation_data() -> None:
    lease = SandboxLease(
        key="opaque-2",
        sandbox=_sandbox(),
        claim_name="claim-abc",
        namespace="tenant-a",
        metadata={"tenant": "a"},
    )
    assert lease.claim_name == "claim-abc"
    assert lease.namespace == "tenant-a"
    assert dict(lease.metadata) == {"tenant": "a"}


def test_lease_is_immutable() -> None:
    lease = SandboxLease(key="opaque-3", sandbox=_sandbox())
    with pytest.raises((AttributeError, TypeError)):
        lease.key = "mutated"  # type: ignore[misc]


@pytest.mark.parametrize("key", ["", None, 5])
def test_lease_rejects_unusable_key(key: object) -> None:
    with pytest.raises(ValueError, match="key must be non-empty text"):
        SandboxLease(key=key, sandbox=_sandbox())  # type: ignore[arg-type]


def test_lease_rejects_missing_sandbox() -> None:
    with pytest.raises(ValueError, match="must be a connected sandbox handle"):
        SandboxLease(key="opaque-4", sandbox=None)


def test_protocol_has_no_delete_method() -> None:
    """Durable deletion is a product operation with no route through here."""
    assert not hasattr(SandboxSessionProvider, "delete")
    assert not hasattr(SandboxSessionProvider, "delete_session")


def test_protocol_is_structurally_satisfiable() -> None:
    class _Provider:
        def acquire(self, config: dict) -> SandboxLease:  # type: ignore[type-arg]
            return SandboxLease(key="k", sandbox=_sandbox())

        def touch(self, lease: SandboxLease) -> None:
            return None

        def close_local(self, lease: SandboxLease) -> None:
            return None

    assert isinstance(_Provider(), SandboxSessionProvider)


def test_incomplete_provider_does_not_satisfy_protocol() -> None:
    class _Partial:
        def acquire(self, config: dict) -> SandboxLease:  # type: ignore[type-arg]
            return SandboxLease(key="k", sandbox=_sandbox())

    assert not isinstance(_Partial(), SandboxSessionProvider)
