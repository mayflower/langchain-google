"""Durable provider-path integration tests against a real agent-sandbox.

These prove the property the unit tests can only assert against fakes: that
closing the adapter leaves the remote Claim running, and that a *fresh*
provider instance reattaches to the same durable workspace.

The fixture creates and deletes its Claim through the official SDK directly,
never through the adapter, because the adapter deliberately has no code path
that deletes durable infrastructure.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from langchain_google_agent_sandbox import (
    ProviderSessionAgentSandboxBackend,
    SandboxLease,
)


def _configured() -> bool:
    if not os.environ.get("LANGCHAIN_SANDBOX_WARM_POOL"):
        return False
    return bool(
        os.environ.get("LANGCHAIN_API_URL")
        or os.environ.get("LANGCHAIN_GATEWAY_NAME")
        or os.environ.get("LANGCHAIN_USE_TUNNEL") == "1"
    )


requires_sandbox = pytest.mark.skipif(
    not _configured(),
    reason="agent-sandbox integration environment is not configured",
)

NAMESPACE = os.environ.get("LANGCHAIN_NAMESPACE", "default")


def _client() -> Any:
    from k8s_agent_sandbox import SandboxClient
    from k8s_agent_sandbox.models import (
        SandboxDirectConnectionConfig,
        SandboxGatewayConnectionConfig,
        SandboxLocalTunnelConnectionConfig,
    )

    if api_url := os.environ.get("LANGCHAIN_API_URL"):
        config: Any = SandboxDirectConnectionConfig(api_url=api_url)
    elif gateway_name := os.environ.get("LANGCHAIN_GATEWAY_NAME"):
        config = SandboxGatewayConnectionConfig(
            gateway_name=gateway_name,
            gateway_namespace=os.environ.get("LANGCHAIN_GATEWAY_NAMESPACE", "default"),
        )
    else:
        config = SandboxLocalTunnelConnectionConfig()
    return SandboxClient(connection_config=config)


class AttachingProvider:
    """Provider that attaches to a Claim someone else already owns.

    This mirrors the production shape: a product control plane owns the Claim,
    and hands this package an authorized handle to an existing session.
    """

    def __init__(self, claim_name: str) -> None:
        self.claim_name = claim_name
        self.client = _client()
        self.acquisitions = 0

    def acquire(self, config: Mapping[str, Any]) -> SandboxLease:
        del config
        self.acquisitions += 1
        sandbox = self.client.get_sandbox(
            claim_name=self.claim_name, namespace=NAMESPACE
        )
        return SandboxLease(
            key=f"lease-{self.claim_name}",
            sandbox=sandbox,
            claim_name=self.claim_name,
            namespace=NAMESPACE,
        )

    def touch(self, lease: SandboxLease) -> None:
        del lease

    def close_local(self, lease: SandboxLease) -> None:
        close = getattr(lease.sandbox, "close_connection", None)
        if callable(close):
            close()


@pytest.fixture
def durable_claim() -> Iterator[str]:
    """Create a Claim outside the adapter, and delete it outside the adapter."""
    client = _client()
    sandbox = client.create_sandbox(
        warmpool=os.environ["LANGCHAIN_SANDBOX_WARM_POOL"],
        namespace=NAMESPACE,
        labels={"test-run": uuid.uuid4().hex[:8]},
    )
    claim_name = sandbox.claim_name
    try:
        yield claim_name
    finally:
        client.delete_sandbox(claim_name=claim_name, namespace=NAMESPACE)


def config_for(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


@requires_sandbox
def test_attach_execute_and_transfer_files(durable_claim: str) -> None:
    provider = AttachingProvider(durable_claim)
    backend = ProviderSessionAgentSandboxBackend(provider)

    entry = backend._entry(config_for("integration"))
    assert entry.lease.claim_name == durable_claim

    backend._active_config = staticmethod(  # type: ignore[assignment]
        lambda: config_for("integration")
    )

    assert backend.execute("pwd").exit_code == 0
    assert backend.write("/durable.txt", "hello\nworld").error is None
    assert backend.read("/durable.txt").file_data["content"] == "hello\nworld"

    uploads = backend.upload_files({"/uploaded.txt": b"payload"})
    assert uploads[0].error is None
    downloads = backend.download_files(["/uploaded.txt"])
    assert downloads[0].content == b"payload"

    backend.close()


@requires_sandbox
def test_close_leaves_the_claim_running_and_a_fresh_provider_reattaches(
    durable_claim: str,
) -> None:
    """The load-bearing durability property, unprovable against a fake."""
    first = ProviderSessionAgentSandboxBackend(AttachingProvider(durable_claim))
    first._active_config = staticmethod(  # type: ignore[assignment]
        lambda: config_for("durable")
    )
    assert first.write("/persisted.txt", "survives").error is None
    first.close()

    # The Claim must still exist after the adapter closed: ask the SDK directly.
    client = _client()
    assert client.get_sandbox_claim_warmpool_name(durable_claim, NAMESPACE)

    # A brand-new provider and backend must find the same workspace contents.
    second_provider = AttachingProvider(durable_claim)
    second = ProviderSessionAgentSandboxBackend(second_provider)
    second._active_config = staticmethod(  # type: ignore[assignment]
        lambda: config_for("durable")
    )
    assert second.read("/persisted.txt").file_data["content"] == "survives"
    assert second_provider.acquisitions == 1
    second.close()

    # Still alive after the second adapter closed too.
    assert client.get_sandbox_claim_warmpool_name(durable_claim, NAMESPACE)


@requires_sandbox
def test_eviction_closes_locally_without_destroying_the_workspace(
    durable_claim: str,
) -> None:
    provider = AttachingProvider(durable_claim)
    backend = ProviderSessionAgentSandboxBackend(provider, max_cached_sessions=1)
    backend._active_config = staticmethod(  # type: ignore[assignment]
        lambda: config_for("evicted")
    )
    backend.write("/kept.txt", "kept")

    # Force the first session out of the local cache.
    backend._entry(config_for("other"))

    assert backend.read("/kept.txt").file_data["content"] == "kept"
    assert provider.acquisitions >= 2
    backend.close()
