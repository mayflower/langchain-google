"""Thin wrappers over the official ``k8s-agent-sandbox`` client.

Every function here calls a documented public method of
``k8s_agent_sandbox.sandbox_client.SandboxClient``. Nothing in this module
reaches into SDK internals, and nothing here manages durable session
lifecycle: the Claim operations that remain serve only the ephemeral
convenience API in :mod:`~langchain_google_agent_sandbox.backend`.
"""

from __future__ import annotations

from typing import Any


def create_sandbox(
    client: Any,
    *,
    warm_pool: str,
    namespace: str,
    sandbox_ready_timeout: int,
    labels: dict[str, str] | None,
    shutdown_after_seconds: int | None,
) -> Any:
    """Create an ephemeral, SDK-named sandbox."""
    return client.create_sandbox(
        warmpool=warm_pool,
        namespace=namespace,
        sandbox_ready_timeout=sandbox_ready_timeout,
        labels=labels,
        shutdown_after_seconds=shutdown_after_seconds,
    )


def delete_sandbox(client: Any, *, claim_name: str, namespace: str) -> None:
    """Delete a Claim this package created through the ephemeral API."""
    client.delete_sandbox(claim_name=claim_name, namespace=namespace)


def normalize_read_bytes(data: Any) -> bytes:
    """Normalize SDK file read results to bytes."""
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    content = getattr(data, "content", None)
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    msg = f"Unsupported file read response type: {type(data).__name__}"
    raise TypeError(msg)
