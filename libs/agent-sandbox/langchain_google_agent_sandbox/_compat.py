from __future__ import annotations

from collections.abc import Iterable
from typing import Any

SESSION_LABEL_KEY = "agent-sandbox.sigs.k8s.io/session-id"


def create_sandbox(
    client: Any,
    *,
    warm_pool: str,
    namespace: str,
    sandbox_ready_timeout: int,
    labels: dict[str, str] | None,
    shutdown_after_seconds: int | None,
) -> Any:
    """Create a sandbox through the pinned v1beta1 SDK."""
    return client.create_sandbox(
        warmpool=warm_pool,
        namespace=namespace,
        sandbox_ready_timeout=sandbox_ready_timeout,
        labels=labels,
        shutdown_after_seconds=shutdown_after_seconds,
    )


def list_sandbox_claims(
    client: Any, *, namespace: str, label_selector: str | None = None
) -> list[str]:
    """List SandboxClaim names through the pinned v1beta1 SDK."""
    claims = client.list_all_sandboxes(
        namespace=namespace,
        label_selector=label_selector,
    )
    return _claim_names(claims)


def get_sandbox(
    client: Any,
    *,
    claim_name: str,
    namespace: str,
    warm_pool: str | None,
) -> Any:
    """Get an existing sandbox and validate its WarmPool."""
    if warm_pool is not None:
        actual = client.get_sandbox_claim_warmpool_name(claim_name, namespace)
    else:
        actual = None
    if warm_pool is not None and actual != warm_pool:
        msg = (
            f"Refusing to reattach claim {namespace}/{claim_name}: "
            f"warm pool {actual!r} does not match requested {warm_pool!r}"
        )
        raise ValueError(msg)
    return client.get_sandbox(claim_name=claim_name, namespace=namespace)


def delete_sandbox(client: Any, *, claim_name: str, namespace: str) -> None:
    """Delete a sandbox claim through the released SDK."""
    client.delete_sandbox(claim_name=claim_name, namespace=namespace)


def validate_label_value(value: str) -> None:
    """Validate a legacy ephemeral-session label value."""
    if len(value) > 63:
        msg = "session_id must be 63 characters or fewer to fit a Kubernetes label"
        raise ValueError(msg)
    if not value:
        msg = "session_id cannot be empty"
        raise ValueError(msg)
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if any(char not in allowed for char in value):
        msg = "session_id contains characters that are invalid in Kubernetes labels"
        raise ValueError(msg)
    if not value[0].isalnum() or not value[-1].isalnum():
        msg = "session_id must start and end with an alphanumeric character"
        raise ValueError(msg)


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


def send_files_update(update: dict[str, Any]) -> None:
    """Best-effort DeepAgents state["files"] sync through LangGraph context."""
    try:
        from langgraph._internal._constants import CONFIG_KEY_SEND
        from langgraph.config import get_config
    except (ImportError, ModuleNotFoundError):
        return
    try:
        config = get_config()
    except RuntimeError:
        return
    send = config.get("configurable", {}).get(CONFIG_KEY_SEND)
    if send is not None:
        send([("files", update)])


def _claim_names(claims: Iterable[Any]) -> list[str]:
    names: list[str] = []
    for claim in claims:
        if isinstance(claim, str):
            names.append(claim)
            continue
        metadata = getattr(claim, "metadata", None)
        name = getattr(metadata, "name", None) if metadata is not None else None
        if isinstance(name, str):
            names.append(name)
    return names
