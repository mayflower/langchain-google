from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

SESSION_LABEL_KEY = "agent-sandbox.sigs.k8s.io/session-id"


class CompatibilityError(RuntimeError):
    """Raised when a requested feature is unavailable in the released SDK."""


def create_sandbox(
    client: Any,
    *,
    template_name: str,
    namespace: str,
    sandbox_ready_timeout: int,
    labels: dict[str, str] | None,
    shutdown_after_seconds: int | None,
) -> Any:
    """Create a sandbox without passing kwargs unsupported by the SDK."""
    method = client.create_sandbox
    params = inspect.signature(method).parameters
    kwargs: dict[str, Any] = {}

    if "template" in params:
        kwargs["template"] = template_name
    elif "template_name" in params:
        kwargs["template_name"] = template_name
    elif "warmpool" in params:
        kwargs["warmpool"] = template_name
    else:
        msg = (
            "SandboxClient.create_sandbox does not expose a template/warmpool argument"
        )
        raise CompatibilityError(msg)

    if "namespace" in params:
        kwargs["namespace"] = namespace
    if "sandbox_ready_timeout" in params:
        kwargs["sandbox_ready_timeout"] = sandbox_ready_timeout
    if labels:
        if "labels" not in params:
            msg = (
                "SandboxClient.create_sandbox does not support labels; "
                "session reattach requires a released SDK with labels support"
            )
            raise CompatibilityError(msg)
        kwargs["labels"] = labels
    if shutdown_after_seconds is not None:
        if "shutdown_after_seconds" not in params:
            msg = (
                "SandboxClient.create_sandbox does not support "
                "shutdown_after_seconds in this released SDK"
            )
            raise CompatibilityError(msg)
        kwargs["shutdown_after_seconds"] = shutdown_after_seconds

    return method(**kwargs)


def list_sandbox_claims(
    client: Any, *, namespace: str, label_selector: str | None = None
) -> list[str]:
    """List SandboxClaim names using public SDK helpers or accessible clients."""
    method = getattr(client, "list_all_sandboxes", None)
    if callable(method):
        params = inspect.signature(method).parameters
        kwargs: dict[str, Any] = {}
        if "namespace" in params:
            kwargs["namespace"] = namespace
        if label_selector is not None and "label_selector" in params:
            kwargs["label_selector"] = label_selector
        claims = method(**kwargs)
        return _claim_names(claims)

    custom_api = getattr(client, "custom_objects_api", None)
    if custom_api is None:
        custom_api = getattr(
            getattr(client, "k8s_helper", None), "custom_objects_api", None
        )
    if custom_api is None:
        msg = (
            "SandboxClient cannot list sandbox claims; install a released SDK "
            "with list_all_sandboxes or an exposed Kubernetes custom_objects_api"
        )
        raise CompatibilityError(msg)

    response = custom_api.list_namespaced_custom_object(
        group="agent-sandbox.sigs.k8s.io",
        version="v1alpha1",
        namespace=namespace,
        plural="sandboxclaims",
        label_selector=label_selector,
    )
    return [
        item["metadata"]["name"]
        for item in response.get("items", [])
        if item.get("metadata", {}).get("name")
    ]


def get_sandbox(
    client: Any,
    *,
    claim_name: str,
    namespace: str,
    template_name: str | None,
) -> Any:
    """Get an existing sandbox and verify its template when the SDK exposes it."""
    method = client.get_sandbox
    params = inspect.signature(method).parameters
    kwargs: dict[str, Any] = {"claim_name": claim_name}
    if "namespace" in params:
        kwargs["namespace"] = namespace
    if template_name is not None and "template_name" in params:
        kwargs["template_name"] = template_name
    sandbox = method(**kwargs)
    actual = get_claim_template(sandbox)
    if template_name is not None and actual is not None and actual != template_name:
        msg = (
            f"Refusing to reattach claim {namespace}/{claim_name}: "
            f"template {actual!r} does not match requested {template_name!r}"
        )
        raise ValueError(msg)
    return sandbox


def delete_sandbox(client: Any, *, claim_name: str, namespace: str) -> None:
    """Delete a sandbox claim through the released SDK."""
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


def get_claim_template(obj: Any) -> str | None:
    """Return a claim template name when a sandbox/metadata object exposes one."""
    for attr in ("template_name", "template", "warmpool"):
        value = getattr(obj, attr, None)
        if isinstance(value, str):
            return value
    metadata = getattr(obj, "metadata", None)
    labels = getattr(metadata, "labels", None) if metadata is not None else None
    if isinstance(labels, dict):
        value = labels.get("agent-sandbox.sigs.k8s.io/template")
        if isinstance(value, str):
            return value
    return None


def validate_label_value(value: str) -> None:
    """Validate local label values before they are sent to Kubernetes."""
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
