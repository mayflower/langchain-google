from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any

from langchain_google_agent_sandbox._lifecycle import factory_atexit_cleanup
from langchain_google_agent_sandbox.backend import AgentSandboxBackend


def create_sandbox_backend_factory(
    template_name: str,
    namespace: str = "default",
    **kwargs: Any,
) -> Callable[[Any], AgentSandboxBackend]:
    """Create a DeepAgents backend factory that eagerly provisions a sandbox.

    DeepAgents calls backend factories and then immediately uses the returned
    backend. The factory therefore enters the managed backend before returning
    it and registers a finalizer to clean up the claim if the application does
    not call ``__exit__`` explicitly.

    Args:
        template_name: Sandbox template or warm pool name.
        namespace: Kubernetes namespace.
        **kwargs: Additional ``AgentSandboxBackend.from_template`` arguments,
            including ``client`` and ``session_id``.

    Returns:
        A callable suitable for ``deepagents.create_deep_agent(backend=...)``.
    """

    def factory(_runtime: Any) -> AgentSandboxBackend:
        backend = AgentSandboxBackend.from_template(
            template_name=template_name,
            namespace=namespace,
            **kwargs,
        )
        backend.__enter__()
        if not backend._reattached:
            backend._finalizer = weakref.finalize(  # type: ignore[attr-defined]
                backend,
                factory_atexit_cleanup,
                backend._sdk_client,
                backend._sandbox,
            )
        return backend

    return factory
