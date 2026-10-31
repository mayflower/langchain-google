from __future__ import annotations

import warnings
from typing import Any

from langchain_google_agent_sandbox.backend import AgentSandboxBackend


class _FactoryCompatibleBackend(AgentSandboxBackend):
    """Concrete 0.7 backend that tolerates one legacy factory invocation."""

    def __call__(self, _runtime: Any) -> AgentSandboxBackend:
        warnings.warn(
            "Calling the backend as a factory is deprecated; pass the initialized "
            "backend directly to DeepAgents 0.7",
            DeprecationWarning,
            stacklevel=2,
        )
        return self


def _initialize_backend(backend: AgentSandboxBackend) -> AgentSandboxBackend:
    backend.__enter__()
    backend._register_finalizer()
    return backend


def create_sandbox_backend(
    warm_pool: str,
    namespace: str = "default",
    **kwargs: Any,
) -> AgentSandboxBackend:
    """Create and initialize an ephemeral DeepAgents 0.7 backend.

    DeepAgents 0.7 accepts concrete backend instances rather than factories.
    This helper enters the managed backend before returning it and registers a
    finalizer in case the application does not exit its context explicitly.

    Args:
        warm_pool: SandboxWarmPool name.
        namespace: Kubernetes namespace.
        **kwargs: Additional ``AgentSandboxBackend.from_warm_pool`` arguments,
            including ``client`` and ``session_id``.

    Returns:
        An initialized backend suitable for
        ``deepagents.create_deep_agent(backend=...)``.
    """
    return _initialize_backend(
        AgentSandboxBackend.from_warm_pool(
            warm_pool=warm_pool,
            namespace=namespace,
            **kwargs,
        )
    )


def create_sandbox_backend_factory(
    warm_pool: str,
    namespace: str = "default",
    **kwargs: Any,
) -> AgentSandboxBackend:
    """Deprecated name for :func:`create_sandbox_backend`.

    DeepAgents 0.7 removed backend factories, so this function now returns an
    initialized backend instance.
    """
    warnings.warn(
        "create_sandbox_backend_factory() now returns a concrete backend for "
        "DeepAgents 0.7; use create_sandbox_backend()",
        DeprecationWarning,
        stacklevel=2,
    )
    return _initialize_backend(
        _FactoryCompatibleBackend.from_warm_pool(
            warm_pool=warm_pool,
            namespace=namespace,
            **kwargs,
        )
    )
