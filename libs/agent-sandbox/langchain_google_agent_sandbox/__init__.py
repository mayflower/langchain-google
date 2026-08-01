from __future__ import annotations

from importlib import metadata

from langchain_google_agent_sandbox.backend import AgentSandboxBackend
from langchain_google_agent_sandbox.factory import (
    create_sandbox_backend,
    create_sandbox_backend_factory,
)
from langchain_google_agent_sandbox.policy import SandboxPolicyWrapper
from langchain_google_agent_sandbox.session_backend import (
    SessionAgentSandboxBackend,
    SessionLifecycleContext,
    SessionSandboxEndpoint,
    default_session_resolver,
)

try:
    __version__ = metadata.version(__package__ or "langchain-google-agent-sandbox")
except metadata.PackageNotFoundError:
    __version__ = ""

__all__ = [
    "AgentSandboxBackend",
    "SandboxPolicyWrapper",
    "SessionAgentSandboxBackend",
    "SessionLifecycleContext",
    "SessionSandboxEndpoint",
    "__version__",
    "create_sandbox_backend",
    "create_sandbox_backend_factory",
    "default_session_resolver",
]
