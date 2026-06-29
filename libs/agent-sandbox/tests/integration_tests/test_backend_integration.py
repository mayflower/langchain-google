from __future__ import annotations

import os

import pytest

from langchain_google_agent_sandbox import (
    AgentSandboxBackend,
    SandboxPolicyWrapper,
    create_sandbox_backend_factory,
)


def _configured() -> bool:
    if not os.environ.get("LANGCHAIN_SANDBOX_TEMPLATE"):
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


def _client():
    from k8s_agent_sandbox import SandboxClient
    from k8s_agent_sandbox.models import (
        SandboxDirectConnectionConfig,
        SandboxGatewayConnectionConfig,
        SandboxLocalTunnelConnectionConfig,
    )

    if api_url := os.environ.get("LANGCHAIN_API_URL"):
        config = SandboxDirectConnectionConfig(api_url=api_url)
    elif gateway_name := os.environ.get("LANGCHAIN_GATEWAY_NAME"):
        config = SandboxGatewayConnectionConfig(
            gateway_name=gateway_name,
            gateway_namespace=os.environ.get("LANGCHAIN_GATEWAY_NAMESPACE", "default"),
        )
    else:
        config = SandboxLocalTunnelConnectionConfig()
    return SandboxClient(connection_config=config)


@pytest.fixture
def backend():
    with AgentSandboxBackend.from_template(
        _client(),
        template_name=os.environ["LANGCHAIN_SANDBOX_TEMPLATE"],
        namespace=os.environ.get("LANGCHAIN_NAMESPACE", "default"),
        root_dir=os.environ.get("LANGCHAIN_ROOT_DIR", "/workspace"),
    ) as value:
        yield value


@pytest.mark.compile
def test_compile_imports() -> None:
    assert AgentSandboxBackend
    assert SandboxPolicyWrapper
    assert callable(create_sandbox_backend_factory)


@requires_sandbox
def test_execute_and_cwd(backend: AgentSandboxBackend) -> None:
    result = backend.execute("pwd")
    assert result.exit_code == 0
    assert os.environ.get("LANGCHAIN_ROOT_DIR", "/workspace") in result.output


@requires_sandbox
def test_file_lifecycle_and_search(backend: AgentSandboxBackend) -> None:
    assert backend.write("/integration.txt", "hello\nworld").error is None
    assert backend.read("/integration.txt").file_data["content"] == "hello\nworld"
    assert backend.edit("/integration.txt", "world", "sandbox").error is None
    assert backend.grep("sandbox", path="/").matches
    assert backend.glob("integration.txt", path="/").matches
    downloads = backend.download_files(["/integration.txt"])
    assert downloads[0].content == b"hello\nsandbox"


@requires_sandbox
def test_policy_wrapper_against_real_backend(backend: AgentSandboxBackend) -> None:
    wrapped = SandboxPolicyWrapper(backend, deny_commands=["rm -rf"])
    result = wrapped.execute("rm -rf /")
    assert result.exit_code == 1
