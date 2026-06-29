# langchain-google-agent-sandbox

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-google-agent-sandbox?label=%20)](https://pypi.org/project/langchain-google-agent-sandbox/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-google-agent-sandbox)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-google-agent-sandbox)](https://pypistats.org/packages/langchain-google-agent-sandbox)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

`langchain-google-agent-sandbox` provides a DeepAgents backend for
Kubernetes `k8s-agent-sandbox` runtimes. It lets LangChain agents run tools in
Kubernetes-native sandboxes while keeping the package itself importable without
a Kubernetes cluster.

## Quick Install

```bash
pip install langchain-google-agent-sandbox
```

Kubernetes is not required just to import the package. A configured
`k8s-agent-sandbox` runtime is only required when you create or attach to a
sandbox.

## Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/integrations/langchain_google_agent_sandbox/). For conceptual guides, tutorials, and examples on using Google integrations, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/google).

## Quickstart

```python
from k8s_agent_sandbox import SandboxClient
from langchain_google_agent_sandbox import AgentSandboxBackend

client = SandboxClient()

with AgentSandboxBackend.from_template(
    client,
    template_name="python-deepagent",
    namespace="default",
    root_dir="/workspace",
) as backend:
    result = backend.execute("python --version")
    print(result.output)
```

## Public API

```python
from langchain_google_agent_sandbox import (
    AgentSandboxBackend,
    SandboxPolicyWrapper,
    create_sandbox_backend_factory,
)
```

## Existing Sandboxes

Use `from_existing` when another component owns sandbox creation and cleanup.

```python
backend = AgentSandboxBackend.from_existing(existing_sandbox, root_dir="/workspace")
```

## Template Managed Sandboxes

Use `from_template` when this package should create a sandbox on enter and
delete it on exit.

```python
with AgentSandboxBackend.from_template(
    client,
    template_name="python-deepagent",
    namespace="default",
    session_id="thread-123",
) as backend:
    print(backend.id)
```

When `session_id` is supplied, the backend looks for exactly one
`SandboxClaim` with the session label. Multiple matches are refused instead of
being resolved arbitrarily.

## DeepAgents Factory

```python
from deepagents import create_deep_agent
from k8s_agent_sandbox import SandboxClient
from langchain_google_agent_sandbox import create_sandbox_backend_factory

agent = create_deep_agent(
    model=model,
    backend=create_sandbox_backend_factory(
        "python-deepagent",
        client=SandboxClient(),
        root_dir="/workspace",
        session_id="thread-123",
    ),
)
```

## Policy Wrapper

`SandboxPolicyWrapper` is a best-effort application guardrail, not a security
boundary. It can deny writes under selected prefixes, deny command substrings,
and emit audit callbacks.

```python
from langchain_google_agent_sandbox import SandboxPolicyWrapper

wrapped = SandboxPolicyWrapper(
    backend,
    deny_prefixes=["/etc", "/proc", "/sys"],
    deny_commands=["rm -rf", "shutdown"],
    strict_audit=False,
)
```

Real isolation comes from the sandbox runtime, container isolation, Kubernetes
policy, and node configuration. Lexical path checks do not resolve symlinks.

## Connection Modes

Connection mode is configured on `k8s_agent_sandbox.SandboxClient`, not on this
backend. The examples cover local tunnel, gateway, and direct API URL modes.

## Sandbox Runtime Requirements

The runtime should provide:

- `sh`, `grep`, `find`, `mkdir`, `test`, `base64`
- API endpoints compatible with released `k8s-agent-sandbox`
- A writable `/workspace` directory by default

The reference runtime in `examples/deepagents-runtime` documents the expected
HTTP surface: `/execute`, `/upload`, `/download`, `/list`, and `/exists`.

## Integration Tests

Integration tests skip unless explicitly configured.

```bash
export LANGCHAIN_SANDBOX_TEMPLATE=python-deepagent
export LANGCHAIN_NAMESPACE=default
export LANGCHAIN_API_URL=http://sandbox-router:8080
make integration_tests
```

Supported environment variables:

- `LANGCHAIN_SANDBOX_TEMPLATE`
- `LANGCHAIN_NAMESPACE`
- `LANGCHAIN_ROOT_DIR`
- `LANGCHAIN_SERVER_PORT`
- `LANGCHAIN_USE_TUNNEL=1`
- `LANGCHAIN_API_URL`
- `LANGCHAIN_GATEWAY_NAME`
- `LANGCHAIN_GATEWAY_NAMESPACE`

## Compatibility Notes

This package uses released/public `k8s-agent-sandbox` APIs only. Optional SDK
conveniences such as labels, `shutdown_after_seconds`, claim listing, and
template validation are routed through local compatibility helpers. If a
released SDK lacks a required feature, the backend raises a precise
`RuntimeError` instead of passing unsupported keyword arguments blindly.

No branch-based or local editable dependency is required.

## Troubleshooting

- If session reattach fails with multiple claims, delete stale claims for that
  session label before retrying.
- If a write outside `/workspace` fails, check whether
  `allow_absolute_paths=True` was intentionally enabled and whether the runtime
  user has filesystem permission.

## Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
