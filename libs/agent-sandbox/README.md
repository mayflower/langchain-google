# langchain-google-agent-sandbox

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-google-agent-sandbox?label=%20)](https://pypi.org/project/langchain-google-agent-sandbox/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-google-agent-sandbox)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-google-agent-sandbox)](https://pypistats.org/packages/langchain-google-agent-sandbox)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

`langchain-google-agent-sandbox` provides a DeepAgents backend for
Kubernetes `k8s-agent-sandbox` runtimes. It lets LangChain agents run tools in
Kubernetes-native sandboxes while keeping the package itself importable without
a Kubernetes cluster. The package implements the DeepAgents 0.7 backend
protocols and requires DeepAgents 0.7.0b2 or newer.

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

with AgentSandboxBackend.from_warm_pool(
    client,
    warm_pool="python-deepagent-pool",
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
    SessionAgentSandboxBackend,
    SessionLifecycleContext,
    SessionSandboxEndpoint,
    create_sandbox_backend,
    create_sandbox_backend_factory,
)
```

`create_sandbox_backend_factory` is a deprecated compatibility name. In
DeepAgents 0.7 it returns a concrete backend instance because callable backend
factories are no longer supported.

## Existing Sandboxes

Use `from_existing` when another component owns sandbox creation and cleanup.

```python
backend = AgentSandboxBackend.from_existing(existing_sandbox, root_dir="/workspace")
```

## Ephemeral Managed Sandboxes

Use `from_warm_pool` when this package should create a sandbox on enter and
delete it on exit. A `SandboxTemplate` defines the runtime; a
`SandboxWarmPool` is the allocation target used by the SDK. A WarmPool with
zero replicas provides cold allocation, while a positive replica count enables
pre-warming.

```python
with AgentSandboxBackend.from_warm_pool(
    client,
    warm_pool="python-deepagent-pool",
    namespace="default",
) as backend:
    print(backend.id)
```

`from_template` remains as a deprecated compatibility alias for
`from_warm_pool`.

## Durable Session Backend

Use one `SessionAgentSandboxBackend` instance directly in DeepAgents or as the
default of a `CompositeBackend`. It reads `configurable.thread_id` from the
active LangGraph config, HMACs the raw identity with a stable secret, and uses
the resulting opaque value for the deterministic Claim name and selector
label. Raw session values are never sent to Kubernetes.

```python
from deepagents import create_deep_agent
from k8s_agent_sandbox import SandboxClient
from langchain_google_agent_sandbox import SessionAgentSandboxBackend

backend = SessionAgentSandboxBackend(
    SandboxClient(cleanup=False),
    warm_pool="python-deepagent-pool",
    namespace="default",
    session_secret=os.environ["SANDBOX_SESSION_SECRET"],
    idle_ttl_seconds=3600,
    renewal_threshold_seconds=600,
    local_cache_ttl_seconds=300,
    max_cached_sessions=128,
)
agent = create_deep_agent(model=model, backend=backend)

agent.invoke(
    {"messages": [("user", "Create report.csv")]},
    config={"configurable": {"thread_id": "thread-123"}},
)
```

For tenant-aware applications, pass a resolver that combines tenant and thread
identity before HMAC encoding:

```python
def resolve_session(config):
    values = config["configurable"]
    return f"{values['tenant_id']}\0{values['thread_id']}"
```

The SDK and Kubernetes object uniqueness coordinate get-or-create across
replicas. Process-local locks only coalesce duplicate calls in one process.
Closing the backend, completing a graph, or exiting the process closes local
connectors and does not delete Claims. Use `delete_session(raw_session_id)` for
explicit deletion. The configured idle TTL is renewed near expiry, while the
agent-sandbox controller remains the authority that deletes abandoned Claims.
Idle local cache entries close only their process-local connector after
`local_cache_ttl_seconds`; a later operation reattaches to the existing Claim.
Set this option to `None` only when the application bounds its session set by
other means. `max_cached_sessions` additionally applies an LRU-style hard bound
to process-local connectors without deleting Claims.

Lifecycle hooks receive only the opaque session ID and the concrete backend:

- `on_session_created`: initialize a genuinely new Claim;
- `on_session_attached`: observe reattachment without restoring again;
- `before_session_deleted`: flush state before explicit deletion.

Consumers that need external admission or durable lifecycle bookkeeping can
use the generic hooks and their immutable `SessionLifecycleContext`:

```python
def admit(context: SessionLifecycleContext) -> None:
    if capacity_is_full(context.raw_session_id):
        raise CapacityUnavailableError("sandbox capacity is full")


backend = SessionAgentSandboxBackend(
    SandboxClient(cleanup=False),
    warm_pool="python-deepagent-pool",
    session_secret=os.environ["SANDBOX_SESSION_SECRET"],
    before_session_acquire=admit,
    after_session_acquire=lambda context, backend, created: record_claim(
        context.claim_name, created=created
    ),
    on_session_accessed=lambda context: record_access(context.opaque_session_id),
    after_session_deleted=lambda context: record_deletion(context.claim_name),
)
```

Lifecycle callbacks are synchronous. `before_session_acquire` may reject an
acquisition before any Kubernetes request is made. Error callbacks are
best-effort observers and never replace the original acquisition or deletion
exception. Keep `on_session_accessed` cheap and throttle durable writes in the
consumer. The legacy `on_session_created`, `on_session_attached`, and
`before_session_deleted` hooks retain their existing signatures and behavior.

Endpoint-aware integrations can call `get_session_sandbox()`,
`get_session_claim_name()`, and `get_session_endpoint(port,
prefer_pod_ip=False)`. An optional legacy-label fallback attaches only when
exactly one migrated random-name Claim matches; ambiguity fails closed.

## DeepAgents 0.7 Backend

```python
from deepagents import create_deep_agent
from k8s_agent_sandbox import SandboxClient
from langchain_google_agent_sandbox import create_sandbox_backend

with create_sandbox_backend(
    "python-deepagent-pool",
    client=SandboxClient(),
    root_dir="/workspace",
) as backend:
    agent = create_deep_agent(
        model=model,
        backend=backend,
    )
    result = agent.invoke(
        {"messages": [("user", "Create report.csv")]},
    )
```

The helper returns an initialized concrete backend, as required by DeepAgents
0.7. It creates an ephemeral managed backend and deletes a newly created Claim
when the backend is explicitly exited or finalized. Explicit exit detaches the
fallback finalizer. It is not the durable session API.

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
export LANGCHAIN_SANDBOX_WARM_POOL=python-deepagent-pool
export LANGCHAIN_NAMESPACE=default
export LANGCHAIN_API_URL=http://sandbox-router:8080
make integration_tests
```

Supported environment variables:

- `LANGCHAIN_SANDBOX_WARM_POOL`
- `LANGCHAIN_NAMESPACE`
- `LANGCHAIN_ROOT_DIR`
- `LANGCHAIN_SERVER_PORT`
- `LANGCHAIN_USE_TUNNEL=1`
- `LANGCHAIN_API_URL`
- `LANGCHAIN_GATEWAY_NAME`
- `LANGCHAIN_GATEWAY_NAMESPACE`

## Compatibility Notes

This package targets the complete DeepAgents 0.7 backend protocol family:
structured file operations, recursive deletion, paginated reads, bounded grep,
glob matching, uploads/downloads, and sandbox execution. It uses public v1beta1
`k8s-agent-sandbox` APIs only. Claim creation, atomic get-or-create, readiness,
validation, renewal, deletion, and endpoint metadata remain SDK
responsibilities. The adapter never issues raw `CustomObjectsApi` calls and has
no v1alpha1 fallback.

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
