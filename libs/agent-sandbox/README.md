# langchain-google-agent-sandbox

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-google-agent-sandbox?label=%20)](https://pypi.org/project/langchain-google-agent-sandbox/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-google-agent-sandbox)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-google-agent-sandbox)](https://pypistats.org/packages/langchain-google-agent-sandbox)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

`langchain-google-agent-sandbox` provides a DeepAgents backend for
Kubernetes `k8s-agent-sandbox` runtimes. It lets LangChain agents run tools in
Kubernetes-native sandboxes while keeping the package itself importable without
a Kubernetes cluster. It implements the DeepAgents 0.7 backend protocols and
requires `deepagents>=0.7.4` and `k8s-agent-sandbox==0.5.4`.

## What this package is, and is not

**This is a DeepAgents protocol adapter, not a sandbox control plane.**

It owns exactly five things: protocol translation, path normalization and
bounded result mapping, a process-local session cache, reattachment after a
stale local handle, and local connector shutdown.

It does **not** decide product admission, derive Kubernetes Claim ownership,
renew leases, delete durable sessions, inspect Pods or PVCs, or use private SDK
internals. Durable session lifecycle is **owned by the consuming product** and
supplied through a [`SandboxSessionProvider`](#durable-session-backend).

A few consequences worth stating plainly:

- The official `k8s-agent-sandbox` SDK and the agent-sandbox operator are used
  **unchanged**. No fork, no vendored files, no import-time patching.
- **Closing an adapter never deletes durable infrastructure.** Eviction and
  shutdown release local client resources only.
- Do not put raw tenant or user identity in Claim labels. Identity mapping
  belongs to the provider, which should hand this package an already-opaque
  lease key.
- **Strong isolation comes from the runtime, operator, and Kubernetes policy**,
  not from this package. The path virtualization here is ergonomics, not a
  security boundary.
- **Cancelling an async call does not cancel the remote process.** The async
  methods wrap synchronous SDK calls; abandoning the awaitable abandons the
  local wait, and the command keeps running in the sandbox.
- **Large-file streaming is not implemented.** The bounded upload/download
  operations are the supported transfer path; move large artifacts through a
  separate service boundary.
- Migration to an eventual upstream DeepAgents integration is expected. None
  exists on PyPI today.

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
    ProviderSessionAgentSandboxBackend,
    SandboxLease,
    SandboxPolicyWrapper,
    SandboxResultLimits,
    SandboxSessionProvider,
    create_sandbox_backend,
    create_sandbox_backend_factory,
)
```

`create_sandbox_backend_factory` is a deprecated compatibility name. In
DeepAgents 0.7 it returns a concrete backend instance because callable backend
factories are no longer supported.

`SessionAgentSandboxBackend` remains importable as a deprecated alias for
`ProviderSessionAgentSandboxBackend` for one migration release. It warns on
construction and **does not** preserve the old lifecycle behavior; see
[Migration](#migration-from-the-pre-provider-api).

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

!!! warning "Ephemeral only"
    `from_warm_pool` creates an SDK-named Claim and **deletes it on context
    exit**. It is unsuitable for durable thread persistence. It also no longer
    accepts `session_id`: adopting an existing Claim by label was a
    control-plane decision about which persistent filesystem an agent receives,
    and that decision now belongs to the provider.

## Durable Session Backend

Durable sessions are supplied by **your product**, through a
`SandboxSessionProvider`. This package caches the lease you return and
translates DeepAgents calls onto it. It never creates, renames, labels, renews,
lists, or deletes a Claim, and never derives a Claim name or hashes identity.

```python
from collections.abc import Mapping
from typing import Any

from langchain_google_agent_sandbox import SandboxLease


class MyControlPlaneProvider:
    """Talks to the product's own sandbox control-plane service."""

    def acquire(self, config: Mapping[str, Any]) -> SandboxLease:
        thread_id = config["configurable"]["thread_id"]
        # Admission, tenant mapping, quota, and Claim ownership all live here.
        session = control_plane.claim_session(thread_id)
        return SandboxLease(
            key=session.opaque_id,
            sandbox=session.sandbox,
            claim_name=session.claim_name,
            namespace=session.namespace,
        )

    def touch(self, lease: SandboxLease) -> None:
        control_plane.record_activity(lease.key)

    def close_local(self, lease: SandboxLease) -> None:
        lease.sandbox.close_connection()
```

Wire it in like any other backend:

```python
from deepagents import create_deep_agent
from langchain_google_agent_sandbox import ProviderSessionAgentSandboxBackend

backend = ProviderSessionAgentSandboxBackend(
    MyControlPlaneProvider(),
    local_cache_ttl_seconds=300,
    max_cached_sessions=128,
    touch_interval_seconds=60,
)
agent = create_deep_agent(model=model, backend=backend)

agent.invoke(
    {"messages": [("user", "Create report.csv")]},
    config={"configurable": {"thread_id": "thread-123"}},
)
```

### What the adapter guarantees

- `acquire()` is called **only** when no usable process-local lease is cached.
  Concurrent first use of one session coalesces into a single call.
- Two different sessions may never share one lease. If a provider returns a
  lease key already held by another session, acquisition is refused rather than
  serving one agent another agent's filesystem.
- `touch()` is throttled to at most one call per `touch_interval_seconds`. It is
  never called once per file operation, and a failure is logged without failing
  the agent's operation.
- On `SandboxNotFoundError` the stale lease is dropped and re-acquired **exactly
  once**. A second failure propagates; there is no retry loop.
- `SandboxClaimFailedError` is terminal. It reports a `Ready=False` reason the
  controller will not retry, so it propagates without a re-acquire.
- Eviction (`local_cache_ttl_seconds`, `max_cached_sessions`) and `close()` call
  `close_local()` only. **No code path here deletes a Claim.**
- Provider errors propagate unchanged, so an admission refusal reaches the
  caller with the product's own message.

### Session identity

By default the cache key is `configurable.thread_id`. Pass `session_resolver`
to scope it differently:

```python
backend = ProviderSessionAgentSandboxBackend(
    provider,
    session_resolver=lambda config: (
        f"{config['configurable']['tenant_id']}/{config['configurable']['thread_id']}"
    ),
)
```

This value is a **process-local cache key only**. It is never sent to
Kubernetes and never turned into a Claim name. Mapping identity to durable
infrastructure is the provider's job, and the provider should return an already
opaque `SandboxLease.key`.

### Bounding results

Every unbounded result is capped and reported through the protocol's own
truncation indicators, never silently dropped:

```python
from langchain_google_agent_sandbox import SandboxResultLimits

limits = SandboxResultLimits(
    execute_output_bytes=1_048_576,
    read_lines=10_000,
    grep_matches=1_000,
    glob_matches=1_000,
    upload_files=100,
    download_files=100,
)
backend = ProviderSessionAgentSandboxBackend(provider, limits=limits)
```

A capped `read` reports the remainder via `next_offset`; `grep` and `glob` set
`truncated`; `execute` sets `truncated` and clips on the UTF-8 encoding so a
bound landing mid-character cannot produce invalid text. Upload and download
return one response per requested file, with a `limit_exceeded` error for those
past the bound.

## Migration from the pre-provider API

The previous `SessionAgentSandboxBackend` owned session lifecycle directly.
That responsibility moved to the provider. Removed arguments are **rejected
with a `TypeError`** naming their replacement rather than being silently
ignored:

| Removed | Replacement |
|---|---|
| `client=` | product `SandboxSessionProvider` |
| `warm_pool=` | provider/control-plane configuration |
| `session_secret=` | provider-owned identity mapping |
| `namespace=` | provider/control-plane configuration |
| `sandbox_ready_timeout=`, `labels=` | provider/control-plane configuration |
| `idle_ttl_seconds=`, `renewal_threshold_seconds=` | provider-owned lease lifetime and renewal |
| `legacy_label_fallback=` | removed; the provider owns session discovery |
| `before_session_acquire=` | provider admission (raise from `acquire()`) |
| `after_session_acquire=`, `on_session_created=`, `on_session_attached=` | provider workspace preparation/eventing |
| `on_session_accessed=` | throttled provider `touch()` |
| `before_session_deleted=`, `after_session_deleted=` | product control-plane close operation |
| `on_session_acquire_error=`, `on_session_delete_error=` | provider error handling |
| `delete_session()` | product control-plane close operation |
| `get_session_endpoint()`, `get_session_claim_name()`, `get_session_sandbox()` | `get_session_lease()`, or a separate endpoint API |
| `opaque_session_id()` | `SandboxLease.key`, chosen by the provider |
| `AgentSandboxBackend.delete_all()` | product control-plane operation |
| `AgentSandboxBackend.from_warm_pool(session_id=...)` | provider-owned session discovery |

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
glob matching, uploads/downloads, and sandbox execution.

It calls **only public** `k8s-agent-sandbox` APIs, and only methods that exist
on the official `SandboxClient`. Claim readiness, validation, and endpoint
metadata remain SDK responsibilities; Claim ownership, renewal, and deletion
are the provider's. The adapter never issues raw `CustomObjectsApi` calls,
never touches `k8s_helper` or a connector, and declares no Kubernetes client
dependency of its own.

Three fences enforce this in CI: a string provenance guard, an AST import
fence, and a client-method allowlist derived by introspecting the installed
SDK. `scripts/assert_upstream_agent_sandbox.py` additionally fails the build if
the dependency drifts off the exact PyPI pin or reintroduces a direct
reference.

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
