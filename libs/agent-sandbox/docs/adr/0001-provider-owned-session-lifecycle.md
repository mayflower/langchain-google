# 1. Provider-owned session lifecycle

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

`libs/agent-sandbox` began as a DeepAgents backend for Kubernetes
`k8s-agent-sandbox` runtimes. To give agent threads a durable workspace, it
grew a second sandbox lifecycle control plane inside what is meant to be a
protocol adapter. That control plane:

- HMAC'd the LangGraph session identity with a shared secret and derived a
  deterministic Claim name from the digest;
- acquired that Claim through `get_or_create_sandbox`, an atomic
  acquire-by-name primitive;
- renewed the Claim's TTL through `renew_sandbox`;
- discovered "legacy" Claims by label selector, refusing ambiguous matches;
- deleted durable Claims through `delete_session()`;
- exposed nine lifecycle hooks so consumers could bolt their own admission,
  bookkeeping, and eventing onto it.

Neither `get_or_create_sandbox` nor `renew_sandbox` exists in the official
`k8s-agent-sandbox` SDK. Both were additions in a Mayflower fork, consumed via
a pinned Git dependency. The official `create_sandbox()` accepts no `name`
argument, so deterministic acquire-by-name is genuinely unavailable upstream —
the fork existed precisely to provide it.

That left three compounding problems:

1. **A fork on the critical path.** Every security fix, API change, and release
   of upstream `agent-sandbox` had to be re-forked and re-pinned by hand.
2. **Product policy in a protocol adapter.** Admission, tenancy, quota, and
   Claim ownership are product decisions. Encoding them here meant every
   consumer inherited one opinion about identity mapping, and the ones who
   disagreed reached for the hook surface to override it.
3. **Deletion authority in a DeepAgents backend.** `delete_session()` and
   `AgentSandboxBackend.delete_all()` could destroy durable user data from a
   component whose job is translating file operations.

## Decision

Durable session lifecycle is **owned by the consuming product** and supplied to
this package through an injected `SandboxSessionProvider`:

```python
class SandboxSessionProvider(Protocol):
    def acquire(self, config: Mapping[str, Any]) -> SandboxLease: ...
    def touch(self, lease: SandboxLease) -> None: ...
    def close_local(self, lease: SandboxLease) -> None: ...
```

The adapter retains exactly five responsibilities: DeepAgents protocol
translation, path normalization and bounded result mapping, a process-local
lease cache, reattachment after a stale local handle, and local connector
shutdown.

The protocol has **no `delete()` method**, deliberately. Explicit durable
deletion is a product control-plane operation and gets no route through a
DeepAgents backend.

With those responsibilities gone, the fork's two methods have no caller, and
the dependency becomes the exact upstream pin `k8s-agent-sandbox==0.5.4`.

## Consequences

**Good.**

- Upstream SDK and operator are consumed unchanged.
- Products map their own tenancy and admission, and can reject an acquisition
  by raising from `acquire()` before any Kubernetes request is made.
- No code path from this package deletes a durable Claim.
- The lease key is opaque to the adapter, so raw tenant or user identity need
  never reach a Kubernetes label.
- The boundary is mechanically enforced: a client-method allowlist derived from
  the installed SDK fails CI if a fork-only call returns.

**Costs.**

- Breaking change. `SessionAgentSandboxBackend` survives one release as a
  warning alias that rejects every removed argument by name rather than
  silently ignoring it. `delete_all`, `get_session_endpoint`,
  `delete_session`, and `from_warm_pool(session_id=...)` are gone.
- Every product wanting durable sessions must now implement a provider. There
  is no batteries-included durable path in this package, by design.
- Deterministic acquire-by-name still has to exist somewhere. This decision
  moves it behind the provider boundary rather than solving it; a product needs
  its own atomic acquisition, whether through a control-plane service, a
  database row, or an upstream primitive if one lands.

## Alternatives rejected

**Maintain `mayflower/agent-sandbox`.** Rejected. It puts a fork of a
fast-moving upstream on the critical path of every consumer, and the
maintenance is unbounded and permanent.

**Copy the fork's SDK patches into this repository.** Rejected. Vendoring
relocates the maintenance burden without reducing it, and hides the divergence
somewhere harder to notice than a Git pin.

**Let the DeepAgents adapter own product admission.** Rejected. Admission
depends on tenancy, quota, billing, and capacity — none of which a filesystem
protocol adapter can see. The hook surface that existed to work around this was
itself evidence the responsibility sat in the wrong place.

**Poll Pods and PVCs from the adapter.** Rejected. It requires a Kubernetes
client dependency and RBAC in every consuming process, and duplicates state the
operator already owns authoritatively.

**Continue private connector calls for streaming and cancellation.** Rejected.
It couples the package to SDK internals with no compatibility promise. Until
the public runtime API supports command IDs and cancellation, this package
offers bounded transfers and is explicit that cancelling an awaitable does not
cancel the remote process.

**Wait for an upstream DeepAgents integration.** Not available. No
`deepagents-k8s-agent-sandbox` package exists on PyPI, and the official
`k8s-agent-sandbox` wheel contains no DeepAgents integration. Migration to one
remains expected if it ever ships.
