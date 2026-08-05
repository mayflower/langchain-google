# DeepAgents Agent Sandbox Example

This example runs a DeepAgent using
`langchain-google-agent-sandbox` as the tool execution backend.

## Prerequisites

- A Kubernetes cluster with `k8s-agent-sandbox` installed
- A v1beta1 `SandboxTemplate` and `SandboxWarmPool`
- One model provider API key

```bash
pip install langchain-google-agent-sandbox
pip install langchain-google-genai
```

Optional model providers:

```bash
pip install langchain-anthropic langchain-openai
```

## Google Model

```bash
export GOOGLE_API_KEY=...
export GOOGLE_MODEL=gemini-3.5-flash
```

## Connection Modes

Local tunnel:

```bash
export LANGCHAIN_SANDBOX_WARM_POOL=python-deepagent-pool
export LANGCHAIN_USE_TUNNEL=1
python main.py --query "Create a script that prints hello"
```

Gateway:

```bash
python main.py \
  --gateway external-http-gateway \
  --gateway-namespace default \
  --warm-pool python-deepagent-pool
```

Direct API URL:

```bash
python main.py \
  --api-url http://sandbox-router:8080 \
  --warm-pool python-deepagent-pool
```

## Durable Sessions

Durable session lifecycle belongs to your product, not to this package. Supply
it through a `SandboxSessionProvider`: the adapter caches the lease you return
and never creates, renews, or deletes a Claim itself.

```python
from langchain_google_agent_sandbox import (
    ProviderSessionAgentSandboxBackend,
    SandboxLease,
)


class MyProvider:
    def acquire(self, config) -> SandboxLease:
        # Admission, tenancy, and Claim ownership live in your control plane.
        session = control_plane.claim_session(config["configurable"]["thread_id"])
        return SandboxLease(key=session.opaque_id, sandbox=session.sandbox)

    def touch(self, lease: SandboxLease) -> None:
        control_plane.record_activity(lease.key)

    def close_local(self, lease: SandboxLease) -> None:
        lease.sandbox.close_connection()


backend = ProviderSessionAgentSandboxBackend(MyProvider())
agent = create_deep_agent(model=model, backend=backend)
agent.invoke(
    {"messages": [("user", "Create data.csv")]},
    config={"configurable": {"thread_id": "thread-123"}},
)
```

See `docs/adr/0001-provider-owned-session-lifecycle.md` for why, and the
package README for the full migration table.

## Policy Wrapper

```python
from langchain_google_agent_sandbox import SandboxPolicyWrapper

backend = SandboxPolicyWrapper(
    backend,
    deny_prefixes=["/etc", "/proc", "/sys"],
    deny_commands=["rm -rf", "shutdown"],
)
```

`SandboxPolicyWrapper` is a best-effort application guardrail, not a security
boundary. Real isolation comes from the sandbox runtime, Kubernetes, container
runtime, and node configuration.

## Troubleshooting

- Missing WarmPool: check `kubectl get sandboxwarmpools -A`.
- Legacy session reattach refuses multiple Claims with the same opaque label.
- Model errors: confirm the selected provider package and API key are installed.
- Filesystem errors: make sure the runtime exposes writable `/workspace`.
