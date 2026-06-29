# DeepAgents Agent Sandbox Example

This example runs a DeepAgent using
`langchain-google-agent-sandbox` as the tool execution backend.

## Prerequisites

- A Kubernetes cluster with `k8s-agent-sandbox` installed
- A sandbox template such as `python-deepagent`
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
export LANGCHAIN_SANDBOX_TEMPLATE=python-deepagent
export LANGCHAIN_USE_TUNNEL=1
python main.py --query "Create a script that prints hello"
```

Gateway:

```bash
python main.py \
  --gateway external-http-gateway \
  --gateway-namespace default \
  --template python-deepagent
```

Direct API URL:

```bash
python main.py \
  --api-url http://sandbox-router:8080 \
  --template python-deepagent
```

## Session Reattach

```bash
python main.py --session-id thread-123 --query "Create data.csv"
python main.py --session-id thread-123 --query "Read data.csv"
```

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

- Missing template: check `kubectl get sandboxtemplates -A`.
- Session reattach ambiguity: delete stale claims with the same session label.
- Model errors: confirm the selected provider package and API key are installed.
- Filesystem errors: make sure the runtime exposes writable `/workspace`.
