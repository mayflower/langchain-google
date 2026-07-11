"""Run a DeepAgent with a Kubernetes agent-sandbox backend."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from deepagents import create_deep_agent
from k8s_agent_sandbox import SandboxClient
from k8s_agent_sandbox.models import (
    SandboxDirectConnectionConfig,
    SandboxGatewayConnectionConfig,
    SandboxLocalTunnelConnectionConfig,
)

from langchain_google_agent_sandbox import AgentSandboxBackend


def get_model() -> Any:
    """Create a chat model from environment configuration."""
    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.environ.get("GOOGLE_MODEL", "gemini-3.5-flash")
        )
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        )
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    print("Set GOOGLE_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY.", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(
        description="Run DeepAgents with langchain-google-agent-sandbox"
    )
    parser.add_argument("--query", "-q")
    parser.add_argument(
        "--warm-pool",
        default=os.environ.get("LANGCHAIN_SANDBOX_WARM_POOL", "python-deepagent-pool"),
    )
    parser.add_argument(
        "--namespace", default=os.environ.get("LANGCHAIN_NAMESPACE", "default")
    )
    parser.add_argument(
        "--root-dir", default=os.environ.get("LANGCHAIN_ROOT_DIR", "/workspace")
    )
    parser.add_argument("--skills", nargs="*", default=[".deepagents/skills"])
    parser.add_argument("--gateway", default=os.environ.get("LANGCHAIN_GATEWAY_NAME"))
    parser.add_argument(
        "--gateway-namespace",
        default=os.environ.get("LANGCHAIN_GATEWAY_NAMESPACE", "default"),
    )
    parser.add_argument("--api-url", default=os.environ.get("LANGCHAIN_API_URL"))
    parser.add_argument(
        "--use-tunnel",
        action="store_true",
        default=os.environ.get("LANGCHAIN_USE_TUNNEL") == "1",
    )
    return parser.parse_args()


def create_client(args: argparse.Namespace) -> SandboxClient:
    """Create a SandboxClient for tunnel, gateway, or direct mode."""
    if args.api_url:
        config = SandboxDirectConnectionConfig(api_url=args.api_url)
    elif args.gateway:
        config = SandboxGatewayConnectionConfig(
            gateway_name=args.gateway,
            gateway_namespace=args.gateway_namespace,
        )
    else:
        _ = args.use_tunnel
        config = SandboxLocalTunnelConnectionConfig()
    return SandboxClient(connection_config=config)


def print_last_message(result: dict[str, Any]) -> None:
    """Print the last AI message from a DeepAgents response."""
    for message in reversed(result.get("messages", [])):
        if getattr(message, "type", None) == "ai":
            print(message.content)
            return


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    model = get_model()
    client = create_client(args)
    with AgentSandboxBackend.from_warm_pool(
        client,
        warm_pool=args.warm_pool,
        namespace=args.namespace,
        root_dir=args.root_dir,
    ) as backend:
        agent = create_deep_agent(model=model, backend=backend, skills=args.skills)
        if args.query:
            print_last_message(agent.invoke({"messages": [("user", args.query)]}))
            return
        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if query in {"exit", "quit"}:
                return
            if query:
                print_last_message(agent.invoke({"messages": [("user", query)]}))


if __name__ == "__main__":
    main()
