"""Architecture fences keeping this package a protocol adapter.

Three independent guards, because no one of them is sufficient:

1. a string provenance guard, catching a fork coordinate reappearing;
2. an AST import fence, catching a Kubernetes client sneaking back in;
3. a client-method allowlist **derived from the installed SDK**, which is the
   only check that would have caught the original fork dependency -- the fork
   was reached through ordinary-looking public calls, not private internals.

The allowlist is derived rather than hardcoded on purpose. A hardcoded list
silently rots on the next SDK bump; a derived one fails loudly and correctly
when upstream removes a method this adapter depends on.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from k8s_agent_sandbox.sandbox_client import SandboxClient

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PACKAGE_ROOT / "langchain_google_agent_sandbox"
SOURCE_FILES = sorted(SOURCE_ROOT.rglob("*.py"))

#: Modules production code must never import. A Kubernetes client of its own
#: would mean this package had started addressing the API server directly.
FORBIDDEN_IMPORTS = {
    "kubernetes",
    "kubernetes_asyncio",
    "k8s_agent_sandbox.k8s_helper",
    "k8s_agent_sandbox.async_k8s_helper",
}

#: Attribute chains that reach past the SDK's public surface.
FORBIDDEN_ATTRIBUTES = {
    "k8s_helper",
    "custom_objects_api",
    "_core_v1_api",
    "_active_connection_sandboxes",
    "send_request",
}

#: Fork coordinates that must never reappear anywhere in the package. Kept
#: identical to FORBIDDEN_STRINGS in scripts/assert_upstream_agent_sandbox.py;
#: an earlier version of this list omitted the bare form and was consequently
#: weaker than the build guard.
FORK_COORDINATES = (
    "github.com/mayflower/agent-sandbox",
    "mayflower/agent-sandbox",
    "a2419b9b7eaeec99f636c46a46ca55731d3f52fc",
)

#: Files permitted to name the coordinates, because naming them is their job:
#: two guards that forbid them and the ADR that records why they were dropped.
#: Anything else mentioning a fork coordinate is a regression.
HISTORICAL_REFERENCES = {
    "assert_upstream_agent_sandbox.py",
    "test_architecture.py",
    "0001-provider-owned-session-lifecycle.md",
}

#: Methods the fork added and upstream never had. Their return would mean the
#: adapter had taken durable lifecycle back from the provider.
FORK_ONLY_METHODS = ("get_or_create_sandbox", "renew_sandbox")

SCAN_SUFFIXES = {".py", ".toml", ".lock", ".md"}
SKIP_DIRS = {".venv", "dist", "__pycache__", ".pytest_cache", ".ruff_cache"}


def official_client_methods() -> set[str]:
    """Public method names on the installed official ``SandboxClient``."""
    return {
        name
        for name, _ in inspect.getmembers(SandboxClient, predicate=inspect.isfunction)
        if not name.startswith("_")
    }


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_source_files_were_found() -> None:
    """Guard the guards: an empty file list would make every fence vacuous."""
    assert len(SOURCE_FILES) >= 5


# --- 1. string provenance ----------------------------------------------------


def test_no_fork_coordinates_anywhere_in_the_package() -> None:
    offenders = []
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in HISTORICAL_REFERENCES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        offenders.extend(
            f"{path.relative_to(PACKAGE_ROOT)}: {needle}"
            for needle in FORK_COORDINATES
            if needle in text
        )
    assert offenders == []


# --- 2. AST import fence -----------------------------------------------------


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_forbidden_imports(path: Path) -> None:
    imported: set[str] = set()
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for name in imported:
        root = name.split(".")[0]
        assert root not in {"kubernetes", "kubernetes_asyncio"}, (
            f"{path.name} imports {name}: the adapter must own no Kubernetes client"
        )
        assert name not in FORBIDDEN_IMPORTS, f"{path.name} imports {name}"


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_private_sdk_attribute_access(path: Path) -> None:
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            msg = f"{path.name}:{node.lineno} reaches SDK internals via .{node.attr}"
            raise AssertionError(msg)


# --- 3. derived client-method allowlist --------------------------------------


def client_calls_in_source() -> dict[str, list[str]]:
    """Map each ``*client.<method>(...)`` call site to its method name."""
    calls: dict[str, list[str]] = {}
    for path in SOURCE_FILES:
        for node in ast.walk(parse(path)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            target = func.value
            if not isinstance(target, ast.Name | ast.Attribute):
                continue
            name = target.id if isinstance(target, ast.Name) else target.attr
            if not name.endswith("client"):
                continue
            calls.setdefault(func.attr, []).append(f"{path.name}:{node.lineno}")
    return calls


def test_every_client_call_exists_on_the_official_sdk() -> None:
    """The load-bearing fence: no call may depend on a fork-only method."""
    allowed = official_client_methods()
    assert allowed, "failed to introspect the installed SandboxClient"

    used = client_calls_in_source()
    unknown = {method: sites for method, sites in used.items() if method not in allowed}
    assert unknown == {}, (
        f"these client calls do not exist on the installed official "
        f"SandboxClient: {unknown}"
    )


@pytest.mark.parametrize("method", FORK_ONLY_METHODS)
def test_fork_only_methods_are_absent_from_sdk_and_source(method: str) -> None:
    assert not hasattr(SandboxClient, method), (
        f"{method} is present on the installed SandboxClient: "
        "this is not the official distribution"
    )
    assert method not in client_calls_in_source()


def test_allowlist_would_reject_a_fork_only_call() -> None:
    """Prove the fence bites rather than merely passing."""
    allowed = official_client_methods()
    for method in FORK_ONLY_METHODS:
        assert method not in allowed
