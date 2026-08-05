"""Contract tests against a fake built to the public Sandbox shape.

The fake mirrors ``k8s_agent_sandbox.sandbox.Sandbox`` as the official SDK
declares it, and nothing else. No private attribute, no connector, no
``k8s_helper``. If the adapter ever needs something outside this shape, these
tests fail at the point of contact rather than in a live cluster.

The shape itself is asserted against the installed SDK, so the fake cannot
quietly drift from the real thing.
"""

from __future__ import annotations

import inspect
import shlex
from typing import Any

import pytest
from k8s_agent_sandbox.sandbox import Sandbox

from langchain_google_agent_sandbox import AgentSandboxBackend


class FakeCommandResult:
    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class FakeCommands:
    """Public ``sandbox.commands`` surface."""

    def __init__(self, filesystem: dict[str, bytes]) -> None:
        self.files = filesystem
        self.commands: list[str] = []

    def run(self, command: str, timeout: int | None = None) -> FakeCommandResult:
        del timeout
        self.commands.append(command)
        if command.startswith("mkdir"):
            return FakeCommandResult()
        if " rm -rf " in command or command.startswith("rm -rf"):
            return FakeCommandResult()
        if "test -" in command or command.startswith("sh -c"):
            return FakeCommandResult(stdout="file")
        return FakeCommandResult(stdout="ok")


class FakeFiles:
    """Public ``sandbox.files`` surface."""

    def __init__(self, filesystem: dict[str, bytes]) -> None:
        self.files = filesystem

    def read(self, path: str) -> bytes:
        if path not in self.files:
            msg = f"no such file: {path}"
            raise FileNotFoundError(msg)
        return self.files[path]

    def write(self, path: str, content: bytes) -> None:
        self.files[path] = content

    def list(self, path: str) -> list[Any]:
        del path
        return []

    def exists(self, path: str) -> bool:
        return path in self.files


class FakeSandbox:
    """A sandbox exposing only what the official public API declares."""

    def __init__(self) -> None:
        self.filesystem: dict[str, bytes] = {}
        self.commands = FakeCommands(self.filesystem)
        self.files = FakeFiles(self.filesystem)
        self.claim_name = "claim-fake"
        self.namespace = "default"
        self.sandbox_id = "sandbox-fake"
        self.service_host = "sandbox-fake.default.svc.cluster.local"
        self.closed = False

    def close_connection(self) -> None:
        self.closed = True


@pytest.fixture
def backend() -> AgentSandboxBackend:
    return AgentSandboxBackend.from_existing(FakeSandbox())


def test_fake_matches_the_official_public_sandbox_shape() -> None:
    """The fake must not drift from the SDK it stands in for."""
    official = {
        name
        for name, _ in inspect.getmembers(Sandbox)
        if not name.startswith("_") and not isinstance(getattr(Sandbox, name), property)
    }
    fake = {name for name in dir(FakeSandbox) if not name.startswith("_")}
    # Everything the fake exposes as a method must exist upstream.
    fake_methods = {name for name in fake if callable(getattr(FakeSandbox, name, None))}
    assert fake_methods <= official | {"commands", "files"}, (
        f"fake exposes methods the official Sandbox lacks: {fake_methods - official}"
    )
    assert "close_connection" in official


def test_backend_touches_no_private_sandbox_attribute() -> None:
    """Any private access would raise on this fake rather than silently work."""

    class StrictSandbox(FakeSandbox):
        def __getattr__(self, name: str) -> Any:
            msg = f"adapter reached for a non-public attribute: {name!r}"
            raise AssertionError(msg)

    backend = AgentSandboxBackend.from_existing(StrictSandbox())
    backend.write("/a.txt", "hello")
    backend.read("/a.txt")
    backend.execute("echo hi")
    backend.close()


def test_write_then_read_roundtrip(backend: AgentSandboxBackend) -> None:
    assert backend.write("/notes.txt", "hello").error is None
    result = backend.read("/notes.txt")
    assert result.file_data["content"] == "hello"


def test_write_addresses_the_sdk_with_runtime_relative_paths(
    backend: AgentSandboxBackend,
) -> None:
    """The public file API takes paths relative to runtime_root, not absolute."""
    backend.write("/notes.txt", "hello")
    assert list(backend._sandbox.filesystem) == ["notes.txt"]


def test_read_missing_file_reports_an_error(backend: AgentSandboxBackend) -> None:
    assert backend.read("/absent.txt").error is not None


def test_edit_replaces_content(backend: AgentSandboxBackend) -> None:
    backend.write("/f.txt", "alpha beta")
    result = backend.edit("/f.txt", "alpha", "gamma")
    assert result.error is None
    assert backend.read("/f.txt").file_data["content"] == "gamma beta"


def test_execute_runs_inside_the_virtual_root(backend: AgentSandboxBackend) -> None:
    backend.execute("ls")
    issued = backend._sandbox.commands.commands[-1]
    assert "/workspace" in issued


def test_close_uses_the_public_close_connection(
    backend: AgentSandboxBackend,
) -> None:
    sandbox = backend._sandbox
    backend.close()
    assert sandbox.closed is True


@pytest.mark.parametrize(
    "hostile",
    [
        "../etc/passwd",
        "../../etc/passwd",
        "/../etc/passwd",
        "subdir/../../../etc/passwd",
    ],
)
def test_path_traversal_cannot_escape_the_virtual_root(
    backend: AgentSandboxBackend, hostile: str
) -> None:
    """Traversal is contained, never escaped.

    A hostile path is either refused or normalized to somewhere *inside* the
    root -- `/../etc/passwd` becomes `etc/passwd` under `/workspace`, not the
    host's `/etc/passwd`. What matters is that no path reaching the SDK is
    absolute or climbs out with `..`.
    """
    backend.write(hostile, "owned")
    backend.read(hostile)

    for path in backend._sandbox.filesystem:
        assert not path.startswith("/"), f"escaped to an absolute path: {path}"
        assert not path.startswith(".."), f"climbed out of the root: {path}"
        assert ".." not in path.split("/"), f"unresolved traversal survived: {path}"


@pytest.mark.parametrize(
    "hostile",
    [
        "/tmp/x; rm -rf /",
        "/tmp/$(whoami)",
        "/tmp/`id`",
        "/tmp/x && cat /etc/shadow",
        "/tmp/x'\"$(id)\"'",
    ],
)
def test_shell_metacharacters_in_paths_are_quoted(
    backend: AgentSandboxBackend, hostile: str
) -> None:
    """Paths reaching a shell must be quoted, never interpolated raw.

    Lexing each issued command must never yield a bare shell operator: if a
    hostile path had been interpolated unquoted, its `;` or `&&` would surface
    here as its own token, meaning the sandbox would run a second command.
    """
    operators = {";", "&&", "||", "|", ">", ">>", "<", "&"}
    backend.delete(hostile)

    issued_commands = backend._sandbox.commands.commands
    assert issued_commands, "delete issued no command to inspect"
    for issued in issued_commands:
        tokens = shlex.split(issued)
        leaked = operators.intersection(tokens)
        assert not leaked, f"unquoted shell operator {leaked} in: {issued}"


@pytest.mark.parametrize(
    "hostile",
    [
        "needle'; rm -rf /; echo '",
        "$(id)",
        "`whoami`",
        "a && cat /etc/shadow",
        'x" ; touch /tmp/pwned #',
    ],
)
def test_shell_metacharacters_in_grep_patterns_are_quoted(
    backend: AgentSandboxBackend, hostile: str
) -> None:
    """A search pattern is data, not shell syntax.

    Patterns come straight from the model, so an unquoted one would let a
    search turn into arbitrary command execution.
    """
    operators = {";", "&&", "||", "|", ">", ">>", "<", "&"}
    backend.grep(hostile)

    issued_commands = backend._sandbox.commands.commands
    assert issued_commands, "grep issued no command to inspect"
    for issued in issued_commands:
        leaked = operators.intersection(shlex.split(issued))
        assert not leaked, f"unquoted shell operator {leaked} in: {issued}"


def test_glob_patterns_never_reach_the_shell(backend: AgentSandboxBackend) -> None:
    """Glob matching happens in Python; only the base path is interpolated."""
    backend.glob("*.py; rm -rf /")
    for issued in backend._sandbox.commands.commands:
        assert "rm -rf /" not in issued


def test_execute_quotes_the_command_it_wraps(backend: AgentSandboxBackend) -> None:
    backend.execute("echo 'hi there'; rm -rf /")
    issued = backend._sandbox.commands.commands[-1]
    # The injected fragment must be inside a quoted argument, not a new command.
    tokens = shlex.split(issued)
    assert tokens[0] == "sh"
    assert tokens[1] == "-c"
    assert len(tokens) == 3


def test_binary_content_round_trips_as_base64(backend: AgentSandboxBackend) -> None:
    backend._sandbox.filesystem["blob.bin"] = b"\xff\xfe\x00"
    result = backend.read("/blob.bin")
    assert result.file_data["encoding"] == "base64"


def test_invalid_utf8_in_a_text_file_falls_back_to_base64(
    backend: AgentSandboxBackend,
) -> None:
    backend._sandbox.filesystem["broken.txt"] = b"\xff\xfe"
    result = backend.read("/broken.txt")
    assert result.file_data["encoding"] == "base64"


def test_utf8_content_round_trips(backend: AgentSandboxBackend) -> None:
    backend.write("/u.txt", "grüße 😀")
    assert backend.read("/u.txt").file_data["content"] == "grüße 😀"
