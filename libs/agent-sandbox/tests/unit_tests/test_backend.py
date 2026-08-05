from __future__ import annotations

import shlex
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from deepagents.backends.protocol import DeleteResult
from deepagents.middleware.filesystem import FilesystemMiddleware
from k8s_agent_sandbox.exceptions import SandboxNotFoundError

from langchain_google_agent_sandbox import (
    AgentSandboxBackend,
    SandboxPolicyWrapper,
    _compat,
    create_sandbox_backend,
    create_sandbox_backend_factory,
)
from langchain_google_agent_sandbox._paths import (
    compile_glob,
    compile_grep_include_glob,
)


class FileEntry:
    def __init__(
        self,
        name: str,
        *,
        type: str = "file",
        size: int | None = 0,
        mod_time: float | None = 1_700_000_000,
    ) -> None:
        self.name = name
        self.type = type
        self.size = size
        self.mod_time = mod_time


class StubCommands:
    def __init__(self, results: list[Any] | None = None) -> None:
        self.results = results or [
            SimpleNamespace(stdout="", stderr="", exit_code=0),
        ]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, command: str, **kwargs: Any) -> Any:
        self.calls.append((command, kwargs))
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return SimpleNamespace(stdout="", stderr="", exit_code=0)


class StubFiles:
    def __init__(
        self, *, read: bytes | str = b"", entries: list[FileEntry] | None = None
    ) -> None:
        self.read_value = read
        self.entries = entries or []
        self.exists_value = False
        self.write_calls: list[tuple[str, bytes]] = []
        self.read_calls: list[str] = []
        self.list_calls: list[str] = []
        self.exists_calls: list[str] = []

    def read(self, path: str) -> bytes | str:
        self.read_calls.append(path)
        if isinstance(self.read_value, BaseException):
            raise self.read_value
        return self.read_value

    def list(self, path: str) -> list[FileEntry]:
        self.list_calls.append(path)
        return self.entries

    def write(self, path: str, content: bytes) -> None:
        self.write_calls.append((path, content))

    def exists(self, path: str) -> bool:
        self.exists_calls.append(path)
        return self.exists_value


class StubSandbox:
    def __init__(
        self,
        *,
        commands: StubCommands | None = None,
        files: StubFiles | None = None,
        claim_name: str | None = "claim-1",
        namespace: str = "default",
    ) -> None:
        self.commands = commands or StubCommands()
        self.files = files or StubFiles()
        self.claim_name = claim_name
        self.namespace = namespace
        self.sandbox_id = claim_name or "sandbox-id"
        self.service_host = f"{self.sandbox_id}.{namespace}.svc.cluster.local"
        self.closed = False

    def get_pod_ip(self) -> str:
        return "10.0.0.8"

    def close_connection(self) -> None:
        self.closed = True


class StubClient:
    def __init__(self, claims: list[str] | None = None) -> None:
        self.claims = claims or []
        self.created: list[dict[str, Any]] = []
        self.deleted: list[tuple[str, str]] = []
        self.listed: list[tuple[str, str | None]] = []
        self.sandbox = StubSandbox()

    def create_sandbox(
        self,
        warmpool: str,
        namespace: str = "default",
        sandbox_ready_timeout: int = 180,
        labels: dict[str, str] | None = None,
        *,
        shutdown_after_seconds: int | None = None,
    ) -> StubSandbox:
        self.created.append(
            {
                "warmpool": warmpool,
                "namespace": namespace,
                "sandbox_ready_timeout": sandbox_ready_timeout,
                "labels": labels,
                "shutdown_after_seconds": shutdown_after_seconds,
            }
        )
        self.sandbox.namespace = namespace
        return self.sandbox

    def list_all_sandboxes(
        self, namespace: str = "default", label_selector: str | None = None
    ) -> list[str]:
        self.list_namespace = namespace
        self.list_selector = label_selector
        self.listed.append((namespace, label_selector))
        return self.claims

    def get_sandbox(self, claim_name: str, namespace: str = "default") -> StubSandbox:
        self.sandbox.claim_name = claim_name
        self.sandbox.namespace = namespace
        return self.sandbox

    def get_sandbox_claim_warmpool_name(self, claim_name: str, namespace: str) -> str:
        return "python"

    def delete_sandbox(self, claim_name: str, namespace: str = "default") -> None:
        self.deleted.append((claim_name, namespace))


def result(stdout: str = "", stderr: str = "", exit_code: int = 0) -> Any:
    return SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=exit_code)


def test_public_imports() -> None:
    assert AgentSandboxBackend
    assert SandboxPolicyWrapper
    assert callable(create_sandbox_backend)
    assert callable(create_sandbox_backend_factory)


def test_execute_combines_streams_and_passes_timeout() -> None:
    sandbox = StubSandbox(commands=StubCommands([result("out", "err", 7)]))
    backend = AgentSandboxBackend.from_existing(sandbox)

    response = backend.execute("echo ok", timeout=9)

    assert response.output == "out\nerr"
    assert response.exit_code == 7
    assert sandbox.commands.calls[0][1]["timeout"] == 9
    assert "cd /workspace" in sandbox.commands.calls[0][0]


def test_execute_quotes_custom_root_dir() -> None:
    sandbox = StubSandbox(commands=StubCommands([result("ok")]))
    backend = AgentSandboxBackend.from_existing(sandbox, root_dir="/tmp/my root")

    response = backend.execute("echo ok")

    assert response.output == "ok"
    assert shlex.split(sandbox.commands.calls[0][0]) == [
        "sh",
        "-c",
        "cd '/tmp/my root' && echo ok",
    ]


def test_from_existing_exposes_runtime_root_and_default_timeout() -> None:
    sandbox = StubSandbox(commands=StubCommands([result("ok")]))
    backend = AgentSandboxBackend.from_existing(
        sandbox,
        root_dir="/app/workspace",
        runtime_root="/app",
        default_timeout_seconds=17,
    )
    backend.execute("pwd")
    assert sandbox.commands.calls[0][1]["timeout"] == 17
    assert backend._runtime_root == "/app"


def test_execute_uses_default_timeout_and_classifies_errors() -> None:
    timeout_backend = AgentSandboxBackend(
        StubSandbox(commands=StubCommands([TimeoutError("slow")])),
        _default_timeout_seconds=5,
    )
    assert timeout_backend.execute("sleep 99").exit_code == -2

    generic_backend = AgentSandboxBackend(
        StubSandbox(commands=StubCommands([RuntimeError("boom")])),
    )
    assert generic_backend.execute("false").exit_code == -1


def test_ls_sorts_filters_and_maps_metadata() -> None:
    files = StubFiles(
        entries=[
            FileEntry("z.txt", size=2),
            FileEntry(".", type="directory"),
            FileEntry("a", type="directory", size=0),
            FileEntry("..", type="directory"),
        ]
    )
    backend = AgentSandboxBackend.from_existing(StubSandbox(files=files))

    response = backend.ls("/")

    assert response.error is None
    assert [entry["path"] for entry in response.entries] == ["/a", "/z.txt"]
    assert response.entries[0]["is_dir"] is True
    assert response.entries[1]["size"] == 2
    assert "2023" in response.entries[1]["modified_at"]


def test_ls_uses_shell_fallback_outside_runtime_root() -> None:
    stdout = (
        "d\t64\t1700000000\tchild-dir\x00"
        "f\t5\t1700000001\tfile.txt\x00"
        "x\t1\t1700000002\tignored\x00"
    )
    commands = StubCommands([result(stdout)])
    files = StubFiles(entries=[FileEntry("should-not-use")])
    backend = AgentSandboxBackend.from_existing(
        StubSandbox(commands=commands, files=files),
        root_dir="/tmp",
    )

    response = backend.ls("/")

    assert response.error is None
    assert files.list_calls == []
    assert [entry["path"] for entry in response.entries] == [
        "/child-dir",
        "/file.txt",
    ]
    assert response.entries[0]["is_dir"] is True
    assert response.entries[1]["size"] == 5
    assert "find -L /tmp" in commands.calls[0][0]


def test_read_window_binary_and_out_of_range() -> None:
    backend = AgentSandboxBackend.from_existing(
        StubSandbox(files=StubFiles(read=b"zero\none\ntwo"))
    )
    response = backend.read("/x.txt", offset=1, limit=1)
    assert response.file_data["content"] == "one\n"
    assert response.total_lines == 3
    assert response.start_line == 2
    assert response.end_line == 2
    assert response.next_offset == 2
    assert "exceeds file length" in backend.read("/x.txt", offset=99).error

    # DeepAgents 0.7.4 clamps degenerate model-supplied bounds rather than
    # erroring: a negative offset floors at 0, and a non-positive limit is an
    # uninspected window flagged with no_lines_requested.
    clamped = backend.read("/x.txt", offset=-1, limit=1)
    assert clamped.file_data["content"] == "zero\n"
    assert clamped.start_line == 1
    for degenerate in (0, -5):
        empty = backend.read("/x.txt", limit=degenerate)
        assert empty.no_lines_requested is True
        assert empty.file_data["content"] == ""
        assert empty.error is None

    bad = AgentSandboxBackend.from_existing(StubSandbox(files=StubFiles(read=b"\xff")))
    file_data = bad.read("/bad.bin").file_data
    assert file_data["content"] == "/w=="
    assert file_data["encoding"] == "base64"

    utf8_binary = AgentSandboxBackend.from_existing(
        StubSandbox(files=StubFiles(read=b"valid utf-8"))
    )
    assert utf8_binary.read("/image.png").file_data["encoding"] == "base64"


def test_read_preserves_line_endings_and_not_found_errors() -> None:
    backend = AgentSandboxBackend.from_existing(
        StubSandbox(files=StubFiles(read=b"zero\r\none\n"))
    )
    assert backend.read("/x.txt").file_data["content"] == "zero\r\none\n"

    missing = AgentSandboxBackend.from_existing(
        StubSandbox(files=StubFiles(read=SandboxNotFoundError("gone")))
    )
    with pytest.raises(SandboxNotFoundError):
        missing.read("/x.txt")


def test_write_overwrites_existing_and_creates_parent() -> None:
    files = StubFiles()
    files.exists_value = True
    backend = AgentSandboxBackend.from_existing(StubSandbox(files=files))
    response = backend.write("/x.txt", "data")
    assert response.error is None
    assert files.write_calls == [("x.txt", b"data")]
    assert files.exists_calls == []

    nested_files = StubFiles()
    commands = StubCommands([result(exit_code=0)])
    nested = AgentSandboxBackend.from_existing(
        StubSandbox(commands=commands, files=nested_files)
    )
    nested_response = nested.write("/dir/x.txt", "data")
    assert nested_response.error is None
    assert nested_files.write_calls == [("dir/x.txt", b"data")]
    assert "mkdir -p /workspace/dir" in commands.calls[0][0]


def test_edit_occurrence_rules() -> None:
    backend = AgentSandboxBackend.from_existing(
        StubSandbox(files=StubFiles(read=b"foo bar foo"))
    )
    assert "appears multiple times" in backend.edit("/x", "foo", "baz").error

    files = StubFiles(read=b"foo bar foo")
    backend = AgentSandboxBackend.from_existing(StubSandbox(files=files))
    response = backend.edit("/x", "foo", "baz", replace_all=True)
    assert response.error is None
    assert response.occurrences == 2
    assert files.write_calls == [("x", b"baz bar baz")]


def test_delete_files_and_directories_recursively() -> None:
    commands = StubCommands([result(), result()])
    backend = AgentSandboxBackend.from_existing(StubSandbox(commands=commands))
    backend._file_state = lambda path: "file"
    file_response = backend.delete("/x.txt")
    assert isinstance(file_response, DeleteResult)
    assert file_response.error is None
    assert "rm -rf -- /workspace/x.txt" in commands.calls[0][0]

    backend._file_state = lambda path: "dir"
    directory_response = backend.delete("/dir")
    assert directory_response.error is None
    assert "rm -rf -- /workspace/dir" in commands.calls[1][0]

    backend._file_state = lambda path: "missing"
    missing = backend.delete("/missing")
    assert missing.path is None
    assert "not found" in missing.error


def test_upload_and_download_partial_success() -> None:
    files = StubFiles(read=b"payload")
    backend = AgentSandboxBackend.from_existing(StubSandbox(files=files))
    backend._file_state = lambda path: "missing" if "ok" in path else "dir"
    backend._dir_state = lambda path: "writable"
    uploads = backend.upload_files(
        [("/ok.txt", b"ok"), ("/ok.bin", b"\xff"), ("/dir", b"bad")]
    )
    assert [item.error for item in uploads] == [None, None, "is_directory"]

    backend._file_state = lambda path: "file" if "ok" in path else "missing"
    downloads = backend.download_files(["/ok.txt", "/missing.txt"])
    assert downloads[0].content == b"payload"
    assert downloads[1].error == "file_not_found"


def test_path_virtualization_and_absolute_write_mode() -> None:
    backend = AgentSandboxBackend.from_existing(StubSandbox())
    assert backend._to_internal("/") == "/workspace"
    assert backend._to_internal("a/../b") == "/workspace/b"
    with pytest.raises(ValueError):
        backend._to_internal("../../etc/passwd")
    with pytest.raises(ValueError):
        backend._to_internal("/bad\x00path")

    sandbox = StubSandbox(commands=StubCommands([result(), result()]))
    backend = AgentSandboxBackend(
        sandbox,
        allow_absolute_paths=True,
        root_dir="/workspace",
    )
    assert backend.write("/tmp/out.txt", "hello").error is None
    assert sandbox.files.write_calls == []
    assert any("base64 -d" in call for call, _ in sandbox.commands.calls)


def test_absolute_paths_share_the_runtime_namespace() -> None:
    files = StubFiles(entries=[FileEntry("item.txt")])
    backend = AgentSandboxBackend.from_existing(
        StubSandbox(files=files),
        root_dir="/app/workspace",
        runtime_root="/app",
        allow_absolute_paths=True,
    )

    assert backend._to_internal("/") == "/app/workspace"
    assert backend._to_internal("/workspace") == "/app/workspace"
    assert backend._to_internal("/uploads/file.txt") == "/app/uploads/file.txt"
    assert backend._to_internal("/app/.agents/state.json") == "/app/.agents/state.json"
    assert backend._to_public("/app/.agents/state.json") == "/app/.agents/state.json"

    response = backend.ls("/workspace")

    assert response.error is None
    assert files.list_calls == ["workspace"]
    assert response.entries[0]["path"] == "/workspace/item.txt"


def test_grep_glob_and_malformed_output() -> None:
    grep_stdout = "/workspace/a b.py\x001:hit\nmalformed\n/workspace/no:bad\n"
    backend = AgentSandboxBackend.from_existing(
        StubSandbox(commands=StubCommands([result(grep_stdout)]))
    )
    matches = backend.grep("hit", path="/space dir").matches
    assert matches == [{"path": "/a b.py", "line": 1, "text": "hit"}]

    find_stdout = (
        "f\t1\t1700000000\t/workspace/main.py\x00"
        "f\t1\t1700000000\t/workspace/pkg/main.py\x00"
    )
    backend = AgentSandboxBackend.from_existing(
        StubSandbox(commands=StubCommands([result(find_stdout)]))
    )
    assert [entry["path"] for entry in backend.glob("**/main.py").matches] == [
        "/main.py",
        "/pkg/main.py",
    ]
    assert compile_glob("a/**/b")("ab") is False
    assert compile_glob("{main,test}.py")("pkg/main.py")
    assert compile_grep_include_glob("src/**/*.py")("src/pkg/main.py")
    assert not compile_grep_include_glob("src/**/*.py")("other/main.py")
    assert "traversal" in backend.glob("../*.py").error


def test_grep_applies_include_glob_and_total_max_count() -> None:
    stdout = (
        "/workspace/src/a.py\x001:first\n"
        "/workspace/src/b.txt\x002:ignored\n"
        "/workspace/src/c.py\x003:second\n"
    )
    backend = AgentSandboxBackend.from_existing(
        StubSandbox(commands=StubCommands([result(stdout), result(stdout)]))
    )

    filtered = backend.grep("hit", glob="src/{a,c}.py", max_count=1)
    assert filtered.matches == [{"path": "/src/a.py", "line": 1, "text": "first"}]
    assert filtered.truncated is True

    complete = backend.grep("hit", glob="src/a.py", max_count=1)
    assert complete.matches == [{"path": "/src/a.py", "line": 1, "text": "first"}]
    assert complete.truncated is False


def test_ephemeral_lifecycle_creates_and_deletes_its_own_claim() -> None:
    client = StubClient()
    backend = AgentSandboxBackend.from_warm_pool(
        client,
        "python",
        namespace="ns",
        shutdown_after_seconds=30,
    )
    with backend:
        assert backend.id == "ns/claim-1"
    assert client.deleted == [("claim-1", "ns")]

    with pytest.raises(RuntimeError, match="not initialized"):
        AgentSandboxBackend.from_warm_pool(StubClient(), "python").execute("pwd")


def test_ephemeral_backend_never_discovers_claims_by_label() -> None:
    """Identity-based Claim discovery belongs to the provider, not here."""
    client = StubClient(claims=["claim-old"])
    backend = AgentSandboxBackend.from_warm_pool(client, "python", namespace="ns")
    with backend:
        # A pre-existing Claim is ignored: the ephemeral path always creates
        # its own rather than adopting one it found by listing.
        assert backend.id == "ns/claim-1"
    assert client.listed == []
    assert client.deleted == [("claim-1", "ns")]

    assert not hasattr(AgentSandboxBackend, "_try_reattach")
    assert not hasattr(AgentSandboxBackend, "SESSION_LABEL_KEY")
    assert not hasattr(AgentSandboxBackend, "delete_all")


def test_drain_rejects_operations_and_cleanup_errors_surface() -> None:
    backend = AgentSandboxBackend.from_existing(StubSandbox())
    backend._draining = True
    with pytest.raises(RuntimeError, match="shutting down"):
        backend.execute("pwd")

    class FailingDeleteClient(StubClient):
        def delete_sandbox(self, claim_name: str, namespace: str = "default") -> None:
            raise RuntimeError("delete failed")

    managed = AgentSandboxBackend.from_warm_pool(FailingDeleteClient(), "python")
    managed.__enter__()
    with pytest.raises(RuntimeError, match="delete failed"):
        managed.__exit__(None, None, None)


def test_compat_create_uses_v1beta1_sdk_contract() -> None:
    class Client:
        def create_sandbox(self, **kwargs: Any) -> str:
            self.kwargs = kwargs
            return "ok"

    client = Client()
    assert (
        _compat.create_sandbox(
            client,
            warm_pool="pool",
            namespace="ns",
            sandbox_ready_timeout=1,
            labels={"a": "b"},
            shutdown_after_seconds=30,
        )
        == "ok"
    )
    assert client.kwargs == {
        "warmpool": "pool",
        "namespace": "ns",
        "sandbox_ready_timeout": 1,
        "labels": {"a": "b"},
        "shutdown_after_seconds": 30,
    }


@pytest.mark.parametrize(
    "removed",
    [
        "list_sandbox_claims",
        "get_sandbox",
        "validate_label_value",
        "SESSION_LABEL_KEY",
        "_claim_names",
    ],
)
def test_compat_no_longer_wraps_claim_discovery(removed: str) -> None:
    """Claim listing and adoption moved to the provider and must not return."""
    assert not hasattr(_compat, removed)


def test_compat_only_calls_official_sdk_methods() -> None:
    """Every method _compat forwards to must exist on the official client."""
    from k8s_agent_sandbox.sandbox_client import SandboxClient

    for method in ("create_sandbox", "delete_sandbox"):
        assert hasattr(SandboxClient, method)


def test_from_template_is_deprecated_alias() -> None:
    with pytest.warns(DeprecationWarning, match="from_warm_pool"):
        backend = AgentSandboxBackend.from_template(StubClient(), "python")
    assert isinstance(backend, AgentSandboxBackend)


def test_policy_wrapper_blocks_and_audits() -> None:
    backend = AgentSandboxBackend.from_existing(
        StubSandbox(commands=StubCommands([result("ok")]))
    )
    audit: list[tuple[str, str, dict[str, Any]]] = []
    wrapped = SandboxPolicyWrapper(
        backend,
        deny_prefixes=["/etc"],
        deny_commands=["rm -rf"],
        audit_log=lambda op, target, meta: audit.append((op, target, meta)),
    )
    assert wrapped.write("/etc/passwd", "x").error.startswith("Policy denied")
    assert wrapped.execute("rm -rf /").exit_code == 1
    assert wrapped.execute("echo ok").output == "ok"
    assert audit[0][0] == "execute"

    wrapped = SandboxPolicyWrapper(
        backend,
        audit_log=lambda op, target, meta: (_ for _ in ()).throw(RuntimeError("down")),
        strict_audit=True,
    )
    assert "Audit log unavailable" in wrapped.execute("echo ok").output


def test_policy_read_operations_pass_through() -> None:
    backend = AgentSandboxBackend.from_existing(
        StubSandbox(
            commands=StubCommands([result("/workspace/a.py:1:hit\n")]),
            files=StubFiles(read=b"hit"),
        )
    )
    wrapped = SandboxPolicyWrapper(
        backend,
        deny_prefixes=["/"],
        deny_commands=["grep"],
    )
    assert wrapped.read("/a.py").file_data["content"] == "hit"
    assert wrapped.grep("hit").matches[0]["text"] == "hit"


def test_policy_blocks_recursive_delete_over_denied_descendant() -> None:
    commands = StubCommands([result()])
    backend = AgentSandboxBackend.from_existing(StubSandbox(commands=commands))
    wrapped = SandboxPolicyWrapper(backend, deny_prefixes=["/protected/data"])

    response = wrapped.delete("/")

    assert FilesystemMiddleware(backend=wrapped)
    assert isinstance(response, DeleteResult)
    assert response.error.startswith("Policy denied")
    assert commands.calls == []


@pytest.mark.asyncio
async def test_policy_delegates_file_operations_and_context() -> None:
    backend = MagicMock(spec=AgentSandboxBackend)
    backend.id = "ns/claim"
    backend.ls.return_value = SimpleNamespace(entries=[], error=None)
    backend.glob.return_value = SimpleNamespace(matches=[], error=None)
    backend.download_files.return_value = []
    backend.edit.return_value = SimpleNamespace(
        path="/file.txt", error=None, occurrences=1
    )
    backend.delete.return_value = DeleteResult(path="/file.txt")
    backend.upload_files.return_value = [SimpleNamespace(path="/file.txt", error=None)]
    wrapped = SandboxPolicyWrapper(backend)

    with wrapped as entered:
        assert entered is wrapped
    async with wrapped as entered:
        assert entered is wrapped
    assert wrapped.ls("/").error is None
    assert wrapped.glob("*.py").error is None
    assert wrapped.download_files(iter(["/file.txt"])) == []
    assert wrapped.edit("/file.txt", "old", "new").error is None
    assert wrapped.delete("/file.txt").path == "/file.txt"
    assert wrapped.upload_files({"/file.txt": b"data"})[0].error is None
    assert wrapped.id == "ns/claim"
    assert backend.__enter__.call_count == 2
    assert backend.__exit__.call_count == 2


def test_policy_denies_edit_and_partial_uploads() -> None:
    backend = MagicMock(spec=AgentSandboxBackend)
    backend.upload_files.return_value = [
        SimpleNamespace(path="/allowed.txt", error=None)
    ]
    wrapped = SandboxPolicyWrapper(backend, deny_prefixes=["/blocked"])

    edit = wrapped.edit("/blocked/file.txt", "old", "new")
    uploads = wrapped.upload_files(
        [
            ("/blocked/file.txt", b"blocked"),
            ("/allowed.txt", b"allowed"),
        ]
    )

    assert edit.error.startswith("Policy denied")
    assert uploads[0].error == "policy_denied"
    assert uploads[1].error is None
    backend.upload_files.assert_called_once_with([("/allowed.txt", b"allowed")])


def test_policy_strict_audit_denies_all_mutations() -> None:
    backend = MagicMock(spec=AgentSandboxBackend)

    def fail_audit(operation: str, target: str, metadata: dict[str, Any]) -> None:
        raise RuntimeError("audit unavailable")

    wrapped = SandboxPolicyWrapper(
        backend,
        audit_log=fail_audit,
        strict_audit=True,
    )

    assert "Audit log unavailable" in wrapped.write("/file.txt", "data").error
    assert "Audit log unavailable" in wrapped.edit("/file.txt", "old", "new").error
    assert "Audit log unavailable" in wrapped.delete("/file.txt").error
    assert (
        "Audit log unavailable"
        in wrapped.upload_files([("/file.txt", b"data")])[0].error
    )
    backend.write.assert_not_called()
    backend.edit.assert_not_called()
    backend.delete.assert_not_called()
    backend.upload_files.assert_not_called()


def test_backend_helper_returns_deepagents_07_instance() -> None:
    client = StubClient()
    backend = create_sandbox_backend(
        "python",
        namespace="ns",
        client=client,
    )

    assert isinstance(backend, AgentSandboxBackend)
    assert FilesystemMiddleware(backend=backend)
    assert client.created
    with backend as entered:
        assert entered is backend
        assert len(client.created) == 1
    assert client.deleted == [("claim-1", "ns")]


def test_backend_helper_finalizer_is_idempotent() -> None:
    client = StubClient()
    backend = create_sandbox_backend("python", namespace="ns", client=client)

    backend._finalizer()
    backend._finalizer()
    assert client.deleted == [("claim-1", "ns")]


def test_factory_finalizer_is_detached_after_explicit_exit() -> None:
    client = StubClient()
    with pytest.warns(DeprecationWarning, match="concrete backend"):
        backend = create_sandbox_backend_factory(
            "python", namespace="ns", client=client
        )
    assert FilesystemMiddleware(backend=backend)
    with pytest.warns(DeprecationWarning, match="Calling the backend"):
        assert backend(SimpleNamespace()) is backend
    finalizer = backend._finalizer

    backend.__exit__(None, None, None)
    assert finalizer is not None
    assert not finalizer.alive
    finalizer()
    assert client.deleted == [("claim-1", "ns")]


def test_factory_creates_its_own_ephemeral_claim() -> None:
    """The deprecated factory path no longer adopts a pre-existing Claim."""
    client = StubClient(claims=["claim-old"])
    with pytest.warns(DeprecationWarning, match="concrete backend"):
        backend = create_sandbox_backend_factory(
            "python",
            namespace="ns",
            client=client,
        )

    assert isinstance(backend, AgentSandboxBackend)
    assert backend.id == "ns/claim-1"
    assert len(client.created) == 1
    assert client.listed == []
    backend.__exit__(None, None, None)
    assert client.deleted == [("claim-1", "ns")]
