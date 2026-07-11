from __future__ import annotations

import builtins
import shlex
from types import SimpleNamespace
from typing import Any

import pytest

from langchain_google_agent_sandbox import (
    AgentSandboxBackend,
    SandboxPolicyWrapper,
    _compat,
    create_sandbox_backend_factory,
)
from langchain_google_agent_sandbox._paths import compile_glob


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
    assert backend.read("/x.txt", offset=1, limit=1).file_data["content"] == "one"
    assert "exceeds file length" in backend.read("/x.txt", offset=99).error

    bad = AgentSandboxBackend.from_existing(StubSandbox(files=StubFiles(read=b"\xff")))
    file_data = bad.read("/bad.bin").file_data
    assert file_data["content"] == "/w=="
    assert file_data["encoding"] == "base64"


def test_write_refuses_existing_and_creates_parent() -> None:
    files = StubFiles()
    files.exists_value = True
    backend = AgentSandboxBackend.from_existing(StubSandbox(files=files))
    assert "already exists" in backend.write("/x.txt", "data").error

    files = StubFiles()
    commands = StubCommands([result(exit_code=0)])
    backend = AgentSandboxBackend.from_existing(
        StubSandbox(commands=commands, files=files)
    )
    response = backend.write("/dir/x.txt", "data")
    assert response.error is None
    assert files.write_calls == [("dir/x.txt", b"data")]
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


def test_delete_file_and_refuse_directory() -> None:
    commands = StubCommands([result()])
    backend = AgentSandboxBackend.from_existing(StubSandbox(commands=commands))
    backend._file_state = lambda path: "file"
    assert backend.delete("/x.txt").error is None
    assert "rm -f -- /workspace/x.txt" in commands.calls[0][0]

    backend._file_state = lambda path: "dir"
    assert "directory" in backend.delete("/dir").error


def test_upload_and_download_partial_success(monkeypatch: pytest.MonkeyPatch) -> None:
    files = StubFiles(read=b"payload")
    backend = AgentSandboxBackend.from_existing(StubSandbox(files=files))
    backend._file_state = lambda path: "missing" if "ok" in path else "dir"
    backend._dir_state = lambda path: "writable"
    files_update: dict[str, Any] = {}
    monkeypatch.setattr(_compat, "send_files_update", files_update.update)

    uploads = backend.upload_files(
        [("/ok.txt", b"ok"), ("/ok.bin", b"\xff"), ("/dir", b"bad")]
    )
    assert [item.error for item in uploads] == [None, None, "is_directory"]
    assert files_update["/ok.txt"]["encoding"] == "utf-8"
    assert files_update["/ok.bin"]["encoding"] == "base64"

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

    sandbox = StubSandbox(commands=StubCommands([result(exit_code=1), result()]))
    backend = AgentSandboxBackend(
        sandbox,
        allow_absolute_paths=True,
        root_dir="/workspace",
    )
    assert backend.write("/tmp/out.txt", "hello").error is None
    assert sandbox.files.write_calls == []
    assert any("base64 -d" in call for call, _ in sandbox.commands.calls)


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
    assert backend.glob("file[z-a].py").error.startswith("invalid glob pattern")
    assert compile_glob("a/**/b")("ab") is False


def test_lifecycle_create_delete_reattach_and_refusals() -> None:
    client = StubClient()
    backend = AgentSandboxBackend.from_warm_pool(
        client,
        "python",
        namespace="ns",
        session_id="thread-1",
        shutdown_after_seconds=30,
    )
    with backend:
        assert (
            client.created[0]["labels"][AgentSandboxBackend.SESSION_LABEL_KEY]
            == "thread-1"
        )
        assert backend.id == "ns/claim-1"
    assert client.deleted == [("claim-1", "ns")]

    client = StubClient(claims=["claim-old"])
    with AgentSandboxBackend.from_warm_pool(
        client, "python", namespace="ns", session_id="s1"
    ) as value:
        assert value.id == "ns/claim-old"
    assert client.deleted == []

    with pytest.raises(RuntimeError, match="Refusing to reattach"):
        AgentSandboxBackend.from_warm_pool(
            StubClient(claims=["a", "b"]), "python", session_id="s1"
        ).__enter__()

    with pytest.raises(RuntimeError, match="not initialized"):
        AgentSandboxBackend.from_warm_pool(StubClient(), "python").execute("pwd")


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


def test_compat_requires_public_sdk_list_method() -> None:
    with pytest.raises(AttributeError):
        _compat.list_sandbox_claims(
            SimpleNamespace(), namespace="ns", label_selector="x=y"
        )


def test_langgraph_file_update_noops_without_graph_context() -> None:
    _compat.send_files_update({"x.txt": {"content": "x", "encoding": "utf-8"}})


def test_langgraph_file_update_noops_when_langgraph_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.startswith("langgraph"):
            msg = "missing langgraph"
            raise ModuleNotFoundError(msg)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _compat.send_files_update({"x.txt": {"content": "x", "encoding": "utf-8"}})


def test_template_validation_when_metadata_is_available() -> None:
    sandbox = SimpleNamespace()

    class Client:
        def get_sandbox_claim_warmpool_name(
            self, claim_name: str, namespace: str
        ) -> str:
            return "other"

        def get_sandbox(self, claim_name: str, namespace: str = "default") -> Any:
            return sandbox

    with pytest.raises(ValueError, match="does not match"):
        _compat.get_sandbox(
            Client(), claim_name="claim", namespace="ns", warm_pool="wanted"
        )


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


def test_factory_enters_backend_and_finalizer_is_idempotent() -> None:
    client = StubClient()
    factory = create_sandbox_backend_factory(
        "python",
        namespace="ns",
        client=client,
        session_id="thread-2",
    )
    backend = factory(SimpleNamespace())

    assert isinstance(backend, AgentSandboxBackend)
    assert client.created
    backend._finalizer()
    backend._finalizer()
    assert client.deleted == [("claim-1", "ns")]


def test_factory_preserves_reattached_backend() -> None:
    client = StubClient(claims=["claim-old"])
    factory = create_sandbox_backend_factory(
        "python",
        namespace="ns",
        client=client,
        session_id="thread-2",
    )

    backend = factory(SimpleNamespace())

    assert isinstance(backend, AgentSandboxBackend)
    assert backend.id == "ns/claim-old"
    assert client.created == []
    assert not hasattr(backend, "_finalizer")
    backend.__exit__(None, None, None)
    assert client.deleted == []
