"""Tests for the provider-backed session backend."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from k8s_agent_sandbox.exceptions import (
    SandboxClaimFailedError,
    SandboxNotFoundError,
)
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from langchain_google_agent_sandbox import (
    AgentSandboxBackend,
    ProviderSessionAgentSandboxBackend,
    SandboxPolicyWrapper,
    SessionAgentSandboxBackend,
    default_session_resolver,
)
from langchain_google_agent_sandbox.provider import SandboxLease
from langchain_google_agent_sandbox.session_backend import _SessionEntry
from tests.unit_tests.test_backend import StubSandbox, result


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ToolCallingFakeModel:
        del tools, kwargs
        return self


def config_for(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


class RecordingProvider:
    """In-memory SandboxSessionProvider standing in for a product control plane.

    It keeps one sandbox per thread id so a re-acquire after eviction returns
    the same durable workspace, exactly as a real control plane would.
    """

    def __init__(self, *, acquire_error: Exception | None = None) -> None:
        self.acquire_calls: list[str] = []
        self.touched: list[str] = []
        self.closed_local: list[str] = []
        self.acquire_error = acquire_error
        self.sandboxes: dict[str, StubSandbox] = {}
        self._lock = threading.Lock()

    def acquire(self, config: Mapping[str, Any]) -> SandboxLease:
        thread_id = default_session_resolver(config)
        if self.acquire_error is not None:
            raise self.acquire_error
        with self._lock:
            self.acquire_calls.append(thread_id)
            sandbox = self.sandboxes.get(thread_id)
            if sandbox is None or sandbox.closed:
                sandbox = StubSandbox(claim_name=f"claim-{thread_id}")
                self.sandboxes[thread_id] = sandbox
        return SandboxLease(
            key=f"lease-{thread_id}",
            sandbox=sandbox,
            claim_name=f"claim-{thread_id}",
            namespace="tenant",
        )

    def touch(self, lease: SandboxLease) -> None:
        self.touched.append(lease.key)

    def close_local(self, lease: SandboxLease) -> None:
        self.closed_local.append(lease.key)


def make_backend(
    provider: Any = None, **kwargs: Any
) -> ProviderSessionAgentSandboxBackend:
    return ProviderSessionAgentSandboxBackend(provider or RecordingProvider(), **kwargs)


def pinned(backend: ProviderSessionAgentSandboxBackend, thread_id: str) -> None:
    """Pin the backend's active config, standing in for a LangGraph run."""
    backend._active_config = staticmethod(lambda: config_for(thread_id))  # type: ignore[assignment]


# --- acquisition and caching -------------------------------------------------


def test_provider_is_called_once_per_session_key() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider)
    for _ in range(4):
        backend._entry(config_for("t1"))
    assert provider.acquire_calls == ["t1"]


def test_distinct_sessions_never_share_a_lease() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider)
    first = backend.get_session_lease(config_for("t1"))
    second = backend.get_session_lease(config_for("t2"))
    assert first.key != second.key
    assert first.sandbox is not second.sandbox
    assert sorted(provider.acquire_calls) == ["t1", "t2"]


def test_provider_reusing_one_lease_for_two_sessions_is_refused() -> None:
    """A shared sandbox would cross-contaminate two agents' filesystems."""

    class CollidingProvider(RecordingProvider):
        def acquire(self, config: Mapping[str, Any]) -> SandboxLease:
            super().acquire(config)
            return SandboxLease(key="same-lease", sandbox=StubSandbox())

    backend = make_backend(CollidingProvider())
    backend._entry(config_for("t1"))
    with pytest.raises(RuntimeError, match="already held by session"):
        backend._entry(config_for("t2"))


def test_one_acquisition_per_key_under_concurrency() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider)
    with ThreadPoolExecutor(max_workers=8) as executor:
        leases = list(
            executor.map(
                lambda _: backend.get_session_lease(config_for("shared")),
                range(16),
            )
        )
    assert {lease.key for lease in leases} == {"lease-shared"}
    assert provider.acquire_calls == ["shared"]


def test_concurrent_distinct_sessions_each_acquire_once() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider, max_cached_sessions=64)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda index: backend.get_session_lease(config_for(f"s{index}")),
                range(16),
            )
        )
    assert sorted(provider.acquire_calls) == sorted(f"s{i}" for i in range(16))


def test_acquire_rejects_non_lease_return() -> None:
    class BadProvider(RecordingProvider):
        def acquire(self, config: Mapping[str, Any]) -> Any:
            return "not-a-lease"

    backend = make_backend(BadProvider())
    with pytest.raises(TypeError, match="must return a SandboxLease"):
        backend._entry(config_for("t1"))


def test_provider_errors_propagate_unchanged() -> None:
    class DeniedError(RuntimeError):
        pass

    denial = DeniedError("capacity full")
    backend = make_backend(RecordingProvider(acquire_error=denial))
    with pytest.raises(DeniedError, match="capacity full") as info:
        backend._entry(config_for("t1"))
    assert info.value is denial


def test_provider_receives_config_not_model_content() -> None:
    """Callbacks must never be handed raw messages or file contents."""
    seen: list[Mapping[str, Any]] = []

    class Inspecting(RecordingProvider):
        def acquire(self, config: Mapping[str, Any]) -> SandboxLease:
            seen.append(config)
            return super().acquire(config)

    backend = make_backend(Inspecting())
    backend._entry(config_for("t1"))
    assert seen == [config_for("t1")]
    assert set(seen[0]) == {"configurable"}


# --- touch throttling --------------------------------------------------------


def test_touch_is_throttled_to_the_configured_interval() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider, touch_interval_seconds=3600)
    for _ in range(5):
        backend._entry(config_for("t1"))
    # Acquisition seeds the interval, so nothing fires inside it.
    assert provider.touched == []


def test_touch_fires_once_the_interval_elapses() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider, touch_interval_seconds=0)
    for _ in range(3):
        backend._entry(config_for("t1"))
    # A zero interval reports every cached access; the acquire itself does not.
    assert provider.touched == ["lease-t1", "lease-t1"]


def test_touch_is_not_called_once_per_file_operation() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider, touch_interval_seconds=3600)
    pinned(backend, "busy")
    for _ in range(10):
        backend.execute("pwd")
    assert provider.touched == []


def test_touch_failure_does_not_break_the_operation() -> None:
    class FlakyTouch(RecordingProvider):
        def touch(self, lease: SandboxLease) -> None:
            raise RuntimeError("control plane down")

    backend = make_backend(FlakyTouch(), touch_interval_seconds=0)
    backend._entry(config_for("t1"))
    assert backend._entry(config_for("t1")) is not None


# --- eviction ----------------------------------------------------------------


def test_ttl_eviction_closes_only_local_resources() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider, local_cache_ttl_seconds=1)
    entry = backend._entry(config_for("old"))
    sandbox = entry.lease.sandbox

    entry.last_accessed_at = time.monotonic() - 10
    backend._entry(config_for("other"))

    assert provider.closed_local == ["lease-old"]
    assert "old" not in backend._entries
    assert sandbox.closed is True
    # The provider was never asked to delete anything.
    assert not hasattr(provider, "deleted")


def test_max_cache_eviction_is_deterministic_oldest_first() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider, max_cached_sessions=2)
    first = backend._entry(config_for("a"))
    second = backend._entry(config_for("b"))
    now = time.monotonic()
    first.last_accessed_at = now - 100
    second.last_accessed_at = now - 1

    backend._entry(config_for("c"))

    assert "a" not in backend._entries
    assert "b" in backend._entries
    assert provider.closed_local == ["lease-a"]


def test_evicted_session_reacquires_the_same_durable_workspace() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider, max_cached_sessions=1)
    backend._entry(config_for("keep"))
    backend._entry(config_for("evicts-keep"))
    assert "keep" not in backend._entries

    lease = backend.get_session_lease(config_for("keep"))
    assert lease.claim_name == "claim-keep"
    assert provider.acquire_calls.count("keep") == 2


def test_close_session_releases_locally_without_deleting() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider)
    backend._entry(config_for("t1"))
    backend.close_session(config_for("t1"))
    assert provider.closed_local == ["lease-t1"]
    assert backend._entries == {}


def test_close_releases_every_session_and_deletes_nothing() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider)
    backend._entry(config_for("t1"))
    backend._entry(config_for("t2"))
    backend.close()
    assert sorted(provider.closed_local) == ["lease-t1", "lease-t2"]
    assert backend._entries == {}
    with pytest.raises(RuntimeError, match="is closed"):
        backend._entry(config_for("t1"))


def test_close_local_failure_still_clears_the_cache() -> None:
    class FailingClose(RecordingProvider):
        def close_local(self, lease: SandboxLease) -> None:
            raise RuntimeError("close failed")

    backend = make_backend(FailingClose())
    backend._entry(config_for("t1"))
    backend.close()
    assert backend._entries == {}


# --- staleness and retry -----------------------------------------------------


def test_missing_sandbox_triggers_exactly_one_reacquire() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider)
    pinned(backend, "t1")
    entry = backend._entry(config_for("t1"))

    calls: list[str] = []

    def fail_once(command: str, **kwargs: Any) -> Any:
        calls.append(command)
        if len(calls) == 1:
            raise SandboxNotFoundError("gone")
        return result("ok")

    entry.lease.sandbox.commands.run = fail_once  # type: ignore[method-assign]

    response = backend.execute("echo hi")

    assert response.exit_code == 0
    assert provider.acquire_calls == ["t1", "t1"]


def test_second_missing_sandbox_propagates_without_retry_loop() -> None:
    def always_missing(command: str, **kwargs: Any) -> Any:
        raise SandboxNotFoundError("gone")

    class AlwaysMissingProvider(RecordingProvider):
        def acquire(self, config: Mapping[str, Any]) -> SandboxLease:
            lease = super().acquire(config)
            lease.sandbox.commands.run = always_missing  # type: ignore[method-assign]
            return lease

    provider = AlwaysMissingProvider()
    backend = make_backend(provider)
    pinned(backend, "t1")

    with pytest.raises(SandboxNotFoundError):
        backend.execute("echo hi")
    # One acquire, one re-acquire, then the failure surfaces. No loop.
    assert provider.acquire_calls == ["t1", "t1"]


def test_terminal_claim_failure_is_not_retried() -> None:
    """SandboxClaimFailedError is terminal and must not drive a re-acquire.

    It reports a Ready=False reason the claim controller will not retry
    (InvalidMetadata, VolumeClaimTemplatesError, ClaimExpired), so re-acquiring
    would be pointless churn. This guards the narrow ``except
    SandboxNotFoundError`` in ``_invoke``: the two are siblings under
    SandboxError, and a broader catch would swallow this one.

    Asserted against the wrapped backend because the concrete backend
    deliberately converts non-SandboxNotFoundError command failures into error
    results rather than raising.
    """
    provider = RecordingProvider()
    backend = make_backend(provider)
    pinned(backend, "t1")
    entry = backend._entry(config_for("t1"))

    def terminal(*args: Any, **kwargs: Any) -> Any:
        raise SandboxClaimFailedError("claim will never become ready")

    entry.backend.execute = terminal  # type: ignore[method-assign]

    with pytest.raises(SandboxClaimFailedError):
        backend.execute("echo hi")
    # No second acquisition: the error was never treated as staleness.
    assert provider.acquire_calls == ["t1"]


def test_terminal_and_stale_errors_are_distinct_siblings() -> None:
    """The retry rule depends on these not being in an inheritance relation."""
    assert not issubclass(SandboxClaimFailedError, SandboxNotFoundError)
    assert not issubclass(SandboxNotFoundError, SandboxClaimFailedError)


def test_provider_terminal_error_during_acquire_propagates() -> None:
    backend = make_backend(
        RecordingProvider(acquire_error=SandboxClaimFailedError("terminal"))
    )
    with pytest.raises(SandboxClaimFailedError, match="terminal"):
        backend._entry(config_for("t1"))


# --- lifecycle boundary ------------------------------------------------------


@pytest.mark.parametrize(
    "removed",
    [
        "delete_session",
        "get_session_endpoint",
        "get_session_claim_name",
        "get_session_sandbox",
        "opaque_session_id",
        "_claim_name",
        "_legacy_claim",
        "_renew",
        "SESSION_LABEL_KEY",
    ],
)
def test_backend_exposes_no_claim_lifecycle_surface(removed: str) -> None:
    assert not hasattr(ProviderSessionAgentSandboxBackend, removed)


def test_lease_key_is_passed_through_not_derived() -> None:
    """The adapter must not hash or transform product identity."""
    backend = make_backend()
    pinned(backend, "t1")
    assert backend.get_session_lease(config_for("t1")).key == "lease-t1"


def test_id_is_stable_and_never_provisions_a_sandbox() -> None:
    """`id` names the backend instance, so reading it must not call acquire."""
    provider = RecordingProvider()
    backend = make_backend(provider)
    pinned(backend, "t1")

    first = backend.id
    assert backend.id == first  # stable across reads
    assert provider.acquire_calls == []  # and free of side effects

    backend._entry(config_for("t1"))
    # Acquiring a session must not change the instance identifier.
    assert backend.id == first
    assert make_backend(RecordingProvider()).id != first


def test_concurrent_sessions_cannot_share_one_lease() -> None:
    """Regression: the ownership check must be atomic with publication.

    Two session keys hold different per-session locks. A check performed
    outside the cache lock passed for both threads before either published,
    leaving two agents on one sandbox filesystem.
    """
    barrier = threading.Barrier(2)

    class CollidingProvider(RecordingProvider):
        def acquire(self, config: Mapping[str, Any]) -> SandboxLease:
            super().acquire(config)
            return SandboxLease(
                key="SHARED", sandbox=self.sandboxes.setdefault("shared", StubSandbox())
            )

    provider = CollidingProvider()
    backend = make_backend(provider)

    original_wrap = backend._wrap

    def wrap_in_the_race_window(sandbox: Any) -> Any:
        # Park both threads after the ownership check, before publication.
        barrier.wait(timeout=5)
        return original_wrap(sandbox)

    backend._wrap = wrap_in_the_race_window  # type: ignore[method-assign]

    accepted: list[str] = []
    refused: list[str] = []

    def worker(thread_id: str) -> None:
        try:
            backend._entry(config_for(thread_id))
            accepted.append(thread_id)
        except RuntimeError as error:
            refused.append(str(error))

    threads = [threading.Thread(target=worker, args=(t,)) for t in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(accepted) == 1, "exactly one session may hold the lease"
    assert len(refused) == 1
    assert "already held by session" in refused[0]

    sandboxes = {id(entry.lease.sandbox) for entry in backend._entries.values()}
    assert len(backend._entries) == len(sandboxes) == 1


def test_losing_the_lease_race_releases_the_opened_connection() -> None:
    """A refused acquisition must not leak the connector it already opened."""
    provider = RecordingProvider()
    backend = make_backend(provider)
    backend._entry(config_for("first"))

    # Force a second session onto the same lease key.
    def collide(config: Mapping[str, Any]) -> SandboxLease:
        return SandboxLease(key="lease-first", sandbox=StubSandbox())

    provider.acquire = collide  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="already held by session"):
        backend._entry(config_for("second"))

    # close_local ran for the rejected lease, so nothing was left open.
    assert provider.closed_local == ["lease-first"]
    assert "second" not in backend._entries


def test_backend_is_not_kept_alive_by_its_shutdown_hook() -> None:
    """Regression: atexit.register(self.close) made every backend immortal.

    In a process creating one backend per tenant, that leaked the backend and
    every sandbox connection it held for the life of the process.
    """
    refs = [weakref.ref(make_backend(RecordingProvider())) for _ in range(25)]
    gc.collect()
    assert [ref() for ref in refs].count(None) == 25


def test_requires_a_provider() -> None:
    with pytest.raises(ValueError, match="provider is required"):
        ProviderSessionAgentSandboxBackend(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"local_cache_ttl_seconds": 0}, "local_cache_ttl_seconds"),
        ({"max_cached_sessions": 0}, "max_cached_sessions"),
        ({"touch_interval_seconds": -1}, "touch_interval_seconds"),
    ],
)
def test_rejects_invalid_cache_configuration(
    kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        make_backend(**kwargs)


# --- deprecated alias --------------------------------------------------------


def test_deprecated_alias_warns_but_works() -> None:
    with pytest.warns(DeprecationWarning, match="ProviderSessionAgentSandboxBackend"):
        backend = SessionAgentSandboxBackend(RecordingProvider())
    assert backend.get_session_lease(config_for("t1")).key == "lease-t1"


@pytest.mark.parametrize(
    "removed",
    [
        "client",
        "warm_pool",
        "session_secret",
        "namespace",
        "before_session_acquire",
        "after_session_acquire",
        "on_session_accessed",
        "on_session_created",
        "idle_ttl_seconds",
        "legacy_label_fallback",
    ],
)
def test_deprecated_alias_rejects_removed_lifecycle_arguments(removed: str) -> None:
    """Removed arguments must fail loudly, not vanish into **kwargs."""
    with pytest.raises(TypeError, match=removed):
        SessionAgentSandboxBackend(RecordingProvider(), **{removed: "x"})


def test_default_session_resolver_requires_thread_id() -> None:
    assert default_session_resolver(config_for("t1")) == "t1"
    with pytest.raises(RuntimeError, match="thread_id is required"):
        default_session_resolver({"configurable": {}})
    with pytest.raises(RuntimeError, match="must be a mapping"):
        default_session_resolver({"configurable": "nope"})


def test_custom_session_resolver_scopes_the_cache_key() -> None:
    provider = RecordingProvider()
    backend = make_backend(
        provider,
        session_resolver=lambda config: config["configurable"]["thread_id"].upper(),
    )
    backend._entry(config_for("t1"))
    assert "T1" in backend._entries


# --- DeepAgents integration --------------------------------------------------


def test_policy_wrapper_accepts_session_backend() -> None:
    backend = make_backend()
    pinned(backend, "policy")
    wrapped = SandboxPolicyWrapper(backend, deny_prefixes=["/etc"])

    assert wrapped.write("/etc/passwd", "blocked").error.startswith("Policy denied")
    assert wrapped.write("/allowed.txt", "ok").error is None


def test_direct_composite_and_middleware_integration() -> None:
    backend = make_backend()
    assert isinstance(backend, SandboxBackendProtocol)
    composite = CompositeBackend(default=backend, routes={})
    assert composite.default is backend
    assert FilesystemMiddleware(backend=backend)
    assert SkillsMiddleware(backend=backend, sources=[])


@pytest.mark.asyncio
async def test_sync_and_async_deepagents_delegation() -> None:
    backend = make_backend()
    pinned(backend, "delegation")
    entry = backend._entry(config_for("delegation"))

    concrete = MagicMock(spec=AgentSandboxBackend)
    concrete.execute.return_value = result("ok")
    concrete.ls.return_value = SimpleNamespace(entries=[], error=None)
    concrete.read.return_value = SimpleNamespace(file_data=None, error=None)
    concrete.write.return_value = SimpleNamespace(path="/x", error=None)
    concrete.edit.return_value = SimpleNamespace(path="/x", error=None, occurrences=1)
    concrete.delete.return_value = SimpleNamespace(path="/x", error=None)
    concrete.grep.return_value = SimpleNamespace(matches=[], error=None)
    concrete.glob.return_value = SimpleNamespace(matches=[], error=None)
    concrete.upload_files.return_value = []
    concrete.download_files.return_value = []
    backend._entries["delegation"] = _SessionEntry(
        backend=concrete,
        lease=entry.lease,
        last_accessed_at=entry.last_accessed_at,
        last_touched_at=entry.last_touched_at,
    )

    backend.execute("pwd")
    backend.ls("/")
    backend.read("/x")
    backend.write("/x", "x")
    backend.edit("/x", "x", "y")
    backend.delete("/x")
    backend.grep("x", max_count=7)
    backend.glob("*")
    backend.upload_files([])
    backend.download_files([])

    await backend.aexecute("pwd")
    await backend.als("/")
    await backend.aread("/x")
    await backend.awrite("/x", "x")
    await backend.aedit("/x", "x", "y")
    await backend.adelete("/x")
    await backend.agrep("x", max_count=3)
    await backend.aglob("*")
    await backend.aupload_files([])
    await backend.adownload_files([])

    assert concrete.execute.call_count == 2
    assert concrete.download_files.call_count == 2
    assert concrete.grep.call_args_list[0].kwargs == {"max_count": 7}
    assert concrete.grep.call_args_list[1].kwargs == {"max_count": 3}


def test_current_deepagents_graph_invocation_uses_session_backend() -> None:
    provider = RecordingProvider()
    backend = make_backend(provider)
    graph = create_deep_agent(
        model=ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"file_path": "/deepagents.txt", "content": "ok"},
                            "id": "write-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        backend=backend,
    )
    result_state = graph.invoke(
        {"messages": [("user", "write the file")]},
        config={"configurable": {"thread_id": "deepagents-current"}},
    )

    assert result_state["messages"][-1].content == "done"
    assert provider.sandboxes["deepagents-current"].files.write_calls == [
        ("deepagents.txt", b"ok"),
    ]
