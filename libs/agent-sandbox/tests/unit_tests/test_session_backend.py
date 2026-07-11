from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from k8s_agent_sandbox.acquisition import SandboxAcquisition
from k8s_agent_sandbox.exceptions import SandboxNotFoundError

from langchain_google_agent_sandbox import (
    AgentSandboxBackend,
    SessionAgentSandboxBackend,
    SessionSandboxEndpoint,
    default_session_resolver,
)
from langchain_google_agent_sandbox._compat import SESSION_LABEL_KEY
from langchain_google_agent_sandbox.session_backend import _SessionEntry
from tests.unit_tests.test_backend import StubSandbox, result


class SessionStubClient:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sandboxes: dict[str, StubSandbox] = {}
        self.labels: dict[str, dict[str, str]] = {}
        self.warm_pools: dict[str, str] = {}
        self.acquire_count = 0
        self.deleted: list[tuple[str, str]] = []
        self.renewed: list[tuple[str, str, int]] = []

    def get_or_create_sandbox(
        self,
        *,
        warmpool: str,
        namespace: str,
        sandbox_ready_timeout: int,
        labels: dict[str, str] | None,
        claim_name: str,
        required_labels: dict[str, str],
        shutdown_after_seconds: int | None,
    ) -> SandboxAcquisition[StubSandbox]:
        del sandbox_ready_timeout, shutdown_after_seconds
        with self.lock:
            self.acquire_count += 1
            created = claim_name not in self.sandboxes
            sandbox = self.sandboxes.get(claim_name)
            if sandbox is None or sandbox.closed:
                sandbox = StubSandbox(claim_name=claim_name, namespace=namespace)
                self.sandboxes[claim_name] = sandbox
            if created:
                self.labels[claim_name] = {**(labels or {}), **required_labels}
                self.warm_pools[claim_name] = warmpool
            else:
                assert self.warm_pools[claim_name] == warmpool
                assert all(
                    self.labels[claim_name].get(key) == value
                    for key, value in required_labels.items()
                )
            return SandboxAcquisition(
                sandbox=sandbox,
                created=created,
                claim_name=claim_name,
                sandbox_id=sandbox.sandbox_id,
                namespace=namespace,
            )

    def list_all_sandboxes(
        self, namespace: str, label_selector: str | None = None
    ) -> list[str]:
        del namespace
        if label_selector is None:
            return list(self.sandboxes)
        key, value = label_selector.split("=", 1)
        return [
            claim for claim, labels in self.labels.items() if labels.get(key) == value
        ]

    def get_sandbox_claim_warmpool_name(self, claim_name: str, namespace: str) -> str:
        del namespace
        if claim_name not in self.warm_pools:
            raise SandboxNotFoundError(claim_name)
        return self.warm_pools[claim_name]

    def get_sandbox(self, claim_name: str, namespace: str) -> StubSandbox:
        sandbox = self.sandboxes[claim_name]
        if sandbox.closed:
            sandbox = StubSandbox(claim_name=claim_name, namespace=namespace)
            self.sandboxes[claim_name] = sandbox
        return sandbox

    def renew_sandbox(
        self, claim_name: str, namespace: str, shutdown_after_seconds: int
    ) -> None:
        if claim_name not in self.sandboxes:
            raise SandboxNotFoundError(claim_name)
        self.renewed.append((claim_name, namespace, shutdown_after_seconds))

    def delete_sandbox(self, claim_name: str, namespace: str) -> None:
        self.deleted.append((claim_name, namespace))
        self.sandboxes.pop(claim_name, None)
        self.labels.pop(claim_name, None)
        self.warm_pools.pop(claim_name, None)

    def add_legacy(self, claim_name: str, opaque_id: str) -> None:
        self.sandboxes[claim_name] = StubSandbox(claim_name=claim_name)
        self.labels[claim_name] = {SESSION_LABEL_KEY: opaque_id}
        self.warm_pools[claim_name] = "python"


def make_backend(
    client: SessionStubClient | None = None,
    **kwargs: Any,
) -> SessionAgentSandboxBackend:
    return SessionAgentSandboxBackend(
        client or SessionStubClient(),
        "python",
        "stable-secret",
        **kwargs,
    )


def test_default_and_custom_session_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {"configurable": {"thread_id": "thread-1", "tenant": "tenant-a"}}
    monkeypatch.setattr(
        SessionAgentSandboxBackend,
        "_active_config",
        staticmethod(lambda: config),
    )
    backend = make_backend()
    assert backend.opaque_session_id() == backend.opaque_session_id("thread-1")

    custom = make_backend(
        session_resolver=lambda value: (
            f"{value['configurable']['tenant']}\0{value['configurable']['thread_id']}"
        )
    )
    assert custom.opaque_session_id() != custom.opaque_session_id("thread-1")
    assert default_session_resolver(config) == "thread-1"


def test_hmac_stability_privacy_and_tenant_isolation() -> None:
    first = make_backend()
    second = make_backend()
    opaque = first.opaque_session_id("tenant-a\0shared-thread")
    assert opaque == second.opaque_session_id("tenant-a\0shared-thread")
    assert opaque != first.opaque_session_id("tenant-b\0shared-thread")
    assert "tenant" not in opaque
    assert "shared-thread" not in first._claim_name(opaque)
    assert len(first._claim_name(opaque)) <= 63


def test_cross_thread_isolation_and_process_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = {"thread_id": "one"}
    monkeypatch.setattr(
        SessionAgentSandboxBackend,
        "_active_config",
        staticmethod(lambda: {"configurable": active}),
    )
    client = SessionStubClient()
    backend = make_backend(client)
    first_claim = backend.get_session_claim_name()
    active["thread_id"] = "two"
    second_claim = backend.get_session_claim_name()
    assert first_claim != second_claim

    restarted = make_backend(client)
    assert restarted.get_session_claim_name("one") == first_claim
    assert client.acquire_count == 3


def test_local_concurrent_acquisition_is_coalesced() -> None:
    client = SessionStubClient()
    backend = make_backend(client)
    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = list(
            executor.map(
                lambda _: backend.get_session_claim_name("same-session"),
                range(16),
            )
        )
    assert len(set(claims)) == 1
    assert client.acquire_count == 1


def test_legacy_fallback_and_multiple_match_refusal() -> None:
    client = SessionStubClient()
    backend = make_backend(client, legacy_label_fallback=True)
    opaque = backend.opaque_session_id("legacy")
    client.add_legacy("sandbox-claim-old", opaque)
    assert backend.get_session_claim_name("legacy") == "sandbox-claim-old"
    assert client.acquire_count == 0

    duplicate_client = SessionStubClient()
    duplicate = make_backend(duplicate_client, legacy_label_fallback=True)
    opaque = duplicate.opaque_session_id("legacy")
    duplicate_client.add_legacy("old-a", opaque)
    duplicate_client.add_legacy("old-b", opaque)
    with pytest.raises(RuntimeError, match="multiple Claims"):
        duplicate.get_session_claim_name("legacy")


def test_idle_ttl_renewal_threshold_and_not_found_reacquire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(
        "langchain_google_agent_sandbox.session_backend.time.monotonic",
        lambda: now[0],
    )
    client = SessionStubClient()
    backend = make_backend(
        client,
        idle_ttl_seconds=100,
        renewal_threshold_seconds=20,
    )
    claim = backend.get_session_claim_name("ttl-session")
    assert len(client.renewed) == 1
    now[0] = 179.0
    backend.get_session_claim_name("ttl-session")
    assert len(client.renewed) == 1
    now[0] = 180.0
    backend.get_session_claim_name("ttl-session")
    assert len(client.renewed) == 2

    client.sandboxes.pop(claim)
    now[0] = 260.0
    backend.get_session_claim_name("ttl-session")
    assert client.acquire_count == 2


def test_created_attached_delete_hooks_and_close_semantics() -> None:
    client = SessionStubClient()
    events: list[tuple[str, str]] = []
    backend = make_backend(
        client,
        on_session_created=lambda opaque, value: events.append(("created", opaque)),
        on_session_attached=lambda opaque, value: events.append(("attached", opaque)),
        before_session_deleted=lambda opaque, value: events.append(("deleted", opaque)),
    )
    claim = backend.get_session_claim_name("hook-session")
    opaque = backend.opaque_session_id("hook-session")
    assert events == [("created", opaque)]

    backend.close_session("hook-session")
    assert client.deleted == []
    backend.get_session_claim_name("hook-session")
    assert events[-1] == ("attached", opaque)

    backend.delete_session("hook-session")
    assert events[-1] == ("deleted", opaque)
    assert client.deleted == [(claim, "default")]

    backend.get_session_claim_name("other")
    backend.close()
    assert client.deleted == [(claim, "default")]


def test_created_hook_failure_deletes_new_claim() -> None:
    client = SessionStubClient()

    def fail(opaque: str, backend: AgentSandboxBackend) -> None:
        del opaque, backend
        raise RuntimeError("restore failed")

    backend = make_backend(client, on_session_created=fail)
    with pytest.raises(RuntimeError, match="restore failed"):
        backend.get_session_claim_name("broken")
    assert len(client.deleted) == 1


def test_delete_uncached_session_passes_none_to_hook() -> None:
    client = SessionStubClient()
    observed: list[AgentSandboxBackend | None] = []
    backend = make_backend(
        client,
        before_session_deleted=lambda opaque, value: observed.append(value),
    )
    expected = backend._claim_name(backend.opaque_session_id("uncached"))
    backend.delete_session("uncached")
    assert observed == [None]
    assert client.deleted == [(expected, "default")]


def test_endpoint_access_and_ipv6_url() -> None:
    backend = make_backend()
    service = backend.get_session_endpoint(5901, session_id="desktop")
    assert isinstance(service, SessionSandboxEndpoint)
    assert service.host.endswith(".svc.cluster.local")
    assert service.url.endswith(":5901")

    pod = backend.get_session_endpoint(9222, prefer_pod_ip=True, session_id="desktop")
    assert pod.host == "10.0.0.8"
    ipv6 = SessionSandboxEndpoint("2001:db8::1", 9222, "c", "s", "ns")
    assert ipv6.url == "http://[2001:db8::1]:9222"


@pytest.mark.asyncio
async def test_sync_and_async_deepagents_delegation() -> None:
    backend = make_backend()
    opaque, entry = backend._entry("delegation")
    del opaque
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
    backend._entries[backend.opaque_session_id("delegation")] = _SessionEntry(
        backend=concrete,
        sandbox=entry.sandbox,
        claim_name=entry.claim_name,
    )
    backend._active_config = lambda: {"configurable": {"thread_id": "delegation"}}

    backend.execute("pwd")
    backend.ls("/")
    backend.read("/x")
    backend.write("/x", "x")
    backend.edit("/x", "x", "y")
    backend.delete("/x")
    backend.grep("x")
    backend.glob("*")
    backend.upload_files([])
    backend.download_files([])

    await backend.aexecute("pwd")
    await backend.als("/")
    await backend.aread("/x")
    await backend.awrite("/x", "x")
    await backend.aedit("/x", "x", "y")
    await backend.adelete("/x")
    await backend.agrep("x")
    await backend.aglob("*")
    await backend.aupload_files([])
    await backend.adownload_files([])

    assert concrete.execute.call_count == 2
    assert concrete.download_files.call_count == 2


def test_direct_composite_and_middleware_integration() -> None:
    backend = make_backend()
    assert isinstance(backend, SandboxBackendProtocol)
    composite = CompositeBackend(default=backend, routes={})
    assert composite.default is backend
    assert FilesystemMiddleware(backend=backend)
    assert SkillsMiddleware(backend=backend, sources=[])
