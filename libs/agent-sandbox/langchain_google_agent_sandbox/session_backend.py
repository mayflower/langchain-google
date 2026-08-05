# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Session-aware DeepAgents backend over a product-owned lease provider."""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
import time
import uuid
import warnings
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from k8s_agent_sandbox.exceptions import SandboxNotFoundError

from langchain_google_agent_sandbox.backend import AgentSandboxBackend
from langchain_google_agent_sandbox.limits import SandboxResultLimits
from langchain_google_agent_sandbox.provider import SandboxLease, SandboxSessionProvider

SessionResolver = Callable[[Mapping[str, Any]], str]

logger = logging.getLogger(__name__)

__all__ = [
    "ProviderSessionAgentSandboxBackend",
    "SessionAgentSandboxBackend",
    "default_session_resolver",
]


def default_session_resolver(config: Mapping[str, Any]) -> str:
    """Resolve ``configurable.thread_id`` from the active LangGraph config.

    This is a pure local read used only to look up a process-local cache entry.
    It derives no Kubernetes identity: the mapping from session to durable
    infrastructure belongs entirely to the provider.

    Args:
        config: The active LangGraph config mapping.

    Returns:
        The thread identifier used as this process's cache key.

    Raises:
        RuntimeError: If the config carries no usable ``thread_id``.
    """
    configurable = config.get("configurable", {})
    if not isinstance(configurable, Mapping):
        msg = "LangGraph configurable must be a mapping"
        raise RuntimeError(msg)
    thread_id = configurable.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        msg = "LangGraph configurable.thread_id is required for sandbox sessions"
        raise RuntimeError(msg)
    return thread_id


def _make_atexit_hook(backend_ref: weakref.ref[Any]) -> Callable[[], None]:
    """Build a shutdown hook that does not keep its backend alive."""

    def hook() -> None:
        backend = backend_ref()
        if backend is not None:
            backend.close()

    return hook


@dataclass
class _SessionEntry:
    """One process-local cached session."""

    backend: AgentSandboxBackend
    lease: SandboxLease
    last_accessed_at: float
    last_touched_at: float


class ProviderSessionAgentSandboxBackend(SandboxBackendProtocol):
    """DeepAgents backend multiplexed over provider-supplied sandbox leases.

    The adapter owns exactly five things: protocol translation, path
    normalization, a process-local lease cache, reattachment after a stale
    local handle, and local connector shutdown.

    It owns no durable lifecycle. It never creates, renames, labels, renews,
    lists, or deletes a Kubernetes Claim, never derives a Claim name, and never
    hashes product identity. Every acquisition goes through
    :meth:`SandboxSessionProvider.acquire`, which is free to reject it.

    Closing this backend, or evicting a cached session, releases only local
    client resources. Remote infrastructure is untouched.
    """

    def __init__(
        self,
        provider: SandboxSessionProvider,
        *,
        session_resolver: SessionResolver | None = None,
        root_dir: str = "/workspace",
        runtime_root: str = "/workspace",
        allow_absolute_paths: bool = False,
        default_timeout_seconds: int | None = 120,
        local_cache_ttl_seconds: int | None = 300,
        max_cached_sessions: int = 128,
        touch_interval_seconds: float = 60.0,
        limits: SandboxResultLimits | None = None,
    ) -> None:
        if provider is None:
            msg = "provider is required; see SandboxSessionProvider"
            raise ValueError(msg)
        if local_cache_ttl_seconds is not None and (
            type(local_cache_ttl_seconds) is not int or local_cache_ttl_seconds <= 0
        ):
            msg = "local_cache_ttl_seconds must be a positive integer or None"
            raise ValueError(msg)
        if type(max_cached_sessions) is not int or max_cached_sessions <= 0:
            msg = "max_cached_sessions must be a positive integer"
            raise ValueError(msg)
        if (
            isinstance(touch_interval_seconds, bool)
            or not isinstance(touch_interval_seconds, (int, float))
            or touch_interval_seconds < 0
        ):
            msg = "touch_interval_seconds must be a non-negative number"
            raise ValueError(msg)

        self._provider = provider
        self._session_resolver = session_resolver or default_session_resolver
        self._root_dir = root_dir
        self._runtime_root = runtime_root
        self._allow_absolute_paths = allow_absolute_paths
        self._default_timeout_seconds = default_timeout_seconds
        self._local_cache_ttl_seconds = local_cache_ttl_seconds
        self._max_cached_sessions = max_cached_sessions
        self._touch_interval_seconds = float(touch_interval_seconds)
        self._limits = limits
        self._entries: dict[str, _SessionEntry] = {}
        # Reverse index enforcing that one lease never backs two session keys.
        self._lease_owners: dict[str, str] = {}
        self._session_locks: weakref.WeakValueDictionary[str, threading.RLock] = (
            weakref.WeakValueDictionary()
        )
        self._cache_lock = threading.RLock()
        self._closed = False
        self._id = f"agent-sandbox/{uuid.uuid4()}"
        # Registering the bound method would make atexit hold a strong
        # reference, keeping every backend -- and its open connections --
        # alive for the life of the process. Hold a weak reference instead,
        # in a per-instance closure so close() can unregister exactly this one.
        self._atexit_hook = _make_atexit_hook(weakref.ref(self))
        atexit.register(self._atexit_hook)

    @staticmethod
    def _active_config() -> Mapping[str, Any]:
        from langgraph.config import get_config

        try:
            return get_config()
        except RuntimeError as error:
            msg = "No active LangGraph config is available for sandbox resolution"
            raise RuntimeError(msg) from error

    def _resolve(
        self, config: Mapping[str, Any] | None = None
    ) -> tuple[str, Mapping[str, Any]]:
        """Return the cache key and the config that produced it."""
        resolved = self._active_config() if config is None else config
        key = self._session_resolver(resolved)
        if not isinstance(key, str) or not key:
            msg = "session resolver must return non-empty text"
            raise RuntimeError(msg)
        return key, resolved

    def _lock_for(self, session_key: str) -> threading.RLock:
        with self._cache_lock:
            return self._session_locks.setdefault(session_key, threading.RLock())

    def _wrap(self, sandbox: Any) -> AgentSandboxBackend:
        return AgentSandboxBackend.from_existing(
            sandbox,
            root_dir=self._root_dir,
            allow_absolute_paths=self._allow_absolute_paths,
            runtime_root=self._runtime_root,
            default_timeout_seconds=self._default_timeout_seconds,
            limits=self._limits,
        )

    def _acquire(self, session_key: str, config: Mapping[str, Any]) -> _SessionEntry:
        """Obtain a lease from the provider and wrap it for DeepAgents use.

        Provider errors propagate unchanged: admission refusals, capacity
        errors, and terminal claim failures are the product's to describe, and
        wrapping them would hide the reason from the caller.
        """
        lease = self._provider.acquire(config)
        if not isinstance(lease, SandboxLease):
            msg = (
                "provider.acquire must return a SandboxLease, got "
                f"{type(lease).__name__}"
            )
            raise TypeError(msg)
        now = time.monotonic()
        return _SessionEntry(
            backend=self._wrap(lease.sandbox),
            lease=lease,
            last_accessed_at=now,
            last_touched_at=now,
        )

    def _touch(self, entry: _SessionEntry, now: float) -> None:
        """Report activity at most once per configured interval."""
        if now - entry.last_touched_at < self._touch_interval_seconds:
            return
        entry.last_touched_at = now
        try:
            self._provider.touch(entry.lease)
        except Exception:
            # touch() is advisory. A control plane hiccup must not fail the
            # agent's file operation; the provider sees its own error.
            logger.exception("provider.touch failed for lease %s", entry.lease.key)

    def _release(self, entry: _SessionEntry) -> None:
        """Release local resources for one entry. Never touches remote state."""
        try:
            entry.backend.close()
        finally:
            try:
                self._provider.close_local(entry.lease)
            except Exception:
                logger.exception(
                    "provider.close_local failed for lease %s", entry.lease.key
                )

    def _evict_idle_entries(self, *, exclude: str | None = None) -> None:
        ttl = self._local_cache_ttl_seconds
        with self._cache_lock:
            entries = list(self._entries.items())
        cutoff = time.monotonic() - ttl if ttl is not None else None
        candidates = {
            session_key
            for session_key, entry in entries
            if session_key != exclude
            and cutoff is not None
            and entry.last_accessed_at <= cutoff
        }
        current_is_cached = any(session_key == exclude for session_key, _ in entries)
        projected_size = len(entries) + (0 if current_is_cached else 1)
        overflow = max(0, projected_size - self._max_cached_sessions)
        capacity_candidates: set[str] = set()
        if overflow:
            oldest = sorted(
                (entry.last_accessed_at, session_key)
                for session_key, entry in entries
                if session_key != exclude and session_key not in candidates
            )
            capacity_candidates = {session_key for _, session_key in oldest[:overflow]}
            candidates.update(capacity_candidates)
        for session_key in candidates:
            with self._lock_for(session_key):
                with self._cache_lock:
                    entry = self._entries.get(session_key)
                    if entry is None:
                        continue
                    if (
                        cutoff is not None
                        and session_key not in capacity_candidates
                        and entry.last_accessed_at > cutoff
                    ):
                        continue
                    self._entries.pop(session_key, None)
                    self._lease_owners.pop(entry.lease.key, None)
                self._release(entry)

    def _entry_locked(
        self, session_key: str, config: Mapping[str, Any]
    ) -> _SessionEntry:
        if self._closed:
            msg = "ProviderSessionAgentSandboxBackend is closed"
            raise RuntimeError(msg)
        with self._cache_lock:
            entry = self._entries.get(session_key)
        if entry is not None:
            now = time.monotonic()
            entry.last_accessed_at = now
            self._touch(entry, now)
            return entry
        entry = self._acquire(session_key, config)
        # Claiming the lease and publishing the entry must happen in one
        # critical section. Two session keys hold *different* per-session
        # locks, so a check performed outside this block could pass for both
        # before either published, leaving two sessions on one sandbox.
        with self._cache_lock:
            owner = self._lease_owners.get(entry.lease.key)
            if owner is None or owner == session_key:
                self._entries[session_key] = entry
                self._lease_owners[entry.lease.key] = session_key
                return entry
        # Losing the race is not an error the caller can retry into safety:
        # the provider handed the same sandbox to two sessions, which would
        # cross-contaminate their filesystems. Release what we just opened.
        self._release(entry)
        msg = (
            f"provider returned lease {entry.lease.key!r} for session "
            f"{session_key!r}, but it is already held by session {owner!r}"
        )
        raise RuntimeError(msg)

    def _invalidate(self, session_key: str) -> None:
        with self._cache_lock:
            entry = self._entries.pop(session_key, None)
            if entry is not None:
                self._lease_owners.pop(entry.lease.key, None)
        if entry is not None:
            self._release(entry)

    def _entry(self, config: Mapping[str, Any] | None = None) -> _SessionEntry:
        session_key, resolved = self._resolve(config)
        self._evict_idle_entries(exclude=session_key)
        with self._lock_for(session_key):
            return self._entry_locked(session_key, resolved)

    def _invoke(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        session_key, config = self._resolve()
        self._evict_idle_entries(exclude=session_key)
        with self._lock_for(session_key):
            entry = self._entry_locked(session_key, config)
            try:
                return getattr(entry.backend, method_name)(*args, **kwargs)
            except SandboxNotFoundError:
                # The sandbox vanished under a cached handle. Drop it and
                # re-acquire exactly once. A second failure is genuine and
                # propagates rather than starting a retry loop.
                #
                # This catch is deliberately narrow. Sibling SDK errors --
                # notably SandboxClaimFailedError, which reports a terminal
                # Ready=False reason the controller will not retry -- must
                # propagate untouched instead of driving a pointless re-acquire.
                self._invalidate(session_key)
                replacement = self._entry_locked(session_key, config)
                return getattr(replacement.backend, method_name)(*args, **kwargs)

    def get_session_lease(
        self, config: Mapping[str, Any] | None = None
    ) -> SandboxLease:
        """Return the provider lease backing an explicit or active session."""
        return self._entry(config).lease

    def close_session(self, config: Mapping[str, Any] | None = None) -> None:
        """Release local resources for one session without deleting anything."""
        session_key, _ = self._resolve(config)
        with self._lock_for(session_key):
            self._invalidate(session_key)

    def close(self) -> None:
        """Release every local connector. Remote Claims are left running."""
        with self._cache_lock:
            if self._closed:
                return
            self._closed = True
            entries = list(self._entries.values())
            self._entries.clear()
            self._lease_owners.clear()
        atexit.unregister(self._atexit_hook)
        for entry in entries:
            self._release(entry)

    @property
    def id(self) -> str:
        """Return a stable identifier for this backend instance.

        The DeepAgents protocol defines this as identifying the *backend
        instance*, and this backend multiplexes many sessions, so it cannot
        name any one of them. It is also a property: resolving it must never
        reach the provider or provision a sandbox as a side effect.

        For per-session identity use :meth:`get_session_lease`.
        """
        return self._id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self._invoke("execute", command, timeout=timeout)

    def ls(self, path: str) -> LsResult:
        return self._invoke("ls", path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self._invoke("read", file_path, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        return self._invoke("write", file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return self._invoke("edit", file_path, old_string, new_string, replace_all)

    def delete(self, file_path: str) -> DeleteResult:
        """Recursively delete a path inside the current session sandbox."""
        return self._invoke("delete", file_path)

    async def adelete(self, file_path: str) -> DeleteResult:
        """Async version of :meth:`delete`."""
        return await asyncio.to_thread(self.delete, file_path)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        return self._invoke("grep", pattern, path, glob, max_count=max_count)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self._invoke("glob", pattern, path)

    def upload_files(
        self, files: dict[str, bytes] | Iterable[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        return self._invoke("upload_files", files)

    def download_files(self, paths: Iterable[str]) -> list[FileDownloadResponse]:
        return self._invoke("download_files", paths)


#: Arguments the pre-provider backend accepted, mapped to their replacement.
#: These are rejected loudly rather than absorbed by ``**kwargs``, because
#: silently ignoring, say, ``session_secret`` would leave a caller believing
#: identity is still being derived here.
_REMOVED_ARGUMENTS = {
    "client": "supply a product SandboxSessionProvider via provider=",
    "warm_pool": "provider/control-plane configuration",
    "session_secret": "provider-owned identity mapping",
    "namespace": "provider/control-plane configuration",
    "sandbox_ready_timeout": "provider/control-plane configuration",
    "labels": "provider/control-plane configuration",
    "idle_ttl_seconds": "provider-owned lease lifetime",
    "renewal_threshold_seconds": "provider-owned lease renewal",
    "legacy_label_fallback": "removed; the provider owns session discovery",
    "on_session_created": "provider workspace preparation/eventing",
    "on_session_attached": "provider workspace preparation/eventing",
    "before_session_deleted": "product control-plane close operation",
    "before_session_acquire": "provider admission",
    "after_session_acquire": "provider workspace preparation/eventing",
    "on_session_acquire_error": "provider error handling",
    "on_session_accessed": "throttled provider touch()",
    "after_session_deleted": "product control-plane close operation",
    "on_session_delete_error": "product control-plane close operation",
}


class SessionAgentSandboxBackend(ProviderSessionAgentSandboxBackend):
    """Deprecated alias for :class:`ProviderSessionAgentSandboxBackend`.

    !!! warning "Removed in a future release"
        This name exists only to give one coordinated migration commit a
        landing place. It does **not** preserve the old lifecycle behavior:
        this backend can no longer create, renew, discover, or delete Claims,
        and it no longer derives Claim names from session identity.
    """

    def __init__(self, provider: SandboxSessionProvider, **kwargs: Any) -> None:
        removed = sorted(set(kwargs) & set(_REMOVED_ARGUMENTS))
        if removed:
            details = "; ".join(
                f"{name} -> {_REMOVED_ARGUMENTS[name]}" for name in removed
            )
            msg = (
                "SessionAgentSandboxBackend no longer owns session lifecycle, so "
                f"these arguments have no effect and were rejected: {details}. "
                "See docs/adr/0001-provider-owned-session-lifecycle.md."
            )
            raise TypeError(msg)
        warnings.warn(
            "SessionAgentSandboxBackend is deprecated; use "
            "ProviderSessionAgentSandboxBackend",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(provider, **kwargs)
