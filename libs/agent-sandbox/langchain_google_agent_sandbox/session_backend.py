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

"""Durable, session-aware DeepAgents backend for agent-sandbox."""

from __future__ import annotations

import asyncio
import atexit
import base64
import hashlib
import hmac
import ipaddress
import logging
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
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

from langchain_google_agent_sandbox import _compat
from langchain_google_agent_sandbox._compat import SESSION_LABEL_KEY
from langchain_google_agent_sandbox.backend import AgentSandboxBackend

SessionResolver = Callable[[Mapping[str, Any]], str]
SessionHook = Callable[[str, AgentSandboxBackend], None]
BeforeDeleteHook = Callable[[str, AgentSandboxBackend | None], None]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionLifecycleContext:
    """Immutable identity and routing data for one session operation."""

    raw_session_id: str
    opaque_session_id: str
    claim_name: str
    namespace: str
    warm_pool: str


BeforeSessionAcquireHook = Callable[[SessionLifecycleContext], None]
AfterSessionAcquireHook = Callable[
    [SessionLifecycleContext, AgentSandboxBackend, bool], None
]
SessionAcquireErrorHook = Callable[[SessionLifecycleContext, Exception], None]
SessionAccessedHook = Callable[[SessionLifecycleContext], None]
AfterSessionDeletedHook = Callable[[SessionLifecycleContext], None]
SessionDeleteErrorHook = Callable[[SessionLifecycleContext, Exception], None]


@dataclass(frozen=True)
class SessionSandboxEndpoint:
    """Connection metadata for one session sandbox endpoint."""

    host: str
    port: int
    claim_name: str
    sandbox_id: str
    namespace: str

    @property
    def url(self) -> str:
        """Return an HTTP URL, bracketing IPv6 hosts per RFC 3986."""
        host = self.host
        try:
            if ipaddress.ip_address(host).version == 6:
                host = f"[{host}]"
        except ValueError:
            pass
        return f"http://{host}:{self.port}"


@dataclass
class _SessionEntry:
    backend: AgentSandboxBackend
    sandbox: Any
    claim_name: str
    last_renewed_at: float | None = None
    last_accessed_at: float = field(default_factory=time.monotonic)


def default_session_resolver(config: Mapping[str, Any]) -> str:
    """Resolve ``configurable.thread_id`` from the active LangGraph config."""
    configurable = config.get("configurable", {})
    if not isinstance(configurable, Mapping):
        msg = "LangGraph configurable must be a mapping"
        raise RuntimeError(msg)
    thread_id = configurable.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        msg = "LangGraph configurable.thread_id is required for sandbox sessions"
        raise RuntimeError(msg)
    return thread_id


class SessionAgentSandboxBackend(SandboxBackendProtocol):
    """Long-lived DeepAgents backend multiplexed by opaque session identity.

    The adapter derives a deterministic Claim name from the active LangGraph
    session, acquires it through the SDK's atomic get-or-create operation, and
    caches one concrete :class:`AgentSandboxBackend` per opaque session in the
    current process. Closing this adapter never deletes Claims.
    """

    SESSION_LABEL_KEY = SESSION_LABEL_KEY

    def __init__(
        self,
        client: Any,
        warm_pool: str,
        session_secret: str | bytes,
        *,
        namespace: str = "default",
        session_resolver: SessionResolver | None = None,
        root_dir: str = "/workspace",
        runtime_root: str = "/workspace",
        allow_absolute_paths: bool = False,
        sandbox_ready_timeout: int = 180,
        labels: dict[str, str] | None = None,
        idle_ttl_seconds: int | None = None,
        renewal_threshold_seconds: int | None = None,
        default_timeout_seconds: int | None = 120,
        local_cache_ttl_seconds: int | None = 300,
        max_cached_sessions: int = 128,
        legacy_label_fallback: bool = False,
        on_session_created: SessionHook | None = None,
        on_session_attached: SessionHook | None = None,
        before_session_deleted: BeforeDeleteHook | None = None,
        before_session_acquire: BeforeSessionAcquireHook | None = None,
        after_session_acquire: AfterSessionAcquireHook | None = None,
        on_session_acquire_error: SessionAcquireErrorHook | None = None,
        on_session_accessed: SessionAccessedHook | None = None,
        after_session_deleted: AfterSessionDeletedHook | None = None,
        on_session_delete_error: SessionDeleteErrorHook | None = None,
    ) -> None:
        if not warm_pool:
            msg = "warm_pool cannot be empty"
            raise ValueError(msg)
        secret = (
            session_secret.encode("utf-8")
            if isinstance(session_secret, str)
            else session_secret
        )
        if not isinstance(secret, bytes) or not secret:
            msg = "session_secret must be non-empty bytes or text"
            raise ValueError(msg)
        if idle_ttl_seconds is not None:
            if type(idle_ttl_seconds) is not int or idle_ttl_seconds <= 0:
                msg = "idle_ttl_seconds must be a positive integer"
                raise ValueError(msg)
            if renewal_threshold_seconds is None:
                renewal_threshold_seconds = max(1, idle_ttl_seconds // 4)
            if (
                type(renewal_threshold_seconds) is not int
                or renewal_threshold_seconds <= 0
                or renewal_threshold_seconds >= idle_ttl_seconds
            ):
                msg = (
                    "renewal_threshold_seconds must be positive and smaller "
                    "than idle_ttl_seconds"
                )
                raise ValueError(msg)
        elif renewal_threshold_seconds is not None:
            msg = "renewal_threshold_seconds requires idle_ttl_seconds"
            raise ValueError(msg)
        if local_cache_ttl_seconds is not None and (
            type(local_cache_ttl_seconds) is not int or local_cache_ttl_seconds <= 0
        ):
            msg = "local_cache_ttl_seconds must be a positive integer or None"
            raise ValueError(msg)
        if type(max_cached_sessions) is not int or max_cached_sessions <= 0:
            msg = "max_cached_sessions must be a positive integer"
            raise ValueError(msg)
        if labels and self.SESSION_LABEL_KEY in labels:
            msg = f"labels must not override {self.SESSION_LABEL_KEY}"
            raise ValueError(msg)

        self._client = client
        self._warm_pool = warm_pool
        self._secret = secret
        self._namespace = namespace
        self._session_resolver = session_resolver or default_session_resolver
        self._root_dir = root_dir
        self._runtime_root = runtime_root
        self._allow_absolute_paths = allow_absolute_paths
        self._sandbox_ready_timeout = sandbox_ready_timeout
        self._labels = dict(labels or {})
        self._idle_ttl_seconds = idle_ttl_seconds
        self._renewal_threshold_seconds = renewal_threshold_seconds
        self._default_timeout_seconds = default_timeout_seconds
        self._local_cache_ttl_seconds = local_cache_ttl_seconds
        self._max_cached_sessions = max_cached_sessions
        self._legacy_label_fallback = legacy_label_fallback
        self._on_session_created = on_session_created
        self._on_session_attached = on_session_attached
        self._before_session_deleted = before_session_deleted
        self._before_session_acquire = before_session_acquire
        self._after_session_acquire = after_session_acquire
        self._on_session_acquire_error = on_session_acquire_error
        self._on_session_accessed = on_session_accessed
        self._after_session_deleted = after_session_deleted
        self._on_session_delete_error = on_session_delete_error
        self._entries: dict[str, _SessionEntry] = {}
        self._session_locks: weakref.WeakValueDictionary[str, threading.RLock] = (
            weakref.WeakValueDictionary()
        )
        self._cache_lock = threading.RLock()
        self._closed = False
        atexit.register(self.close)

    @staticmethod
    def _active_config() -> Mapping[str, Any]:
        from langgraph.config import get_config

        try:
            return get_config()
        except RuntimeError as error:
            msg = "No active LangGraph config is available for sandbox resolution"
            raise RuntimeError(msg) from error

    def opaque_session_id(self, session_id: str | None = None) -> str:
        """Return the stable opaque identifier for an explicit or active session."""
        return self._resolve_context(session_id).opaque_session_id

    def _resolve_context(
        self, session_id: str | None = None
    ) -> SessionLifecycleContext:
        """Resolve all immutable session data exactly once for an operation."""
        raw_identity = session_id
        if raw_identity is None:
            raw_identity = self._session_resolver(self._active_config())
        if not isinstance(raw_identity, str) or not raw_identity:
            msg = "session resolver must return non-empty text"
            raise RuntimeError(msg)
        digest = hmac.new(
            self._secret,
            raw_identity.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        opaque_session_id = base64.b32encode(digest).decode("ascii").rstrip("=").lower()
        return SessionLifecycleContext(
            raw_session_id=raw_identity,
            opaque_session_id=opaque_session_id,
            claim_name=self._claim_name(opaque_session_id),
            namespace=self._namespace,
            warm_pool=self._warm_pool,
        )

    @staticmethod
    def _claim_name(opaque_session_id: str) -> str:
        return f"session-{opaque_session_id}"

    def _lock_for(self, opaque_session_id: str) -> threading.RLock:
        with self._cache_lock:
            return self._session_locks.setdefault(opaque_session_id, threading.RLock())

    def _legacy_claim(self, opaque_session_id: str, claim_name: str) -> str | None:
        if not self._legacy_label_fallback:
            return None
        selector = f"{self.SESSION_LABEL_KEY}={opaque_session_id}"
        matches = _compat.list_sandbox_claims(
            self._client,
            namespace=self._namespace,
            label_selector=selector,
        )
        if len(matches) > 1:
            msg = (
                "Refusing session acquisition because multiple Claims match "
                f"opaque session {opaque_session_id!r}"
            )
            raise RuntimeError(msg)
        if matches and matches[0] != claim_name:
            try:
                self._client.get_sandbox_claim_warmpool_name(
                    claim_name, self._namespace
                )
            except SandboxNotFoundError:
                return matches[0]
            msg = (
                "Refusing session acquisition because both deterministic and "
                f"legacy Claims exist for opaque session {opaque_session_id!r}"
            )
            raise RuntimeError(msg)
        return None

    def _wrap(self, sandbox: Any) -> AgentSandboxBackend:
        return AgentSandboxBackend.from_existing(
            sandbox,
            root_dir=self._root_dir,
            allow_absolute_paths=self._allow_absolute_paths,
            runtime_root=self._runtime_root,
            default_timeout_seconds=self._default_timeout_seconds,
        )

    @staticmethod
    def _notify_error(
        callback: SessionAcquireErrorHook | SessionDeleteErrorHook | None,
        context: SessionLifecycleContext,
        error: Exception,
        *,
        operation: str,
    ) -> None:
        if callback is None:
            return
        try:
            callback(context, error)
        except Exception:
            logger.exception(
                "Session %s error callback failed for opaque session %s",
                operation,
                context.opaque_session_id,
            )

    def _acquire(self, context: SessionLifecycleContext) -> _SessionEntry:
        created = False
        backend: AgentSandboxBackend | None = None
        claim_name = context.claim_name
        try:
            if self._before_session_acquire is not None:
                self._before_session_acquire(context)
            legacy_claim = self._legacy_claim(context.opaque_session_id, claim_name)
            if legacy_claim is not None:
                sandbox = _compat.get_sandbox(
                    self._client,
                    claim_name=legacy_claim,
                    namespace=context.namespace,
                    warm_pool=context.warm_pool,
                )
                claim_name = legacy_claim
            else:
                labels = dict(self._labels)
                acquisition = self._client.get_or_create_sandbox(
                    warmpool=context.warm_pool,
                    namespace=context.namespace,
                    sandbox_ready_timeout=self._sandbox_ready_timeout,
                    labels=labels or None,
                    claim_name=claim_name,
                    required_labels={self.SESSION_LABEL_KEY: context.opaque_session_id},
                    shutdown_after_seconds=self._idle_ttl_seconds,
                )
                sandbox = acquisition.sandbox
                created = acquisition.created

            backend = self._wrap(sandbox)
            entry = _SessionEntry(
                backend=backend,
                sandbox=sandbox,
                claim_name=claim_name,
                last_accessed_at=time.monotonic(),
            )
            self._renew(entry, force=True)
            if created:
                if self._on_session_created is not None:
                    self._on_session_created(context.opaque_session_id, backend)
            elif self._on_session_attached is not None:
                self._on_session_attached(context.opaque_session_id, backend)
            if self._after_session_acquire is not None:
                self._after_session_acquire(context, backend, created)
            return entry
        except Exception as error:
            if created:
                try:
                    _compat.delete_sandbox(
                        self._client,
                        claim_name=claim_name,
                        namespace=context.namespace,
                    )
                except Exception:
                    logger.exception(
                        "Failed to clean up newly created Claim %s",
                        claim_name,
                    )
            if backend is not None:
                backend.close()
            self._notify_error(
                self._on_session_acquire_error,
                context,
                error,
                operation="acquire",
            )
            raise

    def _renew(self, entry: _SessionEntry, *, force: bool = False) -> None:
        if self._idle_ttl_seconds is None:
            return
        now = time.monotonic()
        threshold = self._renewal_threshold_seconds
        if threshold is None:
            msg = "renewal threshold is not initialized"
            raise RuntimeError(msg)
        renewal_interval = self._idle_ttl_seconds - threshold
        if (
            not force
            and entry.last_renewed_at is not None
            and now - entry.last_renewed_at < renewal_interval
        ):
            return
        self._client.renew_sandbox(
            entry.claim_name,
            self._namespace,
            self._idle_ttl_seconds,
        )
        entry.last_renewed_at = now

    def _evict_idle_entries(self, *, exclude: str | None = None) -> None:
        ttl = self._local_cache_ttl_seconds
        with self._cache_lock:
            entries = list(self._entries.items())
        cutoff = time.monotonic() - ttl if ttl is not None else None
        candidates = {
            opaque_session_id
            for opaque_session_id, entry in entries
            if opaque_session_id != exclude
            and cutoff is not None
            and entry.last_accessed_at <= cutoff
        }
        current_is_cached = any(
            opaque_session_id == exclude for opaque_session_id, _ in entries
        )
        projected_size = len(entries) + (0 if current_is_cached else 1)
        overflow = max(0, projected_size - self._max_cached_sessions)
        capacity_candidates: set[str] = set()
        if overflow:
            oldest = sorted(
                (
                    (entry.last_accessed_at, opaque_session_id)
                    for opaque_session_id, entry in entries
                    if opaque_session_id != exclude
                    and opaque_session_id not in candidates
                ),
            )
            capacity_candidates = {
                opaque_session_id for _, opaque_session_id in oldest[:overflow]
            }
            candidates.update(capacity_candidates)
        for opaque_session_id in candidates:
            with self._lock_for(opaque_session_id):
                with self._cache_lock:
                    entry = self._entries.get(opaque_session_id)
                    if entry is None:
                        continue
                    if (
                        cutoff is not None
                        and opaque_session_id not in capacity_candidates
                        and entry.last_accessed_at > cutoff
                    ):
                        continue
                    self._entries.pop(opaque_session_id, None)
                entry.backend.close()

    def _entry_locked(self, context: SessionLifecycleContext) -> _SessionEntry:
        if self._closed:
            msg = "SessionAgentSandboxBackend is closed"
            raise RuntimeError(msg)
        with self._cache_lock:
            entry = self._entries.get(context.opaque_session_id)
        if entry is not None:
            try:
                self._renew(entry)
                entry.last_accessed_at = time.monotonic()
                if self._on_session_accessed is not None:
                    self._on_session_accessed(context)
                return entry
            except SandboxNotFoundError:
                entry.backend.close()
                with self._cache_lock:
                    self._entries.pop(context.opaque_session_id, None)
        entry = self._acquire(context)
        entry.last_accessed_at = time.monotonic()
        with self._cache_lock:
            self._entries[context.opaque_session_id] = entry
        if self._on_session_accessed is not None:
            self._on_session_accessed(context)
        return entry

    def _entry(self, session_id: str | None = None) -> tuple[str, _SessionEntry]:
        context = self._resolve_context(session_id)
        self._evict_idle_entries(exclude=context.opaque_session_id)
        with self._lock_for(context.opaque_session_id):
            entry = self._entry_locked(context)
            return context.opaque_session_id, entry

    def _invoke(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        context = self._resolve_context()
        self._evict_idle_entries(exclude=context.opaque_session_id)
        with self._lock_for(context.opaque_session_id):
            entry = self._entry_locked(context)
            try:
                return getattr(entry.backend, method_name)(*args, **kwargs)
            except SandboxNotFoundError:
                self._invalidate(context.opaque_session_id)
                replacement = self._entry_locked(context)
                return getattr(replacement.backend, method_name)(*args, **kwargs)

    def _invalidate(self, opaque_session_id: str) -> None:
        with self._cache_lock:
            entry = self._entries.pop(opaque_session_id, None)
        if entry is not None:
            entry.backend.close()

    def close_session(self, session_id: str) -> None:
        """Close local connectors for one session without deleting its Claim."""
        context = self._resolve_context(session_id)
        with self._lock_for(context.opaque_session_id):
            self._invalidate(context.opaque_session_id)

    def delete_session(self, session_id: str) -> None:
        """Run the delete hook and explicitly delete one session Claim."""
        context = self._resolve_context(session_id)
        claim_name = context.claim_name
        with self._lock_for(context.opaque_session_id):
            try:
                with self._cache_lock:
                    entry = self._entries.get(context.opaque_session_id)
                if self._before_session_deleted is not None:
                    self._before_session_deleted(
                        context.opaque_session_id,
                        entry.backend if entry is not None else None,
                    )
                if entry is not None:
                    claim_name = entry.claim_name
                else:
                    legacy = self._legacy_claim(context.opaque_session_id, claim_name)
                    if legacy is not None:
                        claim_name = legacy
                _compat.delete_sandbox(
                    self._client,
                    claim_name=claim_name,
                    namespace=context.namespace,
                )
                self._invalidate(context.opaque_session_id)
                if self._after_session_deleted is not None:
                    self._after_session_deleted(context)
            except Exception as error:
                self._notify_error(
                    self._on_session_delete_error,
                    context,
                    error,
                    operation="delete",
                )
                raise

    def close(self) -> None:
        """Close every local connector without deleting Kubernetes Claims."""
        with self._cache_lock:
            if self._closed:
                return
            self._closed = True
            entries = list(self._entries.values())
            self._entries.clear()
        atexit.unregister(self.close)
        for entry in entries:
            entry.backend.close()

    def get_session_sandbox(self, session_id: str | None = None) -> Any:
        """Return the SDK Sandbox handle for an explicit or active session."""
        return self._entry(session_id)[1].sandbox

    def get_session_claim_name(self, session_id: str | None = None) -> str:
        """Return the acquired Claim name for an explicit or active session."""
        return self._entry(session_id)[1].claim_name

    def get_session_endpoint(
        self,
        port: int,
        prefer_pod_ip: bool = False,
        *,
        session_id: str | None = None,
    ) -> SessionSandboxEndpoint:
        """Return service or current Pod endpoint metadata for a session."""
        if type(port) is not int or not 1 <= port <= 65535:
            msg = "port must be an integer between 1 and 65535"
            raise ValueError(msg)
        entry = self._entry(session_id)[1]
        host = entry.sandbox.service_host
        if prefer_pod_ip:
            host = entry.sandbox.get_pod_ip() or host
        return SessionSandboxEndpoint(
            host=host,
            port=port,
            claim_name=entry.claim_name,
            sandbox_id=entry.sandbox.sandbox_id,
            namespace=entry.sandbox.namespace,
        )

    @property
    def id(self) -> str:
        """Return the current session's namespace and Claim name."""
        return f"{self._namespace}/{self.get_session_claim_name()}"

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
        """Recursively delete a path from the current session sandbox."""
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
