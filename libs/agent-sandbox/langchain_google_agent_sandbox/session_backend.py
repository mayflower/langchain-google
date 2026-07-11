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
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from deepagents.backends.protocol import (
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
        legacy_label_fallback: bool = False,
        on_session_created: SessionHook | None = None,
        on_session_attached: SessionHook | None = None,
        before_session_deleted: BeforeDeleteHook | None = None,
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
        self._legacy_label_fallback = legacy_label_fallback
        self._on_session_created = on_session_created
        self._on_session_attached = on_session_attached
        self._before_session_deleted = before_session_deleted
        self._entries: dict[str, _SessionEntry] = {}
        self._session_locks: dict[str, threading.RLock] = {}
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
        return base64.b32encode(digest).decode("ascii").rstrip("=").lower()

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

    def _acquire(self, opaque_session_id: str) -> _SessionEntry:
        claim_name = self._claim_name(opaque_session_id)
        legacy_claim = self._legacy_claim(opaque_session_id, claim_name)
        if legacy_claim is not None:
            sandbox = _compat.get_sandbox(
                self._client,
                claim_name=legacy_claim,
                namespace=self._namespace,
                warm_pool=self._warm_pool,
            )
            created = False
            claim_name = legacy_claim
        else:
            labels = dict(self._labels)
            acquisition = self._client.get_or_create_sandbox(
                warmpool=self._warm_pool,
                namespace=self._namespace,
                sandbox_ready_timeout=self._sandbox_ready_timeout,
                labels=labels or None,
                claim_name=claim_name,
                required_labels={self.SESSION_LABEL_KEY: opaque_session_id},
                shutdown_after_seconds=self._idle_ttl_seconds,
            )
            sandbox = acquisition.sandbox
            created = acquisition.created

        backend = self._wrap(sandbox)
        entry = _SessionEntry(
            backend=backend,
            sandbox=sandbox,
            claim_name=claim_name,
        )
        try:
            self._renew(entry, force=True)
            if created:
                if self._on_session_created is not None:
                    self._on_session_created(opaque_session_id, backend)
            elif self._on_session_attached is not None:
                self._on_session_attached(opaque_session_id, backend)
        except Exception:
            if created:
                _compat.delete_sandbox(
                    self._client,
                    claim_name=claim_name,
                    namespace=self._namespace,
                )
            backend.close()
            raise
        return entry

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

    def _entry(self, session_id: str | None = None) -> tuple[str, _SessionEntry]:
        if self._closed:
            msg = "SessionAgentSandboxBackend is closed"
            raise RuntimeError(msg)
        opaque_session_id = self.opaque_session_id(session_id)
        lock = self._lock_for(opaque_session_id)
        with lock:
            with self._cache_lock:
                entry = self._entries.get(opaque_session_id)
            if entry is not None:
                try:
                    self._renew(entry)
                    return opaque_session_id, entry
                except SandboxNotFoundError:
                    entry.backend.close()
                    with self._cache_lock:
                        self._entries.pop(opaque_session_id, None)
            entry = self._acquire(opaque_session_id)
            with self._cache_lock:
                self._entries[opaque_session_id] = entry
            return opaque_session_id, entry

    def _invoke(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        opaque_session_id, entry = self._entry()
        try:
            return getattr(entry.backend, method_name)(*args, **kwargs)
        except SandboxNotFoundError:
            self._invalidate(opaque_session_id)
            _, replacement = self._entry()
            return getattr(replacement.backend, method_name)(*args, **kwargs)

    def _invalidate(self, opaque_session_id: str) -> None:
        with self._cache_lock:
            entry = self._entries.pop(opaque_session_id, None)
        if entry is not None:
            entry.backend.close()

    def close_session(self, session_id: str) -> None:
        """Close local connectors for one session without deleting its Claim."""
        opaque_session_id = self.opaque_session_id(session_id)
        with self._lock_for(opaque_session_id):
            self._invalidate(opaque_session_id)

    def delete_session(self, session_id: str) -> None:
        """Run the delete hook and explicitly delete one session Claim."""
        opaque_session_id = self.opaque_session_id(session_id)
        claim_name = self._claim_name(opaque_session_id)
        with self._lock_for(opaque_session_id):
            with self._cache_lock:
                entry = self._entries.get(opaque_session_id)
            if self._before_session_deleted is not None:
                self._before_session_deleted(
                    opaque_session_id,
                    entry.backend if entry is not None else None,
                )
            if entry is not None:
                claim_name = entry.claim_name
            else:
                legacy = self._legacy_claim(opaque_session_id, claim_name)
                if legacy is not None:
                    claim_name = legacy
            _compat.delete_sandbox(
                self._client,
                claim_name=claim_name,
                namespace=self._namespace,
            )
            self._invalidate(opaque_session_id)

    def close(self) -> None:
        """Close every local connector without deleting Kubernetes Claims."""
        with self._cache_lock:
            if self._closed:
                return
            self._closed = True
            entries = list(self._entries.values())
            self._entries.clear()
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

    def delete(self, file_path: str) -> WriteResult:
        """Delete a file from the current session sandbox."""
        return self._invoke("delete", file_path)

    async def adelete(self, file_path: str) -> WriteResult:
        """Async version of :meth:`delete`."""
        return await asyncio.to_thread(self.delete, file_path)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        return self._invoke("grep", pattern, path, glob)

    def glob(self, pattern: str, path: str | None = "/") -> GlobResult:
        return self._invoke("glob", pattern, path)

    def upload_files(
        self, files: dict[str, bytes] | Iterable[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        return self._invoke("upload_files", files)

    def download_files(self, paths: Iterable[str]) -> list[FileDownloadResponse]:
        return self._invoke("download_files", paths)
