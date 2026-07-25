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

from __future__ import annotations

import base64
import logging
import posixpath
import shlex
import threading
import warnings
import weakref
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any, cast

from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import (
    _get_backend_read_file_type,
    check_empty_content,
)
from k8s_agent_sandbox.exceptions import SandboxNotFoundError

from langchain_google_agent_sandbox import _compat
from langchain_google_agent_sandbox._compat import SESSION_LABEL_KEY
from langchain_google_agent_sandbox._errors import is_timeout_exception
from langchain_google_agent_sandbox._paths import (
    compile_glob,
    compile_grep_include_glob,
    reject_control_chars,
)

logger = logging.getLogger(__name__)


class AgentSandboxBackend(SandboxBackendProtocol):
    """DeepAgents backend adapter for Kubernetes agent-sandbox runtimes.

    The backend exposes DeepAgents file and command operations while keeping
    user-facing paths virtualized under ``root_dir``. The default root is
    ``/workspace`` because DeepAgents agents commonly expect a workspace-like
    writable directory instead of an application installation directory.

    The adapter can wrap an existing sandbox handle or manage a sandbox created
    from a template. Managed sandboxes are created only when the backend is
    entered, which keeps import and construction free of Kubernetes calls.
    """

    SESSION_LABEL_KEY = SESSION_LABEL_KEY

    def __init__(
        self,
        sandbox: Any | None,
        root_dir: str = "/workspace",
        manage_lifecycle: bool = False,
        allow_absolute_paths: bool = False,
        sdk_client: Any | None = None,
        runtime_root: str = "/workspace",
        _warm_pool: str | None = None,
        _namespace: str = "default",
        _sandbox_ready_timeout: int = 180,
        _labels: dict[str, str] | None = None,
        _shutdown_after_seconds: int | None = None,
        _session_id: str | None = None,
        _default_timeout_seconds: int | None = None,
    ) -> None:
        if not root_dir.startswith("/"):
            msg = f"root_dir must be an absolute path, got: {root_dir}"
            raise ValueError(msg)
        if not runtime_root.startswith("/"):
            msg = f"runtime_root must be an absolute path, got: {runtime_root}"
            raise ValueError(msg)
        if _session_id is not None:
            _compat.validate_label_value(_session_id)
        self._sandbox = sandbox
        self._root_dir = posixpath.normpath(root_dir)
        self._runtime_root = posixpath.normpath(runtime_root)
        self._manage_lifecycle = manage_lifecycle
        self._allow_absolute_paths = allow_absolute_paths
        self._sdk_client = sdk_client
        self._warm_pool = _warm_pool
        self._namespace = _namespace
        self._sandbox_ready_timeout = _sandbox_ready_timeout
        self._labels = _labels
        self._shutdown_after_seconds = _shutdown_after_seconds
        self._session_id = _session_id
        self._default_timeout_seconds = _default_timeout_seconds
        self._reattached = False
        self._draining = False
        self._inflight = 0
        self._inflight_cv = threading.Condition(threading.Lock())
        self._finalizer: weakref.finalize | None = None

    @classmethod
    def from_existing(
        cls,
        sandbox: Any,
        root_dir: str = "/workspace",
        allow_absolute_paths: bool = False,
        runtime_root: str = "/workspace",
        default_timeout_seconds: int | None = 120,
    ) -> AgentSandboxBackend:
        """Wrap an already-connected sandbox without owning its lifecycle.

        Use this when another part of the application is responsible for
        creating and deleting the sandbox. ``__exit__`` will not delete a
        sandbox created through this constructor.

        Args:
            sandbox: Connected ``k8s-agent-sandbox`` sandbox handle.
            root_dir: Virtual root exposed to DeepAgents file operations.
            allow_absolute_paths: Allow writes outside ``root_dir`` through a
                shell fallback. This is dangerous and should be reserved for
                trusted workflows.
            runtime_root: Root understood by the SDK filesystem API.
            default_timeout_seconds: Default command timeout.

        Returns:
            An unmanaged backend ready for immediate use.
        """
        return cls(
            sandbox=sandbox,
            root_dir=root_dir,
            manage_lifecycle=False,
            allow_absolute_paths=allow_absolute_paths,
            runtime_root=runtime_root,
            _default_timeout_seconds=default_timeout_seconds,
        )

    @classmethod
    def from_warm_pool(
        cls,
        client: Any,
        warm_pool: str,
        namespace: str = "default",
        root_dir: str = "/workspace",
        allow_absolute_paths: bool = False,
        sandbox_ready_timeout: int = 180,
        labels: dict[str, str] | None = None,
        shutdown_after_seconds: int | None = None,
        session_id: str | None = None,
        default_timeout_seconds: int | None = 120,
    ) -> AgentSandboxBackend:
        """Create a lifecycle-managed backend from a SandboxWarmPool.

        The sandbox is created on ``__enter__`` and deleted on ``__exit__``.
        When ``session_id`` is supplied, ``__enter__`` first searches for an
        existing claim labeled with that session and reattaches if exactly one
        match exists. Multiple matches are refused because silently picking one
        could attach the agent to the wrong persistent filesystem.

        Args:
            client: Configured released ``SandboxClient``.
            warm_pool: SandboxWarmPool name.
            namespace: Kubernetes namespace containing the claim.
            root_dir: Virtual root exposed to DeepAgents.
            allow_absolute_paths: Allow writes outside ``root_dir``.
            sandbox_ready_timeout: Readiness wait in seconds.
            labels: Additional SandboxClaim labels.
            shutdown_after_seconds: Optional claim TTL when the SDK supports it.
            session_id: Stable label value used for reattach.
            default_timeout_seconds: Default command timeout for execute calls.

        Returns:
            A backend that must be entered before use.
        """
        return cls(
            sandbox=None,
            root_dir=root_dir,
            manage_lifecycle=True,
            allow_absolute_paths=allow_absolute_paths,
            sdk_client=client,
            _warm_pool=warm_pool,
            _namespace=namespace,
            _sandbox_ready_timeout=sandbox_ready_timeout,
            _labels=labels,
            _shutdown_after_seconds=shutdown_after_seconds,
            _session_id=session_id,
            _default_timeout_seconds=default_timeout_seconds,
        )

    @classmethod
    def from_template(
        cls,
        client: Any,
        template_name: str,
        namespace: str = "default",
        root_dir: str = "/workspace",
        allow_absolute_paths: bool = False,
        sandbox_ready_timeout: int = 180,
        labels: dict[str, str] | None = None,
        shutdown_after_seconds: int | None = None,
        session_id: str | None = None,
        default_timeout_seconds: int | None = 120,
    ) -> AgentSandboxBackend:
        """Deprecated alias for :meth:`from_warm_pool`."""
        warnings.warn(
            "from_template() is deprecated; use from_warm_pool()",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_warm_pool(
            client,
            template_name,
            namespace=namespace,
            root_dir=root_dir,
            allow_absolute_paths=allow_absolute_paths,
            sandbox_ready_timeout=sandbox_ready_timeout,
            labels=labels,
            shutdown_after_seconds=shutdown_after_seconds,
            session_id=session_id,
            default_timeout_seconds=default_timeout_seconds,
        )

    def __enter__(self) -> AgentSandboxBackend:
        if not self._manage_lifecycle:
            return self
        if self._sandbox is not None:
            return self
        if self._sdk_client is None:
            msg = "Cannot manage lifecycle without an sdk_client"
            raise RuntimeError(msg)
        self._reattached = False
        self._draining = False
        if self._try_reattach():
            return self

        labels = dict(self._labels) if self._labels else {}
        if self._session_id is not None:
            labels[self.SESSION_LABEL_KEY] = self._session_id
        self._sandbox = _compat.create_sandbox(
            self._sdk_client,
            warm_pool=cast("str", self._warm_pool),
            namespace=self._namespace,
            sandbox_ready_timeout=self._sandbox_ready_timeout,
            labels=labels or None,
            shutdown_after_seconds=self._shutdown_after_seconds,
        )
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        if not self._manage_lifecycle or self._sandbox is None:
            return
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer.detach()
        self._finalizer = None
        with self._inflight_cv:
            self._draining = True
            while self._inflight > 0:
                self._inflight_cv.wait()
        if self._reattached:
            close_connection = getattr(self._sandbox, "close_connection", None)
            if callable(close_connection):
                close_connection()
            self._sandbox = None
            return
        claim = getattr(self._sandbox, "claim_name", None)
        namespace = getattr(self._sandbox, "namespace", None) or self._namespace
        cleanup_error: BaseException | None = None
        try:
            if claim is not None and self._sdk_client is not None:
                _compat.delete_sandbox(
                    self._sdk_client, claim_name=claim, namespace=namespace
                )
        except Exception as error:
            cleanup_error = error
            logger.error(
                "Failed to delete sandbox (claim=%s, namespace=%s): %s",
                claim,
                namespace,
                error,
            )
        finally:
            self._sandbox = None
        if cleanup_error is None:
            return
        if exc_type is None:
            raise cleanup_error
        if exc is not None:
            raise BaseExceptionGroup(
                "sandbox cleanup failed during exception unwind",
                [exc, cleanup_error],
            ) from None
        warnings.warn(
            f"sandbox cleanup failed during exception unwind: "
            f"{type(cleanup_error).__name__}: {cleanup_error}",
            ResourceWarning,
            stacklevel=2,
        )

    async def __aenter__(self) -> AgentSandboxBackend:
        return self.__enter__()

    async def __aexit__(
        self, exc_type: Any, exc: BaseException | None, tb: Any
    ) -> None:
        self.__exit__(exc_type, exc, tb)

    def _register_finalizer(self) -> None:
        """Register fallback cleanup for a factory-managed ephemeral Claim."""
        if self._reattached or self._sandbox is None or self._sdk_client is None:
            return
        from langchain_google_agent_sandbox._lifecycle import factory_atexit_cleanup

        self._finalizer = weakref.finalize(
            self,
            factory_atexit_cleanup,
            self._sdk_client,
            self._sandbox,
        )

    def _try_reattach(self) -> bool:
        if self._sdk_client is None or self._session_id is None:
            return False
        selector = f"{self.SESSION_LABEL_KEY}={self._session_id}"
        claims = _compat.list_sandbox_claims(
            self._sdk_client,
            namespace=self._namespace,
            label_selector=selector,
        )
        if not claims:
            return False
        if len(claims) > 1:
            msg = (
                f"Refusing to reattach: {len(claims)} claims match "
                f"session_id={self._session_id!r} in namespace {self._namespace!r}"
            )
            raise RuntimeError(msg)
        self._sandbox = _compat.get_sandbox(
            self._sdk_client,
            claim_name=claims[0],
            namespace=self._namespace,
            warm_pool=self._warm_pool,
        )
        self._reattached = True
        return True

    def close(self) -> None:
        """Close only the local connector without deleting the Claim."""
        with self._inflight_cv:
            self._draining = True
            while self._inflight > 0:
                self._inflight_cv.wait()
        sandbox = self._sandbox
        self._sandbox = None
        if sandbox is None:
            return
        close_connection = getattr(sandbox, "close_connection", None)
        if callable(close_connection):
            close_connection()

    @contextmanager
    def _track_op(self) -> Iterator[None]:
        with self._inflight_cv:
            if self._draining:
                msg = "Backend is shutting down"
                raise RuntimeError(msg)
            self._inflight += 1
        try:
            yield
        finally:
            with self._inflight_cv:
                self._inflight -= 1
                if self._inflight == 0:
                    self._inflight_cv.notify_all()

    def _assert_sandbox(self) -> Any:
        if self._sandbox is None:
            msg = (
                "Sandbox is not initialized. Use 'with backend:' "
                "or call '__enter__()' before using it."
            )
            raise RuntimeError(msg)
        return self._sandbox

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a shell command with ``root_dir`` as the working directory."""
        sandbox = self._assert_sandbox()
        with self._track_op():
            wrapped = (
                f"sh -c {shlex.quote(f'cd {shlex.quote(self._root_dir)} && {command}')}"
            )
            effective_timeout = (
                timeout if timeout is not None else self._default_timeout_seconds
            )
            try:
                if effective_timeout is None:
                    result = sandbox.commands.run(wrapped)
                else:
                    result = sandbox.commands.run(wrapped, timeout=effective_timeout)
            except SandboxNotFoundError:
                raise
            except Exception as error:
                if is_timeout_exception(error):
                    return ExecuteResponse(
                        output=f"Timed out: {error}",
                        exit_code=-2,
                        truncated=False,
                    )
                return ExecuteResponse(
                    output=f"Error: {error}",
                    exit_code=-1,
                    truncated=False,
                )
        combined = result.stdout
        if result.stderr:
            combined = f"{combined}\n{result.stderr}" if combined else result.stderr
        return ExecuteResponse(
            output=combined,
            exit_code=result.exit_code,
            truncated=False,
        )

    def ls(self, path: str) -> LsResult:
        """List directory entries with stable ordering and metadata where present."""
        with self._track_op():
            try:
                internal_path = self._to_internal(path)
                entries_raw = self._list_entries_at(internal_path)
            except SandboxNotFoundError:
                raise
            except Exception as error:
                return LsResult(entries=[], error=f"Cannot list '{path}': {error}")
        entries: list[FileInfo] = []
        public_dir = self._normalize_public_dir(path)
        for entry in entries_raw:
            name = getattr(entry, "name", None)
            if name in (None, ".", ".."):
                continue
            info = FileInfo(
                path=posixpath.join(public_dir, str(name)),
                is_dir=getattr(entry, "type", None) == "directory",
            )
            size = getattr(entry, "size", None)
            if size is not None:
                info["size"] = int(size)
            mod_time = getattr(entry, "mod_time", None)
            if mod_time is not None:
                info["modified_at"] = datetime.fromtimestamp(
                    float(mod_time), tz=UTC
                ).isoformat()
            entries.append(info)
        entries.sort(key=lambda item: item["path"])
        return LsResult(entries=entries)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read raw UTF-8 content from a file, optionally by line window."""
        if offset < 0:
            return ReadResult(error=f"Line offset must be non-negative, got {offset}")
        if limit <= 0:
            return ReadResult(error=f"Line limit must be positive, got {limit}")
        self._assert_sandbox()
        with self._track_op():
            try:
                internal_path = self._to_internal(file_path)
                content = self._read_bytes_at(internal_path)
            except SandboxNotFoundError:
                raise
            except Exception as error:
                return ReadResult(error=f"Failed to read '{file_path}': {error}")
        if _get_backend_read_file_type(file_path) != "text":
            encoded = base64.b64encode(content).decode("ascii")
            return ReadResult(file_data=FileData(content=encoded, encoding="base64"))
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError:
            encoded = base64.b64encode(content).decode("ascii")
            return ReadResult(file_data=FileData(content=encoded, encoding="base64"))
        empty_message = check_empty_content(decoded)
        if empty_message is not None:
            return ReadResult(
                file_data=FileData(content=empty_message, encoding="utf-8")
            )
        lines = decoded.splitlines(keepends=True)
        start = offset
        if start >= len(lines):
            return ReadResult(
                error=f"Line offset {offset} exceeds file length ({len(lines)} lines)"
            )
        end = min(len(lines), start + limit)
        selected = "".join(lines[start:end])
        return ReadResult(
            file_data=FileData(content=selected, encoding="utf-8"),
            total_lines=len(lines),
            start_line=start + 1,
            end_line=end,
            next_offset=end if end < len(lines) else None,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        """Create or overwrite a UTF-8 file."""
        self._assert_sandbox()
        with self._track_op():
            try:
                internal_path = self._resolve_write_path(file_path)
            except ValueError as error:
                return WriteResult(
                    error=f"Error: Invalid path '{file_path}': {error}",
                    path=file_path,
                )
            try:
                self._ensure_parent_dir(internal_path)
                self._write_bytes_at(internal_path, content.encode("utf-8"))
            except SandboxNotFoundError:
                raise
            except Exception as error:
                return WriteResult(
                    error=f"Error writing '{file_path}': {error}",
                    path=file_path,
                )
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace a string in a UTF-8 file."""
        self._assert_sandbox()
        with self._track_op():
            try:
                internal_path = self._resolve_write_path(file_path)
                raw = self._read_bytes_at(internal_path)
                content = raw.decode("utf-8")
            except SandboxNotFoundError:
                raise
            except UnicodeDecodeError as error:
                return EditResult(
                    error=f"Error: Cannot edit '{file_path}': not valid UTF-8 ({error})",
                    path=file_path,
                    occurrences=0,
                )
            except Exception as error:
                return EditResult(
                    error=f"Error reading '{file_path}': {error}",
                    path=file_path,
                    occurrences=0,
                )
            occurrences = content.count(old_string)
            if occurrences == 0:
                return EditResult(
                    error=f"Error: String not found in file: '{old_string}'",
                    path=file_path,
                    occurrences=0,
                )
            if not replace_all and occurrences > 1:
                return EditResult(
                    error=(
                        f"Error: String '{old_string}' appears multiple times. "
                        "Use replace_all=True to replace all occurrences."
                    ),
                    path=file_path,
                    occurrences=occurrences,
                )
            updated = (
                content.replace(old_string, new_string)
                if replace_all
                else content.replace(old_string, new_string, 1)
            )
            try:
                self._write_bytes_at(internal_path, updated.encode("utf-8"))
            except SandboxNotFoundError:
                raise
            except Exception as error:
                return EditResult(
                    error=f"Error writing '{file_path}': {error}",
                    path=file_path,
                    occurrences=0,
                )
        return EditResult(
            path=file_path,
            occurrences=occurrences if replace_all else 1,
        )

    def delete(self, file_path: str) -> DeleteResult:
        """Delete a file, symlink, or directory tree."""
        self._assert_sandbox()
        with self._track_op():
            try:
                internal_path = self._resolve_write_path(file_path)
            except ValueError as error:
                return DeleteResult(
                    error=f"Error: Invalid path '{file_path}': {error}",
                )
            state = self._file_state(internal_path)
            if state == "missing":
                return DeleteResult(
                    error=f"Error: '{file_path}' not found",
                )
            if state == "error":
                return DeleteResult(
                    error=f"Cannot delete '{file_path}'",
                )
            sandbox = self._assert_sandbox()
            result = sandbox.commands.run(
                shlex.join(["rm", "-rf", "--", internal_path])
            )
            if result.exit_code != 0:
                detail = result.stderr.strip() or f"exit code {result.exit_code}"
                return DeleteResult(
                    error=f"Error deleting '{file_path}': {detail}",
                )
        return DeleteResult(path=file_path)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search literal text in files under ``path``."""
        sandbox = self._assert_sandbox()
        base_path = path or "/"
        try:
            include_matcher = compile_grep_include_glob(glob) if glob else None
        except Exception as error:
            return GrepResult(
                matches=[],
                error=f"invalid grep glob pattern {glob!r}: {error}",
            )
        with self._track_op():
            try:
                internal_path = self._to_internal(base_path)
            except ValueError as error:
                return GrepResult(
                    matches=[], error=f"Invalid path '{base_path}': {error}"
                )
            parts = ["grep", "-rHnFZ"]
            parts.extend(["-e", shlex.quote(pattern), shlex.quote(internal_path)])
            run_kwargs: dict[str, Any] = {}
            if self._default_timeout_seconds is not None:
                run_kwargs["timeout"] = self._default_timeout_seconds
            try:
                result = sandbox.commands.run(" ".join(parts), **run_kwargs)
            except SandboxNotFoundError:
                raise
            except Exception as error:
                return GrepResult(
                    matches=[],
                    error=f"grep failed in '{base_path}': {error}",
                )
        if result.exit_code >= 2:
            detail = (
                result.stderr.strip()
                or result.stdout.strip()
                or (f"exit code {result.exit_code}")
            )
            return GrepResult(
                matches=[],
                error=f"grep failed in '{base_path}': {detail}",
            )
        if not result.stdout.strip():
            return GrepResult(matches=[])
        matches: list[GrepMatch] = []
        for line in result.stdout.splitlines():
            nul_pos = line.find("\0")
            if nul_pos < 0:
                split = line.split(":", 2)
                if len(split) != 3:
                    continue
                raw_path, line_no, text = split
            else:
                raw_path = line[:nul_pos]
                split = line[nul_pos + 1 :].split(":", 1)
                if len(split) != 2:
                    continue
                line_no, text = split
            try:
                line_int = int(line_no)
            except ValueError:
                continue
            relative_path = (
                posixpath.basename(raw_path)
                if raw_path == internal_path
                else posixpath.relpath(raw_path, internal_path)
            )
            if include_matcher is not None and not include_matcher(relative_path):
                continue
            matches.append(
                GrepMatch(path=self._to_public(raw_path), line=line_int, text=text)
            )
        if max_count is not None and len(matches) > max_count:
            return GrepResult(matches=matches[:max_count], truncated=True)
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching ``pattern`` under ``path``."""
        sandbox = self._assert_sandbox()
        base_path = path or "/"
        if ".." in PurePosixPath(pattern).parts:
            return GlobResult(
                matches=[],
                error="Path traversal is not allowed in glob patterns",
            )
        try:
            matcher = compile_glob(pattern.lstrip("/"))
        except Exception as error:
            return GlobResult(
                matches=[],
                error=f"invalid glob pattern {pattern!r}: {error}",
            )
        with self._track_op():
            try:
                internal_path = self._to_internal(base_path)
            except ValueError as error:
                return GlobResult(
                    matches=[], error=f"Invalid path '{base_path}': {error}"
                )
            command = (
                f"find -P {shlex.quote(internal_path)} -mindepth 1"
                f" -type f -printf 'f\\t%s\\t%T@\\t%p\\0'"
            )
            run_kwargs: dict[str, Any] = {}
            if self._default_timeout_seconds is not None:
                run_kwargs["timeout"] = self._default_timeout_seconds
            try:
                result = sandbox.commands.run(command, **run_kwargs)
            except SandboxNotFoundError:
                raise
            except Exception as error:
                return GlobResult(
                    matches=[],
                    error=f"glob failed in '{base_path}': {error}",
                )
        if result.exit_code != 0 and not result.stdout:
            detail = result.stderr.strip() or f"exit code {result.exit_code}"
            return GlobResult(
                matches=[], error=f"glob failed in '{base_path}': {detail}"
            )
        entries: list[FileInfo] = []
        for record in result.stdout.split("\x00"):
            if not record:
                continue
            split = record.split("\t", 3)
            if len(split) != 4:
                continue
            type_char, size_str, mod_str, raw = split
            if type_char != "f":
                continue
            rel_path = posixpath.relpath(raw, internal_path)
            if not matcher(rel_path):
                continue
            info = FileInfo(path=self._to_public(raw), is_dir=False)
            try:
                info["size"] = int(size_str)
            except ValueError:
                pass
            try:
                info["modified_at"] = datetime.fromtimestamp(
                    float(mod_str), tz=UTC
                ).isoformat()
            except (OSError, ValueError):
                pass
            entries.append(info)
        entries.sort(key=lambda item: item["path"])
        return GlobResult(matches=entries)

    def upload_files(
        self, files: dict[str, bytes] | Iterable[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload files and report per-file success or failure."""
        self._assert_sandbox()
        pairs = list(files.items()) if isinstance(files, dict) else list(files)
        responses: list[FileUploadResponse] = []
        with self._track_op():
            for path, payload in pairs:
                try:
                    internal_path = self._resolve_write_path(path)
                except ValueError:
                    responses.append(
                        FileUploadResponse(path=path, error="invalid_path")
                    )
                    continue
                state = self._file_state(internal_path)
                if state in {"error", "dir", "denied"}:
                    error = {
                        "error": "upload_failed",
                        "dir": "is_directory",
                        "denied": "permission_denied",
                    }[state]
                    responses.append(FileUploadResponse(path=path, error=error))
                    continue
                parent_state = self._dir_state(posixpath.dirname(internal_path))
                if parent_state in {"error", "not_dir", "denied"}:
                    error = {
                        "error": "upload_failed",
                        "not_dir": "invalid_path",
                        "denied": "permission_denied",
                    }[parent_state]
                    responses.append(FileUploadResponse(path=path, error=error))
                    continue
                if parent_state == "missing":
                    try:
                        self._ensure_parent_dir(internal_path)
                    except Exception:
                        responses.append(
                            FileUploadResponse(path=path, error="upload_failed")
                        )
                        continue
                try:
                    self._write_bytes_at(internal_path, payload)
                except Exception:
                    responses.append(
                        FileUploadResponse(path=path, error="upload_failed")
                    )
                    continue
                responses.append(FileUploadResponse(path=path, error=None))
        return responses

    def download_files(self, paths: Iterable[str]) -> list[FileDownloadResponse]:
        """Download files and report per-file success or failure."""
        self._assert_sandbox()
        responses: list[FileDownloadResponse] = []
        with self._track_op():
            for path in paths:
                try:
                    internal_path = self._to_internal(path)
                except ValueError:
                    responses.append(
                        FileDownloadResponse(
                            path=path, content=None, error="invalid_path"
                        )
                    )
                    continue
                state = self._file_state(internal_path)
                if state in {"error", "missing", "dir", "denied"}:
                    error = {
                        "error": "download_failed",
                        "missing": "file_not_found",
                        "dir": "is_directory",
                        "denied": "permission_denied",
                    }[state]
                    responses.append(
                        FileDownloadResponse(path=path, content=None, error=error)
                    )
                    continue
                try:
                    content = self._read_bytes_at(internal_path)
                except Exception:
                    responses.append(
                        FileDownloadResponse(
                            path=path, content=None, error="download_failed"
                        )
                    )
                    continue
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
        return responses

    @staticmethod
    def delete_all(
        client: Any,
        namespace: str = "default",
        best_effort: bool = True,
        label_selector: str | None = None,
    ) -> int:
        """Delete sandbox claims in ``namespace``.

        Args:
            client: Configured released ``SandboxClient``.
            namespace: Namespace to clean up.
            best_effort: Continue after individual delete failures.
            label_selector: Optional Kubernetes label selector. Without it,
                every claim in the namespace is deleted.

        Returns:
            Number of successfully deleted claims.
        """
        claims = _compat.list_sandbox_claims(
            client,
            namespace=namespace,
            label_selector=label_selector,
        )
        deleted = 0
        for claim in claims:
            try:
                _compat.delete_sandbox(client, claim_name=claim, namespace=namespace)
                deleted += 1
            except Exception:
                if not best_effort:
                    raise
                logger.warning(
                    "delete_all: failed to delete %s in namespace %s",
                    claim,
                    namespace,
                    exc_info=True,
                )
        return deleted

    @property
    def id(self) -> str:
        """Return ``namespace/claim`` when available, else a stable fallback."""
        if self._sandbox is not None:
            namespace = getattr(self._sandbox, "namespace", None) or "default"
            claim = getattr(self._sandbox, "claim_name", None)
            if claim:
                return f"{namespace}/{claim}"
        return "agent-sandbox"

    def _ensure_parent_dir(self, internal_path: str) -> None:
        sandbox = self._assert_sandbox()
        parent = posixpath.dirname(internal_path)
        result = sandbox.commands.run(shlex.join(["mkdir", "-p", parent]))
        if result.exit_code != 0:
            detail = result.stderr.strip() or f"exit code {result.exit_code}"
            msg = f"Cannot create parent directory '{parent}': {detail}"
            raise RuntimeError(msg)

    def _resolve_write_path(self, path: str) -> str:
        candidate = path.strip()
        if not candidate:
            msg = "empty path"
            raise ValueError(msg)
        reject_control_chars(candidate)
        normalized = posixpath.normpath(candidate)
        if (
            self._allow_absolute_paths
            and normalized.startswith("/")
            and normalized != self._root_dir
            and not normalized.startswith(self._root_dir + "/")
        ):
            return normalized
        return self._to_internal(candidate)

    def _to_internal(self, path: str) -> str:
        stripped = path.strip() or "/"
        reject_control_chars(stripped)
        normalized = posixpath.normpath(stripped)
        if (
            self._allow_absolute_paths
            and normalized.startswith("/")
            and normalized != "/"
        ):
            if normalized == self._runtime_root or normalized.startswith(
                self._runtime_root + "/"
            ):
                internal_path = normalized
            else:
                internal_path = posixpath.normpath(
                    posixpath.join(self._runtime_root, normalized.lstrip("/"))
                )
            self._to_runtime_relative(internal_path)
            return internal_path
        if normalized == self._root_dir or normalized.startswith(self._root_dir + "/"):
            normalized = normalized[len(self._root_dir) :]
        normalized = normalized.lstrip("/")
        internal_path = posixpath.normpath(posixpath.join(self._root_dir, normalized))
        rel = posixpath.relpath(internal_path, self._root_dir)
        if rel == ".." or rel.startswith("../"):
            msg = f"Path '{path}' escapes root_dir '{self._root_dir}'"
            raise ValueError(msg)
        return internal_path

    def _to_runtime_relative(self, internal_path: str) -> str:
        rel = posixpath.relpath(internal_path, self._runtime_root)
        if rel == ".." or rel.startswith("../"):
            msg = (
                f"Internal path '{internal_path}' is outside runtime_root "
                f"'{self._runtime_root}'"
            )
            raise ValueError(msg)
        return "." if rel == "." else rel

    def _to_public(self, internal_path: str) -> str:
        rel = posixpath.relpath(internal_path, self._root_dir)
        if self._allow_absolute_paths and (rel == ".." or rel.startswith("../")):
            self._to_runtime_relative(internal_path)
            return internal_path
        return "/" if rel == "." else "/" + rel

    def _normalize_public_dir(self, path: str) -> str:
        if not path or path == "/":
            return "/"
        if path == self._root_dir or path.startswith(self._root_dir + "/"):
            path = path[len(self._root_dir) :]
        return "/" + path.strip("/")

    def _list_entries_at(self, internal_path: str) -> list[Any]:
        sandbox = self._assert_sandbox()
        try:
            runtime_rel = self._to_runtime_relative(internal_path)
        except ValueError:
            runtime_rel = None
        if runtime_rel is not None:
            return list(sandbox.files.list(runtime_rel))

        command = (
            f"find -L {shlex.quote(internal_path)} -mindepth 1 -maxdepth 1"
            f" \\( -type d -printf 'd\\t%s\\t%T@\\t%f\\0' \\)"
            f" -o \\( -printf 'f\\t%s\\t%T@\\t%f\\0' \\)"
        )
        run_kwargs: dict[str, Any] = {}
        if self._default_timeout_seconds is not None:
            run_kwargs["timeout"] = self._default_timeout_seconds
        result = sandbox.commands.run(command, **run_kwargs)
        if result.exit_code != 0 and not result.stdout:
            detail = result.stderr.strip() or f"exit code {result.exit_code}"
            msg = f"list '{internal_path}' failed: {detail}"
            raise RuntimeError(msg)

        entries: list[Any] = []
        for record in result.stdout.split("\x00"):
            if not record:
                continue
            split = record.split("\t", 3)
            if len(split) != 4:
                continue
            type_char, size_str, mod_str, name = split
            if type_char not in {"d", "f"}:
                continue
            metadata: dict[str, Any] = {
                "name": name,
                "type": "directory" if type_char == "d" else "file",
            }
            try:
                metadata["size"] = int(size_str)
            except ValueError:
                pass
            try:
                metadata["mod_time"] = float(mod_str)
            except ValueError:
                pass
            entries.append(SimpleNamespace(**metadata))
        return entries

    def _write_bytes_at(self, internal_path: str, payload: bytes) -> None:
        sandbox = self._assert_sandbox()
        try:
            runtime_rel = self._to_runtime_relative(internal_path)
        except ValueError:
            runtime_rel = None
        if runtime_rel is not None:
            sandbox.files.write(runtime_rel, payload)
            return
        encoded = base64.b64encode(payload).decode("ascii")
        pipeline = (
            f"printf %s {shlex.quote(encoded)} "
            f"| base64 -d > {shlex.quote(internal_path)}"
        )
        result = sandbox.commands.run(f"sh -c {shlex.quote(pipeline)}")
        if result.exit_code != 0:
            detail = result.stderr.strip() or f"exit code {result.exit_code}"
            msg = f"Write to '{internal_path}' failed: {detail}"
            raise RuntimeError(msg)

    def _read_bytes_at(self, internal_path: str) -> bytes:
        sandbox = self._assert_sandbox()
        try:
            runtime_rel = self._to_runtime_relative(internal_path)
        except ValueError:
            runtime_rel = None
        if runtime_rel is not None:
            return _compat.normalize_read_bytes(sandbox.files.read(runtime_rel))
        command = f"sh -c {shlex.quote(f'base64 < {shlex.quote(internal_path)}')}"
        result = sandbox.commands.run(command)
        if result.exit_code != 0:
            detail = result.stderr.strip() or f"exit code {result.exit_code}"
            msg = f"Read from '{internal_path}' failed: {detail}"
            raise RuntimeError(msg)
        return base64.b64decode(result.stdout)

    _FILE_STATE_VALID = frozenset({"missing", "dir", "file", "denied"})
    _DIR_STATE_VALID = frozenset({"missing", "writable", "denied", "not_dir"})

    def _probe_state(self, check_script: str, valid: frozenset[str], label: str) -> str:
        sandbox = self._assert_sandbox()
        result = sandbox.commands.run(f"sh -c {shlex.quote(check_script)}")
        output = result.stdout.strip()
        if not output and result.exit_code != 0:
            logger.warning(
                "_%s command failed: exit_code=%d, stderr=%s",
                label,
                result.exit_code,
                result.stderr,
            )
            return "error"
        state = output or "missing"
        return state if state in valid else "error"

    def _file_state(self, internal_path: str) -> str:
        check = (
            f"if [ ! -e {shlex.quote(internal_path)} ] && "
            f"[ ! -L {shlex.quote(internal_path)} ]; then echo missing; exit 0; fi; "
            f"if [ -d {shlex.quote(internal_path)} ]; then echo dir; exit 0; fi; "
            f"if [ -r {shlex.quote(internal_path)} ]; then echo file; else echo denied; fi"
        )
        return self._probe_state(check, self._FILE_STATE_VALID, "file_state")

    def _dir_state(self, internal_path: str) -> str:
        check = (
            f"if [ ! -e {shlex.quote(internal_path)} ]; then echo missing; exit 0; fi; "
            f"if [ -d {shlex.quote(internal_path)} ]; then "
            f"if [ -w {shlex.quote(internal_path)} ]; then echo writable; else echo denied; fi; "
            f"exit 0; fi; "
            f"echo not_dir"
        )
        return self._probe_state(check, self._DIR_STATE_VALID, "dir_state")
