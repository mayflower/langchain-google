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

import logging
import posixpath
from collections.abc import Callable, Iterable
from typing import Any, cast

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

logger = logging.getLogger(__name__)


class SandboxPolicyWrapper(SandboxBackendProtocol):
    """Best-effort application guardrail around ``AgentSandboxBackend``.

    The wrapper can block writes/edits/uploads under configured path prefixes,
    block commands by substring, and invoke an audit callback. It is not a
    security boundary: real isolation must come from the sandbox runtime,
    container, Kubernetes, and node configuration.
    """

    def __init__(
        self,
        backend: SandboxBackendProtocol,
        deny_prefixes: list[str] | None = None,
        deny_commands: list[str] | None = None,
        audit_log: Callable[[str, str, dict[str, Any]], None] | None = None,
        *,
        strict_audit: bool = False,
    ) -> None:
        self._backend = backend
        self._deny_prefixes = [
            self._normalize_prefix(self._canonicalize_path(prefix))
            for prefix in (deny_prefixes or [])
        ]
        self._deny_commands = deny_commands or []
        self._audit_log = audit_log
        self._strict_audit = strict_audit

    def __enter__(self) -> SandboxPolicyWrapper:
        enter = getattr(self._backend, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        exit_backend = getattr(self._backend, "__exit__", None)
        if callable(exit_backend):
            exit_backend(exc_type, exc, tb)

    async def __aenter__(self) -> SandboxPolicyWrapper:
        return self.__enter__()

    async def __aexit__(
        self, exc_type: Any, exc: BaseException | None, tb: Any
    ) -> None:
        self.__exit__(exc_type, exc, tb)

    @staticmethod
    def _canonicalize_path(path: str) -> str:
        rooted = "/" + (path.strip() or "/").lstrip("/")
        return posixpath.normpath(rooted)

    @staticmethod
    def _normalize_prefix(path: str) -> str:
        canonical = SandboxPolicyWrapper._canonicalize_path(path)
        return canonical.rstrip("/") + "/" if canonical != "/" else "/"

    def _is_denied_path(self, path: str) -> bool:
        normalized = self._normalize_prefix(self._canonicalize_path(path))
        return any(normalized.startswith(prefix) for prefix in self._deny_prefixes)

    def _emit_audit(
        self, operation: str, target: str, metadata: dict[str, Any]
    ) -> str | None:
        if self._audit_log is None:
            return None
        try:
            self._audit_log(operation, target, metadata)
        except Exception as error:
            if self._strict_audit:
                logger.error(
                    "Audit log callback failed for %s on %s; refusing operation",
                    operation,
                    target,
                    exc_info=True,
                )
                return f"Audit log unavailable; refusing {operation}: {error}"
            logger.warning(
                "Audit log callback failed for %s on %s; operation will proceed",
                operation,
                target,
                exc_info=True,
            )
        return None

    def _is_denied_command(self, command: str) -> bool:
        return any(pattern in command for pattern in self._deny_commands)

    def ls(self, path: str) -> LsResult:
        return self._backend.ls(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self._backend.read(file_path, offset, limit)

    def grep(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> GrepResult:
        return self._backend.grep(pattern, path, glob)

    def glob(self, pattern: str, path: str | None = "/") -> GlobResult:
        return self._backend.glob(pattern, path)

    def download_files(self, paths: Iterable[str]) -> list[FileDownloadResponse]:
        return self._backend.download_files(list(paths))

    def write(self, file_path: str, content: str) -> WriteResult:
        if self._is_denied_path(file_path):
            return WriteResult(
                error=f"Policy denied: writes not allowed under '{file_path}'",
                path=file_path,
            )
        deny = self._emit_audit("write", file_path, {"size": len(content)})
        if deny is not None:
            return WriteResult(error=deny, path=file_path)
        return self._backend.write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        if self._is_denied_path(file_path):
            return EditResult(
                error=f"Policy denied: edits not allowed under '{file_path}'",
                path=file_path,
                occurrences=0,
            )
        deny = self._emit_audit("edit", file_path, {"replace_all": replace_all})
        if deny is not None:
            return EditResult(error=deny, path=file_path, occurrences=0)
        return self._backend.edit(file_path, old_string, new_string, replace_all)

    def delete(self, file_path: str) -> WriteResult:
        """Delete a file when path policy and audit permit it."""
        if self._is_denied_path(file_path):
            return WriteResult(
                error=f"Policy denied: deletes not allowed under '{file_path}'",
                path=file_path,
            )
        deny = self._emit_audit("delete", file_path, {})
        if deny is not None:
            return WriteResult(error=deny, path=file_path)
        delete_backend = getattr(self._backend, "delete", None)
        if not callable(delete_backend):
            return WriteResult(
                error="Policy backend does not support file deletion",
                path=file_path,
            )
        return cast("WriteResult", delete_backend(file_path))

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if self._is_denied_command(command):
            return ExecuteResponse(
                output="Policy denied: command blocked by policy",
                exit_code=1,
                truncated=False,
            )
        deny = self._emit_audit("execute", command, {})
        if deny is not None:
            return ExecuteResponse(output=deny, exit_code=1, truncated=False)
        return self._backend.execute(command, timeout=timeout)

    def upload_files(
        self, files: dict[str, bytes] | Iterable[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        pairs = list(files.items()) if isinstance(files, dict) else list(files)
        responses: list[FileUploadResponse | None] = [None] * len(pairs)
        allowed: list[tuple[int, str, bytes]] = []

        for index, (path, payload) in enumerate(pairs):
            if self._is_denied_path(path):
                responses[index] = FileUploadResponse(
                    path=path, error=cast("Any", "policy_denied")
                )
                continue
            deny = self._emit_audit("upload", path, {"size": len(payload)})
            if deny is not None:
                responses[index] = FileUploadResponse(
                    path=path, error=cast("Any", deny)
                )
                continue
            allowed.append((index, path, payload))

        if allowed:
            backend_responses = self._backend.upload_files(
                [(path, payload) for _, path, payload in allowed]
            )
            for (index, _, _), response in zip(
                allowed, backend_responses, strict=False
            ):
                responses[index] = response

        return cast("list[FileUploadResponse]", responses)

    @property
    def id(self) -> str:
        return self._backend.id
