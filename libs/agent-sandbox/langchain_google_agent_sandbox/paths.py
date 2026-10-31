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
"""Mapping between public, internal and runtime-relative sandbox paths."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass

from langchain_google_agent_sandbox._paths import reject_control_chars

__all__ = ["SandboxPaths"]


@dataclass(frozen=True)
class SandboxPaths:
    """How a backend addresses files in its sandbox.

    Three forms of one path exist. The *public* path is what DeepAgents tools
    see, rooted at ``/``. The *internal* path is the absolute path inside the
    sandbox container, under ``root_dir``. The *runtime-relative* path is what
    the sandbox runtime's file API takes, relative to ``runtime_root``.

    Code that talks to the sandbox runtime directly uses this to address the
    same files the backend does.
    """

    root_dir: str
    runtime_root: str
    allow_absolute_paths: bool = False

    def to_internal(self, path: str) -> str:
        """Map a public path to its internal path, rejecting escapes."""
        stripped = path.strip() or "/"
        reject_control_chars(stripped)
        normalized = posixpath.normpath(stripped)
        if (
            self.allow_absolute_paths
            and normalized.startswith("/")
            and normalized != "/"
        ):
            if normalized == self.runtime_root or normalized.startswith(
                self.runtime_root + "/"
            ):
                internal_path = normalized
            else:
                internal_path = posixpath.normpath(
                    posixpath.join(self.runtime_root, normalized.lstrip("/"))
                )
            self.to_runtime_relative(internal_path)
            return internal_path
        if normalized == self.root_dir or normalized.startswith(self.root_dir + "/"):
            normalized = normalized[len(self.root_dir) :]
        normalized = normalized.lstrip("/")
        internal_path = posixpath.normpath(posixpath.join(self.root_dir, normalized))
        rel = posixpath.relpath(internal_path, self.root_dir)
        if rel == ".." or rel.startswith("../"):
            msg = f"Path '{path}' escapes root_dir '{self.root_dir}'"
            raise ValueError(msg)
        return internal_path

    def to_runtime_relative(self, internal_path: str) -> str:
        """Express an internal path relative to ``runtime_root``."""
        rel = posixpath.relpath(internal_path, self.runtime_root)
        if rel == ".." or rel.startswith("../"):
            msg = (
                f"Internal path '{internal_path}' is outside runtime_root "
                f"'{self.runtime_root}'"
            )
            raise ValueError(msg)
        return "." if rel == "." else rel

    def to_public(self, internal_path: str) -> str:
        """Map an internal path back to its public path."""
        rel = posixpath.relpath(internal_path, self.root_dir)
        if self.allow_absolute_paths and (rel == ".." or rel.startswith("../")):
            self.to_runtime_relative(internal_path)
            return internal_path
        return "/" if rel == "." else "/" + rel

    def write_target(self, path: str) -> str:
        """The internal path a write to the public ``path`` replaces or creates."""
        candidate = path.strip()
        if not candidate:
            msg = "empty path"
            raise ValueError(msg)
        reject_control_chars(candidate)
        normalized = posixpath.normpath(candidate)
        if (
            self.allow_absolute_paths
            and normalized.startswith("/")
            and normalized != self.root_dir
            and not normalized.startswith(self.root_dir + "/")
        ):
            return normalized
        return self.to_internal(candidate)
