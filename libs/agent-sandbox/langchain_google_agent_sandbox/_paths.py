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

import posixpath
from collections.abc import Callable
from functools import lru_cache
from pathlib import PurePosixPath

from wcmatch import glob as wcglob


def reject_control_chars(path: str) -> None:
    """Reject NUL and ASCII control characters before shell/runtime use."""
    if any(ord(char) < 0x20 for char in path):
        msg = f"Path contains ASCII control characters: {path!r}"
        raise ValueError(msg)


def canonical_public_path(path: str) -> str:
    """Normalize a user-facing path without consulting the sandbox filesystem."""
    normalized = posixpath.normpath(path.strip() or "/")
    return "/" + normalized.lstrip("/")


@lru_cache(maxsize=256)
def compile_glob(pattern: str) -> Callable[[str], bool]:
    """Compile DeepAgents recursive glob semantics, including brace expansion."""
    flags = wcglob.BRACE | wcglob.GLOBSTAR | wcglob.DOTMATCH
    compiled = wcglob.compile("**/" + pattern.lstrip("/"), flags=flags)
    return lambda path: bool(compiled.match(path))


@lru_cache(maxsize=256)
def compile_grep_include_glob(pattern: str) -> Callable[[str], bool]:
    """Compile DeepAgents grep include-glob semantics."""
    flags = wcglob.BRACE | wcglob.GLOBSTAR
    anchored = "/" in pattern
    compiled = wcglob.compile(pattern.lstrip("/"), flags=flags)
    if anchored:
        return lambda path: bool(compiled.match(path))
    return lambda path: bool(compiled.match(PurePosixPath(path).name))
