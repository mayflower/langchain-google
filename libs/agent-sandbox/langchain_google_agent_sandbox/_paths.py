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
import re
from collections.abc import Callable


def reject_control_chars(path: str) -> None:
    """Reject NUL and ASCII control characters before shell/runtime use."""
    if any(ord(char) < 0x20 for char in path):
        msg = f"Path contains ASCII control characters: {path!r}"
        raise ValueError(msg)


def canonical_public_path(path: str) -> str:
    """Normalize a user-facing path without consulting the sandbox filesystem."""
    normalized = posixpath.normpath(path.strip() or "/")
    return "/" + normalized.lstrip("/")


def compile_glob(pattern: str) -> Callable[[str], bool]:
    """Compile a glob pattern with recursive ``**`` semantics."""
    if "/" not in pattern:
        segment_regex = _translate_glob_segment(pattern)
        compiled_basename = re.compile("^" + segment_regex + "$")
        return lambda path: (
            compiled_basename.fullmatch(posixpath.basename(path)) is not None
        )

    segments = pattern.split("/")
    collapsed: list[str] = []
    for segment in segments:
        if segment == "**" and collapsed and collapsed[-1] == "**":
            continue
        collapsed.append(segment)
    segments = collapsed

    if len(segments) == 1 and segments[0] == "**":
        return lambda path: True

    regex_parts: list[str] = []
    for index, segment in enumerate(segments):
        is_first = index == 0
        is_last = index == len(segments) - 1
        if segment == "**":
            if is_first:
                regex_parts.append("(?:[^/]+/)*")
            elif is_last:
                regex_parts.append("/.*")
            else:
                regex_parts.append("/(?:[^/]+/)*")
            continue
        if not is_first and segments[index - 1] != "**":
            regex_parts.append("/")
        regex_parts.append(_translate_glob_segment(segment))

    compiled = re.compile("^" + "".join(regex_parts) + "$")
    return lambda path: compiled.fullmatch(path) is not None


def _translate_glob_segment(segment: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(segment):
        char = segment[index]
        if char == "*":
            out.append("[^/]*")
            index += 1
        elif char == "?":
            out.append("[^/]")
            index += 1
        elif char == "[":
            end = segment.find("]", index + 1)
            if end == -1:
                out.append(re.escape(char))
                index += 1
            else:
                out.append(segment[index : end + 1])
                index = end + 1
        else:
            out.append(re.escape(char))
            index += 1
    return "".join(out)
