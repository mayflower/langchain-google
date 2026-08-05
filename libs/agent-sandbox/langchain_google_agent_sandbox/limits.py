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

"""Configurable bounds on results returned to a DeepAgents agent.

Sandbox operations can produce unbounded output: a `grep` across a large
workspace, a command that writes megabytes to stdout, a bulk download. Every
such result is bounded here and reported through the protocol's own truncation
indicators, so an agent is never handed silently-dropped data.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

__all__ = ["SandboxResultLimits"]


@dataclass(frozen=True, slots=True)
class SandboxResultLimits:
    """Upper bounds applied to backend results.

    Defaults are deliberately generous: they exist to stop pathological cases
    from exhausting an agent's context, not to shape ordinary output. Lower
    them when a model's context budget is tight.

    Attributes:
        execute_output_bytes: Maximum UTF-8 byte length of combined command
            output. Truncation sets `ExecuteResponse.truncated`.
        read_lines: Maximum lines returned by a single read. A capped read
            reports the remainder through `ReadResult.next_offset`.
        grep_matches: Maximum grep matches. Truncation sets
            `GrepResult.truncated`.
        glob_matches: Maximum glob matches. Truncation sets
            `GlobResult.truncated`.
        upload_files: Maximum files accepted by one upload call. Files beyond
            the bound are reported individually with a `limit_exceeded` error
            rather than dropped.
        download_files: Maximum files served by one download call, reported
            the same way.
    """

    execute_output_bytes: int = 1_048_576
    read_lines: int = 10_000
    grep_matches: int = 1_000
    glob_matches: int = 1_000
    upload_files: int = 100
    download_files: int = 100

    def __post_init__(self) -> None:
        """Reject non-positive bounds, which would silently disable results."""
        for spec in fields(self):
            value = getattr(self, spec.name)
            # bool is an int subclass and would pass an isinstance check while
            # producing a nonsensical bound of 0 or 1.
            if type(value) is not int or value <= 0:
                msg = f"{spec.name} must be a positive integer, got {value!r}"
                raise ValueError(msg)


DEFAULT_LIMITS = SandboxResultLimits()
