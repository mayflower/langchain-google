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

try:
    from requests.exceptions import Timeout as _RequestsTimeout
except ImportError:  # pragma: no cover - optional transport
    _RequestsTimeout = None  # type: ignore[assignment,misc]

try:
    from httpx import TimeoutException as _HttpxTimeout
except ImportError:  # pragma: no cover - optional transport
    _HttpxTimeout = None  # type: ignore[assignment,misc]


def is_timeout_exception(exc: BaseException) -> bool:
    """Return whether an SDK exception chain represents a timeout."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, TimeoutError):
            return True
        if _RequestsTimeout is not None and isinstance(cur, _RequestsTimeout):
            return True
        if _HttpxTimeout is not None and isinstance(cur, _HttpxTimeout):
            return True
        if "timeout" in type(cur).__name__.lower():
            return True
        cur = cur.__cause__ or cur.__context__
    return False
