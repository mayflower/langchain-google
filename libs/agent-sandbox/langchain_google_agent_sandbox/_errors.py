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

from k8s_agent_sandbox.exceptions import SandboxRequestError
from urllib3.exceptions import MaxRetryError, NewConnectionError


def is_connection_setup_error(exc: BaseException) -> bool:
    """Identify SDK failures before a connection could send the command."""
    if not isinstance(exc, SandboxRequestError) or exc.status_code is not None:
        return False
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, NewConnectionError):
            return True
        current = (
            current.reason
            if isinstance(current, MaxRetryError)
            else current.__cause__ or current.__context__
        )
    return False


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
