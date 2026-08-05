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

"""Provider boundary between this protocol adapter and a product control plane.

This package is a DeepAgents protocol adapter, not a sandbox control plane. It
does not decide admission, derive Claim ownership, renew leases, or delete
durable sessions. Those are product concerns, supplied through
:class:`SandboxSessionProvider`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "SandboxLease",
    "SandboxSessionProvider",
]


@dataclass(frozen=True, slots=True)
class SandboxLease:
    """An authorized, already-provisioned or attached sandbox session.

    A lease is produced by the product's control plane and consumed by this
    adapter. The adapter treats it as opaque: it never parses ``key`` to derive
    Kubernetes identity, and never uses ``claim_name`` to address the API
    server.

    Attributes:
        key: Opaque, stable identifier for the leased session. Safe to place in
            process-local logs. The adapter uses it only for cache bookkeeping.
        sandbox: Connected ``k8s_agent_sandbox.sandbox.Sandbox`` handle.
        claim_name: Informational Claim name, for product-side correlation.
        namespace: Informational Kubernetes namespace.
        metadata: Free-form product annotations passed through untouched.
    """

    key: str
    sandbox: Any
    claim_name: str | None = None
    namespace: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject leases that cannot participate in cache bookkeeping."""
        if not isinstance(self.key, str) or not self.key:
            msg = "SandboxLease.key must be non-empty text"
            raise ValueError(msg)
        if self.sandbox is None:
            msg = "SandboxLease.sandbox must be a connected sandbox handle"
            raise ValueError(msg)


@runtime_checkable
class SandboxSessionProvider(Protocol):
    """Product-owned lifecycle boundary used by the protocol adapter.

    Implementations map tenant, user, and thread identity to durable sandbox
    infrastructure. In mAIstack this is backed by a dedicated sandbox
    control-plane service.

    There is deliberately no ``delete()`` method. Explicit durable deletion is a
    product operation, not a DeepAgents backend operation, so it has no route
    through this adapter.
    """

    def acquire(self, config: Mapping[str, Any]) -> SandboxLease:
        """Return an authorized, usable lease for the active run.

        May call a remote control plane, and may reject the request (for
        example on capacity or admission grounds) by raising. The adapter calls
        this only when it has no usable process-local lease cached.

        Args:
            config: The active LangGraph/DeepAgents config mapping.

        Returns:
            An authorized lease whose sandbox is ready for immediate use.
        """
        ...

    def touch(self, lease: SandboxLease) -> None:
        """Record best-effort session activity.

        The adapter throttles these calls to at most one per configured
        interval, so this is never invoked once per low-level file operation.
        Implementations should still treat it as advisory and cheap.

        Args:
            lease: The lease whose activity is being reported.
        """
        ...

    def close_local(self, lease: SandboxLease) -> None:
        """Release local client resources for a lease.

        Must not delete a Claim or any other remote infrastructure. Called on
        cache eviction and adapter shutdown, and must tolerate being called
        after the underlying connection is already closed.

        Args:
            lease: The lease whose local resources should be released.
        """
        ...
