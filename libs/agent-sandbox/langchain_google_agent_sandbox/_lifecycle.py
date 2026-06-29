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
from typing import Any

logger = logging.getLogger(__name__)


def factory_atexit_cleanup(sdk_client: Any, sandbox: Any) -> None:
    """Best-effort sandbox teardown for factory-created backends."""
    if sdk_client is None or sandbox is None:
        return
    claim = getattr(sandbox, "claim_name", None)
    namespace = getattr(sandbox, "namespace", None) or "default"
    if claim is None:
        return
    try:
        sdk_client.delete_sandbox(claim_name=claim, namespace=namespace)
    except Exception as exc:
        status = getattr(exc, "status", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        if status == 404:
            return
        try:
            logger.error(
                "Finalizer failed to delete sandbox (claim=%s, namespace=%s): %s: %s",
                claim,
                namespace,
                type(exc).__name__,
                exc,
            )
        except Exception:  # pragma: no cover - interpreter shutdown guard
            pass
