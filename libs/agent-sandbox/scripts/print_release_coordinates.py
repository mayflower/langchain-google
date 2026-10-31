#!/usr/bin/env python
"""Print the immutable coordinates a consuming product must pin.

The companion mAIstack pack pins this package by commit SHA, never by branch,
because a branch moves and would silently change what a product depends on.
Run this after the work is committed and emit the result into that pack.

Usage:
    python scripts/print_release_coordinates.py

Refuses to print a SHA when the working tree is dirty: the coordinates would
describe a commit that does not contain what was actually tested.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SDK_NAME = "k8s-agent-sandbox"


def git(*args: str) -> str:
    """Run a git command inside the package's repository."""
    return subprocess.run(
        ["git", *args],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def package_version() -> str:
    """Read the declared version, preferring the installed distribution."""
    data = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def sdk_version() -> str:
    """Report the pinned SDK version, from the installed distribution if present."""
    try:
        return metadata.version(SDK_NAME)
    except metadata.PackageNotFoundError:
        data = tomllib.loads(
            (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        for dep in data["project"]["dependencies"]:
            if dep.replace(" ", "").startswith(f"{SDK_NAME}=="):
                return dep.replace(" ", "").split("==", 1)[1]
        msg = f"cannot determine {SDK_NAME} version"
        raise SystemExit(msg) from None


def main() -> int:
    """Emit the record block, or explain why it would be misleading."""
    dirty = git("status", "--porcelain")
    if dirty:
        print(
            "refusing to print coordinates: the working tree has uncommitted "
            "changes, so the SHA would not describe what was tested.\n"
            f"{dirty}",
            file=sys.stderr,
        )
        return 1

    print(f"LANGCHAIN_AGENT_SANDBOX_COMMIT_SHA={git('rev-parse', 'HEAD')}")
    print(f"LANGCHAIN_AGENT_SANDBOX_PACKAGE_VERSION={package_version()}")
    print(f"K8S_AGENT_SANDBOX_SDK_VERSION={sdk_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
