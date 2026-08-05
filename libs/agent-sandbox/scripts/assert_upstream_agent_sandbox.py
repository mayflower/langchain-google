#!/usr/bin/env python
"""Assert the SDK dependency resolves to the official PyPI distribution.

This package must depend on unmodified upstream ``k8s-agent-sandbox``. A fork,
a Git dependency, or a local direct reference would silently reintroduce the
private control-plane surface this adapter was built to shed, so the build
fails loudly instead.

Usage:
    python scripts/assert_upstream_agent_sandbox.py [dist/*.whl ...]

Checks, in order:

1. ``pyproject.toml`` pins the exact expected version and uses no direct
   reference (``@ git+``, ``@ file:``, ``@ http``).
2. ``uv.lock`` records the distribution with a registry source, not a git or
   directory source.
3. The installed distribution, if importable, reports the expected version and
   carries no ``direct_url.json`` recording a URL install.
4. Any wheel paths passed as arguments declare the expected pin in their
   metadata.
5. No tracked file mentions a known fork coordinate.
"""

from __future__ import annotations

import json
import sys
import tomllib
import zipfile
from importlib import metadata
from pathlib import Path

EXPECTED_NAME = "k8s-agent-sandbox"
EXPECTED_VERSION = "0.5.4"
EXPECTED_PIN = f"{EXPECTED_NAME}=={EXPECTED_VERSION}"

#: Coordinates of the retired Mayflower fork. Their reappearance anywhere in
#: tracked sources means the cutover regressed.
FORBIDDEN_STRINGS = (
    "github.com/mayflower/agent-sandbox",
    "mayflower/agent-sandbox",
    "a2419b9b7eaeec99f636c46a46ca55731d3f52fc",
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

#: Only text worth scanning; skip caches, virtualenvs, and build output.
SCAN_SUFFIXES = {".py", ".toml", ".lock", ".md", ".yml", ".yaml", ".cfg", ".txt"}
SKIP_DIRS = {
    ".venv",
    ".git",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}


class ProvenanceError(AssertionError):
    """A dependency-provenance guarantee was violated."""


def check_pyproject(path: Path) -> None:
    """Verify the declared dependency is an exact PyPI pin."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    dependencies = data.get("project", {}).get("dependencies", [])
    matches = [dep for dep in dependencies if EXPECTED_NAME in dep]
    if not matches:
        msg = f"{path}: no {EXPECTED_NAME} dependency declared"
        raise ProvenanceError(msg)
    for dep in matches:
        if "@" in dep:
            msg = (
                f"{path}: {EXPECTED_NAME} uses a direct reference ({dep!r}); "
                f"expected the plain pin {EXPECTED_PIN!r}"
            )
            raise ProvenanceError(msg)
        if dep.replace(" ", "") != EXPECTED_PIN:
            msg = f"{path}: expected {EXPECTED_PIN!r}, found {dep!r}"
            raise ProvenanceError(msg)


def check_lock(path: Path) -> None:
    """Verify the lockfile resolves the SDK from a registry."""
    if not path.exists():
        msg = f"{path}: lockfile is missing"
        raise ProvenanceError(msg)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    for package in data.get("package", []):
        if package.get("name") != EXPECTED_NAME:
            continue
        version = package.get("version")
        if version != EXPECTED_VERSION:
            msg = (
                f"{path}: locked {EXPECTED_NAME} {version}, expected {EXPECTED_VERSION}"
            )
            raise ProvenanceError(msg)
        source = package.get("source", {})
        if "registry" not in source:
            msg = (
                f"{path}: {EXPECTED_NAME} resolves from {source!r}; "
                "expected a registry source"
            )
            raise ProvenanceError(msg)
        return
    msg = f"{path}: {EXPECTED_NAME} is not present in the lockfile"
    raise ProvenanceError(msg)


def check_installed() -> None:
    """Verify the installed distribution is the expected registry build."""
    try:
        dist = metadata.distribution(EXPECTED_NAME)
    except metadata.PackageNotFoundError:
        return  # Nothing installed in this environment; other checks still apply.
    if dist.version != EXPECTED_VERSION:
        msg = f"installed {EXPECTED_NAME} {dist.version}, expected {EXPECTED_VERSION}"
        raise ProvenanceError(msg)
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        url = json.loads(direct_url).get("url", "")
        msg = (
            f"installed {EXPECTED_NAME} came from a direct URL ({url}); "
            "expected a PyPI registry install"
        )
        raise ProvenanceError(msg)


def check_wheel(path: Path) -> None:
    """Verify a built wheel declares the expected pin."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if not names:
            msg = f"{path}: wheel has no METADATA"
            raise ProvenanceError(msg)
        metadata_text = archive.read(names[0]).decode("utf-8")

    requires = [
        line.split(":", 1)[1].strip()
        for line in metadata_text.splitlines()
        if line.lower().startswith("requires-dist:")
    ]
    matches = [req for req in requires if EXPECTED_NAME in req]
    if not matches:
        msg = f"{path}: wheel does not require {EXPECTED_NAME}"
        raise ProvenanceError(msg)
    for req in matches:
        normalized = req.replace(" ", "")
        if "@" in normalized or normalized != EXPECTED_PIN:
            msg = f"{path}: wheel requires {req!r}, expected {EXPECTED_PIN!r}"
            raise ProvenanceError(msg)


def check_no_fork_references(root: Path) -> None:
    """Verify no tracked source mentions the retired fork."""
    offenders: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue  # This guard names the strings it forbids.
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        offenders.extend(
            f"{path.relative_to(root)}: {needle}"
            for needle in FORBIDDEN_STRINGS
            if needle in text
        )
    if offenders:
        listed = "\n  ".join(offenders)
        msg = f"fork references must not reappear:\n  {listed}"
        raise ProvenanceError(msg)


def resolve_wheels(patterns: list[str]) -> list[Path]:
    """Expand wheel arguments, which a shell may or may not have globbed."""
    wheels: list[Path] = []
    for pattern in patterns:
        matched = sorted(Path().glob(pattern))
        wheels.extend(matched or [Path(pattern)])
    return wheels


def main(argv: list[str]) -> int:
    """Run every provenance check, reporting the first failure."""
    try:
        check_pyproject(PACKAGE_ROOT / "pyproject.toml")
        check_lock(PACKAGE_ROOT / "uv.lock")
        check_installed()
        check_no_fork_references(PACKAGE_ROOT)
        for wheel in resolve_wheels(argv):
            check_wheel(wheel)
    except ProvenanceError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"OK: {EXPECTED_PIN} resolves from PyPI with no fork references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
