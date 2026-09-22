"""Run the release-time provenance guard as part of the normal test suite.

`scripts/assert_upstream_agent_sandbox.py` is the build-time gate. Wiring it in
here means a fork reference or a drifted pin fails an ordinary `pytest` run
too, rather than only at release, when it is most expensive to discover.

These tests also assert the guard *fails* on each violation class. A guard
nobody has watched fail is a guard nobody knows works.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = PACKAGE_ROOT / "scripts" / "assert_upstream_agent_sandbox.py"


def load_guard() -> ModuleType:
    """Import the guard script as a module without putting scripts/ on sys.path."""
    spec = importlib.util.spec_from_file_location("_provenance_guard", GUARD_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guard() -> ModuleType:
    return load_guard()


def test_guard_script_exists() -> None:
    assert GUARD_PATH.is_file()


def test_repository_passes_the_provenance_guard(guard: ModuleType) -> None:
    """The real check: this checkout must satisfy every provenance rule."""
    assert guard.main([]) == 0


def test_declared_dependency_is_the_exact_pypi_pin(guard: ModuleType) -> None:
    guard.check_pyproject(PACKAGE_ROOT / "pyproject.toml")


def test_lockfile_resolves_from_a_registry(guard: ModuleType) -> None:
    guard.check_lock(PACKAGE_ROOT / "uv.lock")


def test_installed_distribution_is_not_a_direct_url_install(
    guard: ModuleType,
) -> None:
    guard.check_installed()


def test_guard_scans_the_repository_root_not_only_the_package(
    guard: ModuleType,
) -> None:
    """A fork reference can hide in a repo-root workflow file or Dockerfile."""
    root = guard.scan_root()
    assert (root / ".git").exists()
    # Scanning must reach outside libs/agent-sandbox when in a git checkout.
    assert root == PACKAGE_ROOT or PACKAGE_ROOT.is_relative_to(root)


def test_dockerfiles_are_scanned_despite_having_no_suffix(
    guard: ModuleType,
) -> None:
    assert "Dockerfile" in guard.SCAN_NAMES


def test_guard_and_architecture_fence_forbid_identical_strings(
    guard: ModuleType,
) -> None:
    """Divergence here once made the fence strictly weaker than the guard."""
    from tests.unit_tests.test_architecture import FORK_COORDINATES

    assert set(guard.FORBIDDEN_STRINGS) == set(FORK_COORDINATES)


# --- the guard must actually bite ------------------------------------------


def test_guard_rejects_a_git_direct_reference(
    guard: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        "[project]\ndependencies = "
        '["k8s-agent-sandbox @ git+https://example.invalid/fork.git@abc"]\n'
    )
    with pytest.raises(guard.ProvenanceError, match="direct reference"):
        guard.check_pyproject(path)


@pytest.mark.parametrize(
    "spec", ["k8s-agent-sandbox==0.5.2", "k8s-agent-sandbox>=0.5.4"]
)
def test_guard_rejects_a_drifted_or_loose_pin(
    guard: ModuleType, tmp_path: Path, spec: str
) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(f'[project]\ndependencies = ["{spec}"]\n')
    with pytest.raises(guard.ProvenanceError, match="expected"):
        guard.check_pyproject(path)


def test_guard_rejects_a_missing_dependency(guard: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\ndependencies = ["wcmatch>=11.0"]\n')
    with pytest.raises(guard.ProvenanceError, match="no k8s-agent-sandbox"):
        guard.check_pyproject(path)


def test_guard_rejects_a_lockfile_resolving_from_git(
    guard: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "uv.lock"
    path.write_text(
        '[[package]]\nname = "k8s-agent-sandbox"\nversion = "1.0.3"\n'
        '[package.source]\ngit = "https://example.invalid/fork.git"\n'
    )
    with pytest.raises(guard.ProvenanceError, match="registry source"):
        guard.check_lock(path)


def test_guard_rejects_a_missing_lockfile(guard: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(guard.ProvenanceError, match="lockfile is missing"):
        guard.check_lock(tmp_path / "absent.lock")


def test_guard_rejects_a_reintroduced_fork_reference(
    guard: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "somewhere.py").write_text(
        f"URL = 'https://{guard.FORBIDDEN_STRINGS[0]}'\n"
    )
    with pytest.raises(guard.ProvenanceError, match="must not reappear"):
        guard.check_no_fork_references(tmp_path)


def test_guard_rejects_a_fork_reference_in_a_dockerfile(
    guard: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "Dockerfile").write_text(f"# {guard.FORBIDDEN_STRINGS[1]}\n")
    with pytest.raises(guard.ProvenanceError, match="must not reappear"):
        guard.check_no_fork_references(tmp_path)


def test_guard_rejects_a_fork_reference_in_a_workflow_file(
    guard: ModuleType, tmp_path: Path
) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(f"# {guard.FORBIDDEN_STRINGS[2]}\n")
    with pytest.raises(guard.ProvenanceError, match="must not reappear"):
        guard.check_no_fork_references(tmp_path)


def test_historical_references_are_allowed_to_name_the_fork(
    guard: ModuleType, tmp_path: Path
) -> None:
    """The ADR and the guards themselves must not trip their own check."""
    for name in guard.HISTORICAL_REFERENCES:
        (tmp_path / name).write_text(f"# {guard.FORBIDDEN_STRINGS[0]}\n")
    guard.check_no_fork_references(tmp_path)
