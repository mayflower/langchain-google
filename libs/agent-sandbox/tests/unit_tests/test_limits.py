"""Tests for configurable result bounds."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from langchain_google_agent_sandbox import AgentSandboxBackend
from langchain_google_agent_sandbox.limits import SandboxResultLimits

from .test_backend import StubCommands, StubFiles, StubSandbox


@pytest.mark.parametrize(
    "field",
    [
        "execute_output_bytes",
        "read_lines",
        "grep_matches",
        "glob_matches",
        "upload_files",
        "download_files",
    ],
)
@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "8"])
def test_limits_reject_non_positive_or_non_int(field: str, bad: Any) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        SandboxResultLimits(**{field: bad})


def test_limits_defaults_are_positive() -> None:
    limits = SandboxResultLimits()
    assert limits.execute_output_bytes > 0
    assert limits.read_lines > 0


def _backend(limits: SandboxResultLimits, **sandbox_kwargs: Any) -> AgentSandboxBackend:
    return AgentSandboxBackend.from_existing(
        StubSandbox(**sandbox_kwargs), limits=limits
    )


def _stdout(text: str) -> StubCommands:
    return StubCommands([SimpleNamespace(stdout=text, stderr="", exit_code=0)])


def test_execute_output_is_bounded_and_flagged() -> None:
    backend = _backend(
        SandboxResultLimits(execute_output_bytes=10),
        commands=_stdout("x" * 50),
    )
    response = backend.execute("cat big")
    assert response.output == "x" * 10
    assert response.truncated is True


def test_execute_output_under_bound_is_not_flagged() -> None:
    backend = _backend(
        SandboxResultLimits(execute_output_bytes=100),
        commands=_stdout("short"),
    )
    response = backend.execute("echo short")
    assert response.output == "short"
    assert response.truncated is False


def test_execute_bound_never_splits_a_codepoint() -> None:
    """A bound landing mid-character must not yield invalid text."""
    # "ä" is two UTF-8 bytes; a 3-byte bound cuts the second one in half.
    backend = _backend(
        SandboxResultLimits(execute_output_bytes=3),
        commands=_stdout("äää"),
    )
    response = backend.execute("echo")
    assert response.output == "ä"
    assert response.truncated is True


def test_read_is_bounded_and_reports_remainder_via_next_offset() -> None:
    content = "".join(f"line{i}\n" for i in range(20)).encode("utf-8")
    backend = _backend(SandboxResultLimits(read_lines=5), files=StubFiles(read=content))
    result = backend.read("/f.txt", offset=0, limit=1000)
    assert result.file_data["content"].count("\n") == 5
    assert result.start_line == 1
    assert result.end_line == 5
    # The remaining 15 lines are reachable, not dropped.
    assert result.next_offset == 5
    assert result.total_lines == 20


def test_read_caller_limit_below_cap_still_wins() -> None:
    content = "".join(f"line{i}\n" for i in range(20)).encode("utf-8")
    backend = _backend(
        SandboxResultLimits(read_lines=100), files=StubFiles(read=content)
    )
    result = backend.read("/f.txt", offset=0, limit=2)
    assert result.end_line == 2
    assert result.next_offset == 2


def test_grep_is_bounded_and_flagged() -> None:
    hits = "\n".join(f"/workspace/f{i}.py:1:match" for i in range(20))
    backend = _backend(SandboxResultLimits(grep_matches=5), commands=_stdout(hits))
    result = backend.grep("match")
    assert len(result.matches) == 5
    assert result.truncated is True


def test_grep_caller_max_count_below_cap_still_wins() -> None:
    hits = "\n".join(f"/workspace/f{i}.py:1:match" for i in range(20))
    backend = _backend(SandboxResultLimits(grep_matches=15), commands=_stdout(hits))
    result = backend.grep("match", max_count=2)
    assert len(result.matches) == 2
    assert result.truncated is True


def test_grep_under_bound_is_not_flagged() -> None:
    hits = "\n".join(f"/workspace/f{i}.py:1:match" for i in range(3))
    backend = _backend(SandboxResultLimits(grep_matches=100), commands=_stdout(hits))
    result = backend.grep("match")
    assert len(result.matches) == 3
    assert result.truncated is False


def _find_records(count: int) -> str:
    return "".join(f"f\t10\t0\t/workspace/f{i}.py\x00" for i in range(count))


def test_glob_is_bounded_and_flagged() -> None:
    backend = _backend(
        SandboxResultLimits(glob_matches=4), commands=_stdout(_find_records(20))
    )
    result = backend.glob("*.py")
    assert len(result.matches) == 4
    assert result.truncated is True


def test_glob_under_bound_is_not_flagged() -> None:
    backend = _backend(
        SandboxResultLimits(glob_matches=50), commands=_stdout(_find_records(3))
    )
    result = backend.glob("*.py")
    assert len(result.matches) == 3
    assert result.truncated is False


def test_upload_beyond_bound_is_refused_not_dropped() -> None:
    backend = _backend(SandboxResultLimits(upload_files=2))
    payload = {f"/f{i}.txt": b"data" for i in range(5)}
    responses = backend.upload_files(payload)

    # One response per requested file: nothing silently disappears.
    assert len(responses) == 5
    assert [r.path for r in responses] == list(payload)
    assert [r.error for r in responses[2:]] == ["limit_exceeded"] * 3


def test_download_beyond_bound_is_refused_not_dropped() -> None:
    backend = _backend(
        SandboxResultLimits(download_files=1), files=StubFiles(read=b"ok")
    )
    backend._file_state = lambda path: "file"  # type: ignore[method-assign]
    responses = backend.download_files(["/a.txt", "/b.txt", "/c.txt"])

    assert len(responses) == 3
    assert responses[0].error is None
    assert [r.error for r in responses[1:]] == ["limit_exceeded"] * 2
    assert all(r.content is None for r in responses[1:])
