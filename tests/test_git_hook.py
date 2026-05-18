"""Tests for the loom-code post-commit hook installer + the
debouncing logic in :mod:`loom_code._post_commit`.

We exercise the installer against a real ``.git/hooks`` directory
in a tmp_path; we exercise the debouncer by stubbing the refresh
callable and watching the counter file. No real git commits, no
real graphify/loominit subprocess — just the orchestration.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from loom_code.git_hook import _MARKER, install, is_installed, uninstall


@pytest.fixture
def fake_git_repo(tmp_path: Path) -> Path:
    """Create a minimal git-repo-like tmp dir — just enough
    ``.git/`` structure for the hook installer to recognise it."""
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    return tmp_path


def test_install_into_fresh_repo_creates_hook(fake_git_repo: Path) -> None:
    status = install(fake_git_repo)
    assert status == "installed"
    hook = fake_git_repo / ".git" / "hooks" / "post-commit"
    assert hook.is_file()
    body = hook.read_text()
    assert _MARKER in body
    assert "loom_code._post_commit" in body
    # Executable bits set so git will actually run it.
    mode = hook.stat().st_mode
    assert mode & stat.S_IXUSR


def test_install_skips_non_git_dir(tmp_path: Path) -> None:
    # No ``.git`` directory — not a git repo.
    status = install(tmp_path)
    assert status.startswith("skipped")


def test_install_is_idempotent(fake_git_repo: Path) -> None:
    install(fake_git_repo)
    status = install(fake_git_repo)
    assert status == "updated"
    # Only ONE loom-code section should remain in the hook.
    body = (fake_git_repo / ".git" / "hooks" / "post-commit").read_text()
    assert body.count(_MARKER) == 1


def test_install_preserves_existing_hook_content(
    fake_git_repo: Path,
) -> None:
    """Other tools' hooks must survive a loom-code install."""
    hook = fake_git_repo / ".git" / "hooks" / "post-commit"
    hook.write_text(
        "#!/bin/sh\n"
        "# pre-existing hook from another tool\n"
        "echo 'hello from other tool'\n"
    )
    hook.chmod(0o755)
    install(fake_git_repo)
    body = hook.read_text()
    assert "hello from other tool" in body
    assert _MARKER in body


def test_uninstall_removes_loomcode_section(fake_git_repo: Path) -> None:
    install(fake_git_repo)
    assert is_installed(fake_git_repo)
    status = uninstall(fake_git_repo)
    assert status == "removed"
    assert not is_installed(fake_git_repo)


def test_uninstall_preserves_other_tools(fake_git_repo: Path) -> None:
    hook = fake_git_repo / ".git" / "hooks" / "post-commit"
    hook.write_text(
        "#!/bin/sh\necho 'other tool'\n"
    )
    install(fake_git_repo)
    uninstall(fake_git_repo)
    body = hook.read_text()
    assert "other tool" in body
    assert _MARKER not in body


def test_uninstall_drops_file_when_nothing_else_remains(
    fake_git_repo: Path,
) -> None:
    """If we installed into an empty repo (only shebang + our
    block), uninstalling should delete the whole file rather than
    leave a useless ``#!/bin/sh`` shell that other tools might
    misread."""
    install(fake_git_repo)
    uninstall(fake_git_repo)
    hook = fake_git_repo / ".git" / "hooks" / "post-commit"
    assert not hook.exists()


def test_is_installed_false_on_fresh_repo(fake_git_repo: Path) -> None:
    assert not is_installed(fake_git_repo)


def test_is_installed_false_on_non_git_dir(tmp_path: Path) -> None:
    assert not is_installed(tmp_path)


# --- Debouncer logic from _post_commit -----------------------------------


def test_debouncer_increments_until_threshold(tmp_path: Path) -> None:
    """The counter file increments on each post-commit invocation
    and only fires the refresh on the 5th call (default threshold)."""
    from loom_code._post_commit import _maybe_refresh

    counter = tmp_path / "counter.txt"
    fired = [0]

    def refresh() -> None:
        fired[0] += 1

    # 4 invocations: no refresh, counter walks 1 → 4.
    for expected_count in range(1, 5):
        _maybe_refresh(counter_file=counter, refresh_fn=refresh)
        assert fired[0] == 0
        assert counter.read_text() == str(expected_count)

    # 5th invocation: refresh fires, counter resets to 0.
    _maybe_refresh(counter_file=counter, refresh_fn=refresh)
    assert fired[0] == 1
    assert counter.read_text() == "0"


def test_debouncer_swallows_refresh_errors(tmp_path: Path) -> None:
    """A failing refresh callable must NOT crash — git hooks
    breaking commits is the failure mode we're guarding against."""
    from loom_code._post_commit import _maybe_refresh

    counter = tmp_path / "counter.txt"
    # Push the counter to threshold by writing it directly.
    counter.write_text("4")

    def boom() -> None:
        raise RuntimeError("indexer crashed")

    # Should not raise.
    _maybe_refresh(counter_file=counter, refresh_fn=boom)
    # Counter stays at threshold (5) so the next commit retries.
    # We don't reset on failure — half-success would be confusing.
    assert counter.read_text() == "4"


def test_debouncer_recovers_from_corrupt_counter(tmp_path: Path) -> None:
    """A garbled counter file (manual edit, race, partial write)
    resets to 0 — never crashes."""
    from loom_code._post_commit import _maybe_refresh

    counter = tmp_path / "counter.txt"
    counter.write_text("not a number")

    fired = [0]

    def refresh() -> None:
        fired[0] += 1

    # Should not raise, treats garbage as 0, increments to 1.
    _maybe_refresh(counter_file=counter, refresh_fn=refresh)
    assert fired[0] == 0
    assert counter.read_text() == "1"


def test_debouncer_handles_missing_loom_dir(tmp_path: Path) -> None:
    """``_post_commit.main`` is a no-op for projects without a
    ``.loom/`` directory — graceful skip, never touches git."""
    import sys

    from loom_code._post_commit import main

    sys_argv_saved = sys.argv
    try:
        sys.argv = ["_post_commit", str(tmp_path)]
        exit_code = main()
        assert exit_code == 0
    finally:
        sys.argv = sys_argv_saved


# --- Marker placement to satisfy linters that don't like top-level
# imports of ``os`` going unused — used below in case of CI matrix
# differences that platform-gate some tests in the future.
_ = os
