"""Tests for auto-checkpoint (loom_code.checkpoint).

Drives a REAL temp git repo (git is a hard dep of the feature) so the
snapshot/restore round-trip is exercised against actual git plumbing,
not a mock. Offline + sync.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from loom_code import checkpoint as cp


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _init_repo(tmp_path: Path) -> Path:
    root = tmp_path
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.co")
    _git(root, "config", "user.name", "t")
    (root / ".loom").mkdir()
    (root / "a.py").write_text("v1\n")
    _git(root, "add", "-A")
    # Commit needs the hook-free path; tests run git directly so the
    # project's commit-block hook (a Claude harness hook) doesn't apply.
    _git(root, "commit", "-qm", "init")
    return root


def test_checkpoint_then_restore_tracked(tmp_path: Path) -> None:
    root = _init_repo(tmp_path)
    # Dirty a tracked file, snapshot, dirty further, restore.
    (root / "a.py").write_text("v2\n")
    snap, err = cp.checkpoint(root, summary="edit a")
    assert snap is not None, err
    assert snap.seq == 1
    (root / "a.py").write_text("v3-LATER\n")
    restored, err = cp.restore(root, snap.seq)
    assert restored is not None, err
    assert (root / "a.py").read_text() == "v2\n"


def test_checkpoint_captures_untracked(tmp_path: Path) -> None:
    # The whole reason we don't use ``git stash create``: untracked
    # files must be in the snapshot.
    root = _init_repo(tmp_path)
    (root / "new.py").write_text("brand new\n")
    snap, err = cp.checkpoint(root, summary="add new.py")
    assert snap is not None, err
    # Snapshot commit should contain new.py.
    files = _git(root, "ls-tree", "--name-only", "-r", snap.sha)
    assert "new.py" in files


def test_snapshot_does_not_touch_tree_or_index(tmp_path: Path) -> None:
    root = _init_repo(tmp_path)
    (root / "a.py").write_text("dirty\n")
    (root / "untracked.py").write_text("x\n")

    def _code_status() -> list[str]:
        # Ignore .loom/ churn: checkpoint() legitimately writes
        # checkpoints.json there, which makes .loom appear in status.
        # The invariant we assert is that the user's CODE tree + index
        # are untouched by snapshotting — not loom-code's own state dir.
        out = _git(root, "status", "--porcelain")
        return [
            ln
            for ln in out.splitlines()
            if ".loom/" not in ln and not ln.rstrip().endswith(".loom")
        ]

    before = _code_status()
    cp.checkpoint(root, summary="snap")
    after = _code_status()
    assert before == after
    assert (root / "a.py").read_text() == "dirty\n"
    assert (root / "untracked.py").read_text() == "x\n"


def test_loom_dir_excluded_from_snapshot(tmp_path: Path) -> None:
    root = _init_repo(tmp_path)
    (root / ".loom" / "junk.tmp").write_text("internal state\n")
    snap, err = cp.checkpoint(root, summary="snap")
    assert snap is not None, err
    files = _git(root, "ls-tree", "--name-only", "-r", snap.sha)
    assert ".loom" not in files


def test_list_checkpoints_ordering(tmp_path: Path) -> None:
    root = _init_repo(tmp_path)
    (root / "a.py").write_text("c1\n")
    cp.checkpoint(root, summary="one")
    (root / "a.py").write_text("c2\n")
    cp.checkpoint(root, summary="two")
    cps = cp.list_checkpoints(root)
    assert [c.seq for c in cps] == [1, 2]
    assert cps[-1].summary == "two"  # most recent last


def test_restore_latest_when_seq_none(tmp_path: Path) -> None:
    root = _init_repo(tmp_path)
    (root / "a.py").write_text("first\n")
    cp.checkpoint(root)
    (root / "a.py").write_text("second\n")
    cp.checkpoint(root)
    (root / "a.py").write_text("third-current\n")
    restored, err = cp.restore(root)  # latest
    assert restored is not None, err
    # Latest checkpoint captured "second".
    assert (root / "a.py").read_text() == "second\n"


def test_restore_takes_safety_checkpoint(tmp_path: Path) -> None:
    # /undo must itself be undoable: restoring snapshots current state.
    root = _init_repo(tmp_path)
    (root / "a.py").write_text("cp1\n")
    cp.checkpoint(root)  # seq 1
    n_before = len(cp.list_checkpoints(root))
    cp.restore(root, 1)
    n_after = len(cp.list_checkpoints(root))
    assert n_after == n_before + 1  # a safety checkpoint was added


def test_non_git_repo_is_graceful(tmp_path: Path) -> None:
    (tmp_path / ".loom").mkdir()
    snap, err = cp.checkpoint(tmp_path)
    assert snap is None
    assert "not a git" in err.lower()
    restored, err2 = cp.restore(tmp_path)
    assert restored is None
    assert "not a git" in err2.lower()


def test_restore_unknown_seq(tmp_path: Path) -> None:
    root = _init_repo(tmp_path)
    (root / "a.py").write_text("x\n")
    cp.checkpoint(root)
    restored, err = cp.restore(root, 999)
    assert restored is None
    assert "999" in err


def test_no_checkpoints_to_restore(tmp_path: Path) -> None:
    root = _init_repo(tmp_path)
    restored, err = cp.restore(root)
    assert restored is None
    assert "no checkpoint" in err.lower()
