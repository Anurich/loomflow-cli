"""Git worktree lifecycle for session isolation.

An *isolated* session edits in its own git worktree on branch
``loom/<session_id>`` — a separate working copy of the same repo — so
two concurrent sessions on one repo (two ``loom-code`` terminals, two
desktop chat tabs) can't collide on disk. When done, the branch is
either merged into the base branch or discarded, and the worktree is
removed.

This is the SHARED lifecycle: the loom-code CLI drives it via
``/isolate`` `/review` `/merge` `/discard`, and the desktop sidecar can
use the same functions instead of its own copy.

Git-only. Every function degrades gracefully — git failures come back
as ``(None, error)`` / ``(False, error)`` rather than raising, so a
bad git state never crashes the REPL.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorktreeInfo:
    """A live session worktree: where it is, its branch, and the base
    branch it was forked from (and will merge back into)."""

    path: Path
    branch: str
    base: str


def _git(
    cwd: Path | str, args: list[str], *, timeout: int = 60
) -> tuple[int, str, str]:
    """Run a git subcommand in ``cwd``. Returns (returncode, out, err);
    never raises on a non-zero exit — callers inspect the code."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def is_git_repo(root: Path | str) -> bool:
    return (Path(root) / ".git").exists()


def worktree_path(root: Path | str, session_id: str) -> Path:
    return Path(root) / ".loom" / "worktrees" / session_id


def branch_name(session_id: str) -> str:
    # session ids (chat-<...> / ULIDs) are ref-safe; "loom/" namespaces
    # them so they're easy to spot + bulk-clean.
    return f"loom/{session_id}"


def current_branch(root: Path | str) -> str:
    rc, out, _ = _git(root, ["symbolic-ref", "--short", "HEAD"])
    return out.strip() if rc == 0 and out.strip() else "HEAD"


def create(
    root: Path | str, session_id: str
) -> tuple[WorktreeInfo | None, str]:
    """Create a worktree for ``session_id``. Returns ``(info, "")`` or
    ``(None, error)``. Idempotent-ish: if the branch already exists
    (re-isolate after a discard) it's reused."""
    root = Path(root)
    if not is_git_repo(root):
        return None, "not a git repository"
    wt = worktree_path(root, session_id)
    branch = branch_name(session_id)
    base = current_branch(root)
    try:
        wt.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return None, f"mkdir failed: {exc}"
    rc, _out, err = _git(root, ["worktree", "add", str(wt), "-b", branch])
    if rc != 0:
        rc2, _o2, err2 = _git(root, ["worktree", "add", str(wt), branch])
        if rc2 != 0:
            return None, (err or err2).strip()
    return WorktreeInfo(path=wt, branch=branch, base=base), ""


def diff(info: WorktreeInfo) -> tuple[str, str]:
    """Unified diff of the worktree (committed-on-branch + uncommitted)
    vs the base branch. Returns ``(diff_text, error)``."""
    rc, out, err = _git(info.path, ["diff", info.base])
    if rc != 0:
        return "", err.strip()
    return out, ""


def merge(root: Path | str, info: WorktreeInfo) -> tuple[bool, str]:
    """Commit the worktree's uncommitted edits (only when dirty), then
    merge its branch into the base branch from the MAIN tree. Returns
    ``(ok, error)``. Refuses if the main tree isn't on the base branch
    (to avoid merging into the wrong branch); aborts on conflict."""
    root = Path(root)
    rc, out, _ = _git(info.path, ["status", "--porcelain"])
    if rc == 0 and out.strip():
        _git(info.path, ["add", "-A"])
        _git(info.path, ["commit", "-m", f"loom session {info.branch}"])
    cur = current_branch(root)
    if cur != info.base:
        return False, (
            f"the main working tree is on '{cur}', not the session's "
            f"base '{info.base}' — switch back before merging"
        )
    rc, _o, err = _git(root, ["merge", "--no-edit", info.branch])
    if rc != 0:
        _git(root, ["merge", "--abort"])
        return False, f"merge conflict: {err.strip()}"
    return True, ""


def remove(root: Path | str, info: WorktreeInfo) -> None:
    """Remove the worktree + delete its branch. Best-effort — used on
    both discard and post-merge cleanup."""
    root = Path(root)
    _git(root, ["worktree", "remove", "--force", str(info.path)])
    _git(root, ["branch", "-D", info.branch])
