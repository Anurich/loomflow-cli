"""Auto-checkpoint — silent, automatic snapshots for fearless edits.

The retention feature that beats Cursor's: before the agent makes a
batch of changes, loom-code snapshots the ENTIRE working tree (tracked
edits AND untracked new files) as a real git commit object — then the
user can revert one step with ``/undo`` even mid-session, without
knowing git. Cursor's checkpoints are an opaque per-edit snapshot; ours
are inspectable git objects you can ``git show``, durable across a
crash, and they never touch your branch, index, or working tree.

Why not ``git stash``: ``git stash create`` silently DROPS untracked
files (verified) — so a checkpoint taken right before the agent writes
a brand-new file couldn't restore the pre-write state. Instead we build
a snapshot the robust way:

    1. copy the repo's index to a TEMP index file (outside the tree),
    2. ``git add -A`` against that temp index (stages tracked + new),
    3. ``git write-tree`` → a tree object of the full working state,
    4. ``git commit-tree`` → a commit object parented on HEAD.

The real index + working tree are never touched (the temp index
absorbs the staging), so taking a checkpoint is invisible to the user
and to any in-flight git state. Restore is ``git restore --source=
<snapshot> --worktree -- .`` which rewrites the working tree to the
snapshot without moving HEAD or the branch.

The snapshot stack lives in ``.loom/checkpoints.json`` (capped). Like
everything in loom-code's ``.loom`` state, this is best-effort: a git
failure, a non-repo, a disk error — every function degrades to a no-op
+ a returned error string, never an exception that could kill a turn.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CHECKPOINTS_FILENAME = "checkpoints.json"
_SCHEMA_VERSION = 1

# Keep the last N checkpoints. A long session shouldn't accumulate
# thousands of dangling commit objects; old snapshots fall off the
# stack (the commit objects become unreachable + get GC'd by git).
_MAX_CHECKPOINTS = 50


@dataclass(frozen=True)
class Checkpoint:
    """One snapshot: its sequence number, the snapshot commit SHA, a
    one-line summary (the prompt that triggered it), and when."""

    seq: int
    sha: str
    summary: str
    created_at: str


def _git(
    cwd: Path | str, args: list[str], *, env: dict[str, str] | None = None
) -> tuple[int, str, str]:
    """Run a git subcommand. Returns (rc, stdout, stderr); never raises
    on non-zero exit. ``env`` overlays os.environ (used to point git at
    a temp index)."""
    full_env = {**os.environ, **env} if env else None
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=60,
            env=full_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def _is_git_repo(root: Path | str) -> bool:
    return (Path(root) / ".git").exists()


def _checkpoints_path(root: Path | str) -> Path:
    return Path(root) / ".loom" / _CHECKPOINTS_FILENAME


def _load(root: Path | str) -> dict[str, Any]:
    """Load the checkpoint stack, or a fresh empty one. Never raises."""
    path = _checkpoints_path(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": _SCHEMA_VERSION, "next_seq": 1, "stack": []}
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("stack"), list)
    ):
        return {"version": _SCHEMA_VERSION, "next_seq": 1, "stack": []}
    return data


def _save(root: Path | str, data: dict[str, Any]) -> None:
    """Persist the stack. Best-effort — a disk error silently no-ops."""
    path = _checkpoints_path(root)
    try:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _snapshot_commit(root: Path) -> tuple[str | None, str]:
    """Build a commit object capturing the FULL working tree (tracked +
    untracked, excluding .loom) without touching the real index/tree.

    Returns ``(sha, "")`` or ``(None, error)``. The technique: stage
    everything into a throwaway temp index, write a tree from it, and
    commit-tree that parented on HEAD.
    """
    # HEAD is the parent. A repo with no commits yet has no HEAD — we
    # snapshot with no parent in that case (root commit).
    rc, head, _ = _git(root, ["rev-parse", "HEAD"])
    parent = head.strip() if rc == 0 else ""

    # Temp index OUTSIDE the worktree so it never appears in the
    # snapshot (an in-tree temp index leaked itself into the tree —
    # verified). Seed it from the real index so unchanged staged state
    # is preserved; if there's no index yet, git creates one.
    tmp_fd, tmp_index = tempfile.mkstemp(prefix="loom-ckpt-index-")
    os.close(tmp_fd)
    try:
        real_index = root / ".git" / "index"
        if real_index.is_file():
            try:
                Path(tmp_index).write_bytes(real_index.read_bytes())
            except OSError:
                pass  # start from empty temp index
        env = {"GIT_INDEX_FILE": tmp_index}
        # Stage tracked changes + untracked files. .loom is excluded so
        # the snapshot doesn't churn on our own state files (it's also
        # usually gitignored, but be explicit).
        rc, _o, err = _git(
            root, ["add", "-A", "--", ".", ":!.loom"], env=env
        )
        if rc != 0:
            return None, f"git add failed: {err.strip()}"
        rc, tree, err = _git(root, ["write-tree"], env=env)
        if rc != 0:
            return None, f"write-tree failed: {err.strip()}"
        tree = tree.strip()
        commit_args = ["commit-tree", tree, "-m", "loom checkpoint"]
        if parent:
            commit_args[1:1] = ["-p", parent]
        rc, sha, err = _git(root, commit_args, env=env)
        if rc != 0:
            return None, f"commit-tree failed: {err.strip()}"
        return sha.strip(), ""
    finally:
        try:
            os.unlink(tmp_index)
        except OSError:
            pass


def checkpoint(
    root: Path | str, summary: str = ""
) -> tuple[Checkpoint | None, str]:
    """Take a checkpoint of the current working tree. Returns
    ``(Checkpoint, "")`` or ``(None, error)``.

    Called automatically before a turn that will write. A no-op (returns
    an error string, never raises) when ``root`` isn't a git repo — the
    feature simply doesn't apply outside version control.
    """
    root = Path(root)
    if not _is_git_repo(root):
        return None, "not a git repository"
    sha, err = _snapshot_commit(root)
    if sha is None:
        return None, err
    data = _load(root)
    seq = int(data.get("next_seq", 1))
    cp = Checkpoint(
        seq=seq,
        sha=sha,
        summary=summary.replace("\n", " ").strip()[:200],
        created_at=_now_iso(),
    )
    stack: list[dict[str, Any]] = data.get("stack", [])
    stack.append(
        {
            "seq": cp.seq,
            "sha": cp.sha,
            "summary": cp.summary,
            "created_at": cp.created_at,
        }
    )
    # Cap the stack — oldest fall off (their commit objects become
    # unreachable and git GCs them eventually).
    if len(stack) > _MAX_CHECKPOINTS:
        stack = stack[-_MAX_CHECKPOINTS:]
    data["stack"] = stack
    data["next_seq"] = seq + 1
    data["version"] = _SCHEMA_VERSION
    _save(root, data)
    return cp, ""


def list_checkpoints(root: Path | str) -> list[Checkpoint]:
    """The checkpoint stack, most-recent LAST (so [-1] is the latest).
    Never raises."""
    data = _load(root)
    out: list[Checkpoint] = []
    for item in data.get("stack", []):
        try:
            out.append(
                Checkpoint(
                    seq=int(item["seq"]),
                    sha=str(item["sha"]),
                    summary=str(item.get("summary", "")),
                    created_at=str(item.get("created_at", "")),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return out


def restore(
    root: Path | str, seq: int | None = None
) -> tuple[Checkpoint | None, str]:
    """Restore the working tree to checkpoint ``seq`` (or the latest
    when ``seq`` is None). Returns ``(restored_checkpoint, "")`` or
    ``(None, error)``.

    Rewrites the WORKING TREE to the snapshot via ``git restore
    --source`` — HEAD and the branch are untouched, so this is a pure
    working-tree revert, not a history rewrite. Files the snapshot
    didn't have are NOT deleted (git restore only writes tracked-in-
    snapshot paths); this is intentionally conservative — we never rm a
    user file on undo. Before restoring we take a SAFETY checkpoint of
    the current state, so ``/undo`` is itself undoable (redo).
    """
    root = Path(root)
    if not _is_git_repo(root):
        return None, "not a git repository"
    checkpoints = list_checkpoints(root)
    if not checkpoints:
        return None, "no checkpoints to restore"
    if seq is None:
        target = checkpoints[-1]
    else:
        match = [c for c in checkpoints if c.seq == seq]
        if not match:
            return None, f"no checkpoint #{seq}"
        target = match[0]

    # Safety net: snapshot the current (pre-restore) state so an
    # accidental /undo can itself be undone. Best-effort; a failure here
    # doesn't block the restore the user asked for.
    checkpoint(root, summary=f"before restoring #{target.seq}")

    rc, _o, err = _git(
        root, ["restore", "--source", target.sha, "--worktree", "--", "."]
    )
    if rc != 0:
        # Older git without ``restore``: fall back to checkout-tree.
        rc2, _o2, err2 = _git(
            root, ["checkout", target.sha, "--", "."]
        )
        if rc2 != 0:
            return None, f"restore failed: {(err or err2).strip()}"
    return target, ""
