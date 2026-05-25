"""Install / uninstall loom-code's debounced post-commit hook.

The hook (a small shell wrapper around ``python -m
loom_code._post_commit``) counts commits since the last refresh
per indexer and runs an incremental rebuild every N commits
(default 5). See :mod:`loom_code._post_commit` for the actual
refresh logic.

Why this lives in loom-code and not graphify / loominit
individually: there's only ONE ``.git/hooks/post-commit`` slot
per repo. A shared hook that checks which indexers are set up
and refreshes whichever apply lets ``/graphify on`` and
``/loominit`` coexist without clobbering each other.

The installer is idempotent. It marks its lines with a
``# loom-code-hook`` sentinel so subsequent installs detect the
prior hook and skip; uninstall removes those lines and leaves
any other hook content (other tools, manual scripts) intact.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

# Sentinel comment we write to mark our section of the hook.
# Anything between this sentinel and the next blank line is
# considered loom-code's; the rest is left alone on uninstall.
_MARKER = "# loom-code-hook"
_HOOK_NAME = "post-commit"


def _git_hooks_dir(project_root: Path) -> Path | None:
    """Resolve the git hooks directory for ``project_root`` — or
    ``None`` if this isn't a git repo. Respects ``core.hooksPath``
    when set (some teams move hooks out of ``.git/``)."""
    git_dir = project_root / ".git"
    if not git_dir.exists():
        return None
    # Honour ``git config core.hooksPath`` if set. Sub-projects in
    # a workspace + teams using shared-hook tooling rely on this.
    import subprocess
    try:
        result = subprocess.run(
            ["git", "config", "--get", "core.hooksPath"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        result = None
    if result is not None and result.returncode == 0 and result.stdout.strip():
        hooks_path = result.stdout.strip()
        # Relative paths are relative to the worktree root, not .git.
        return (project_root / hooks_path).resolve()
    # Default: .git/hooks (or hooks/ inside a bare repo).
    if git_dir.is_dir():
        return git_dir / "hooks"
    # ``.git`` could be a file (worktree / submodule). Resolve.
    return _resolve_dotgit_file(git_dir) / "hooks"


def _resolve_dotgit_file(git_file: Path) -> Path:
    """Worktrees + submodules: ``.git`` is a regular file with a
    ``gitdir: <abs path>`` pointer. Follow it to the real git dir."""
    text = git_file.read_text(encoding="utf-8").strip()
    for line in text.splitlines():
        if line.startswith("gitdir:"):
            return Path(line.split(":", 1)[1].strip())
    return git_file.parent  # fallback


def is_installed(project_root: Path) -> bool:
    """True iff the loom-code section is present in the post-commit
    hook for this project."""
    hooks_dir = _git_hooks_dir(project_root)
    if hooks_dir is None:
        return False
    hook_path = hooks_dir / _HOOK_NAME
    if not hook_path.is_file():
        return False
    return _MARKER in hook_path.read_text(encoding="utf-8")


def install(project_root: Path) -> str:
    """Install (or refresh) the loom-code post-commit hook.

    Returns a status string suitable for printing to the user:
    ``"installed"`` (newly added), ``"updated"`` (replaced an
    existing loom-code section), ``"skipped: not a git repo"``,
    or ``"skipped: <reason>"`` on failure.

    Idempotent — calling on an already-installed repo
    just refreshes the section in case the marker / runner
    path changed between loom-code versions.
    """
    hooks_dir = _git_hooks_dir(project_root)
    if hooks_dir is None:
        return "skipped: not a git repo"
    try:
        hooks_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"skipped: cannot mkdir {hooks_dir}: {exc}"
    hook_path = hooks_dir / _HOOK_NAME

    # The line we want present. Background (``&``) + stdout/err
    # redirect so the hook never blocks the commit or spams the
    # terminal. ``sys.executable`` pins to whichever Python
    # loom-code is running in — avoids the "wrong python on PATH"
    # class of bug.
    loomcode_block = (
        f"{_MARKER}\n"
        f"exec {sys.executable} -m loom_code._post_commit "
        f'"$(git rev-parse --show-toplevel)" '
        f">/dev/null 2>&1 &\n"
    )

    updated = False
    if hook_path.is_file():
        existing = hook_path.read_text(encoding="utf-8")
        if _MARKER in existing:
            # Replace the existing section.
            existing = _strip_loomcode_section(existing)
            updated = True
        # Append our block to whatever else is there.
        new_content = existing.rstrip() + "\n\n" + loomcode_block
    else:
        new_content = "#!/bin/sh\n" + loomcode_block

    try:
        hook_path.write_text(new_content, encoding="utf-8")
        # Mark executable (mode 0o755) — git won't run a hook
        # without +x on POSIX.
        st = hook_path.stat()
        exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        hook_path.chmod(st.st_mode | exec_bits)
    except OSError as exc:
        return f"skipped: write failed: {exc}"
    return "updated" if updated else "installed"


def uninstall(project_root: Path) -> str:
    """Remove loom-code's section from the post-commit hook.
    Leaves other tools' hook lines intact. Returns ``"removed"``,
    ``"not present"``, or ``"skipped: <reason>"``."""
    hooks_dir = _git_hooks_dir(project_root)
    if hooks_dir is None:
        return "skipped: not a git repo"
    hook_path = hooks_dir / _HOOK_NAME
    if not hook_path.is_file():
        return "not present"
    existing = hook_path.read_text(encoding="utf-8")
    if _MARKER not in existing:
        return "not present"
    stripped = _strip_loomcode_section(existing)
    # If only the shebang remains (or empty), drop the file entirely
    # so we don't leave a no-op behind that other tools might find.
    if stripped.strip() in ("", "#!/bin/sh"):
        try:
            hook_path.unlink()
        except OSError as exc:
            return f"skipped: unlink failed: {exc}"
    else:
        try:
            hook_path.write_text(stripped, encoding="utf-8")
        except OSError as exc:
            return f"skipped: write failed: {exc}"
    return "removed"


def _strip_loomcode_section(content: str) -> str:
    """Remove the ``_MARKER`` line + the ``exec ...`` line that
    follows it. Preserves everything else in the hook file."""
    lines = content.splitlines()
    out: list[str] = []
    skip_count = 0
    for line in lines:
        if skip_count > 0:
            skip_count -= 1
            continue
        if line.strip() == _MARKER:
            # Skip this line + the exec line that comes right
            # after. Two-line block per ``install()``.
            skip_count = 1
            continue
        out.append(line)
    # Collapse any resulting double blank lines.
    cleaned: list[str] = []
    prev_blank = False
    for line in out:
        if not line.strip():
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        cleaned.append(line)
    return "\n".join(cleaned).rstrip() + "\n"
