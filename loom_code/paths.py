"""Shared file-path resolution — the ``expandPath`` equivalent.

loom-code follows Claude Code's boundary model: the file *tools*
resolve any path the OS can name (``~``-expansion, cwd-relative,
absolute), and the SECURITY boundary lives in the permission layer
(:mod:`loom_code.permissions` + the approval gate), NOT in a hard
cwd wall inside the tool. This is the opposite of loomflow's built-in
tools, whose ``_resolve_within`` raises on anything outside the
workdir — which is why pointing loom-code at a file one directory up
used to fail with "file not found" no matter the path.

``resolve_path`` is the single canonical resolver every loom-code
file tool runs its ``path`` argument through:

* ``~`` / ``~user`` → home expansion,
* a RELATIVE path → resolved against the project root (so a bare
  ``loom_code/agent.py`` still means the project file, the common
  case),
* an ABSOLUTE path → taken as-is (normalised).

It does NOT decide whether the path is allowed — that's the gate's
job. It only turns "what the model typed" into a concrete absolute
path for the gate to rule on and the tool to act on.
"""

from __future__ import annotations

from pathlib import Path


def resolve_path(path: str, project_root: Path | str) -> Path:
    """Resolve ``path`` to an absolute :class:`Path`.

    Relative paths anchor to ``project_root`` (bare names stay
    project files); ``~`` expands to home; absolute paths pass
    through. Always returns a resolved (symlink- and ``..``-collapsed)
    absolute path so the caller can permission-check the REAL target.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(project_root) / p
    return p.resolve()


def is_within(path: Path | str, root: Path | str) -> bool:
    """True if ``path`` is inside ``root`` (or equal). Both are
    resolved first, so symlinks / ``..`` can't fake containment."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False
