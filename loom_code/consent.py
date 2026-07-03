"""Session registry of files the USER explicitly referenced.

The consent model in one line: **any path the user types or pastes IS
the permission to TARGET that file** — bare or ``@``-mentioned — but
not the permission to skip confirmation. A referenced file may be
``edit``/``multi_edit``ed even outside the project root; the
ApprovalGate STILL forces a diff preview + confirm for every
outside-project edit, in EVERY mode (accept-edits / yolo / allow-all /
--yes), so no out-of-tree file is ever mutated without a human seeing
the change. That gate is why granting on a bare paste is safe: a path
that merely appears in a pasted stack trace becomes a candidate, but
the user still sees + rejects the diff prompt before anything is
written — it is never silently mutated.

Module-level on purpose: the REPL registers paths as it expands
mentions, and the edit tools — built once at agent construction,
long before any mention exists — consult it lazily per call. A
plain set beats threading a handle through six build functions.
Lifetime is the process (one REPL session); nothing persists.
"""

from __future__ import annotations

from pathlib import Path

_granted: set[Path] = set()


def grant(path: Path | str) -> None:
    """Record that the user explicitly referenced ``path``."""
    try:
        _granted.add(Path(path).expanduser().resolve())
    except OSError:
        pass


def is_granted(path: Path | str) -> bool:
    """True if the user referenced exactly this file this session."""
    try:
        return Path(path).expanduser().resolve() in _granted
    except OSError:
        return False


def reset() -> None:
    """Drop all grants (used by /clear and tests)."""
    _granted.clear()
