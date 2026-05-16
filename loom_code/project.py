"""Project detection — find the repo root + load context files.

A coding agent that doesn't know where the project starts and what
its conventions are wastes turns rediscovering them. This module
answers two questions cheaply, once, at startup:

1. **Where's the project root?** Walk up from cwd looking for a
   ``.git`` dir (fall back to cwd if none — loom-code still works
   on a loose folder of files).
2. **What are the project's conventions?** Read the first context
   file we find — ``CLAUDE.md`` / ``AGENTS.md`` / ``.loom/context.md``
   — and hand it to the system prompt so the agent starts already
   knowing the house rules.

Note: ``LOOM.md`` is intentionally NOT in the static-bake candidate
list. It's the loominit-generated codebase INDEX (large, sectioned,
covers the whole repo), and as of loominit slice 3 it gets per-turn
BM25 retrieval into the ``loom_index`` working block via
:class:`loom_code.loominit.injection.LoomRetriever`. Baking it
statically here would double-ship: once verbatim every turn, once
as retrieved sections. ``CLAUDE.md`` / ``AGENTS.md`` stay as
static bake because they encode "house rules" (small, every-turn
relevant) — a different role from the codebase index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Context-file names checked in priority order. CLAUDE.md first
# because it's the de-facto standard a lot of repos already have;
# AGENTS.md is the cross-tool convention; .loom/context.md is the
# loom-code-native opt-in for a dedicated house-rules file.
# LOOM.md is deliberately absent — see module docstring.
_CONTEXT_FILENAMES = (
    "CLAUDE.md",
    "AGENTS.md",
    ".loom/context.md",
)

# Cap the context file we inline into the system prompt — a
# runaway 50KB CLAUDE.md would blow the budget. Past this we
# truncate with a note; the agent can ``read`` the full file.
_MAX_CONTEXT_CHARS = 8_000


@dataclass(frozen=True, slots=True)
class Project:
    """Everything loom-code needs to know about where it's running."""

    root: Path
    """The project root — git toplevel, or cwd if not a git repo."""

    is_git: bool
    """True when ``root`` contains a ``.git`` directory."""

    context_file: Path | None
    """Path to the context file we found (CLAUDE.md etc.), or None."""

    context_text: str
    """The context file's body, truncated to a budget. Empty string
    when no context file exists."""


def detect_project(start: Path | str | None = None) -> Project:
    """Detect the project rooted at (or above) ``start`` (default cwd).

    Walks up looking for ``.git``; if found, that's the root and
    ``is_git`` is True. Otherwise the root is ``start`` itself —
    loom-code still runs, it just doesn't have a repo boundary.
    """
    cwd = Path(start).resolve() if start else Path.cwd().resolve()

    root = cwd
    is_git = False
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".git").exists():
            root = candidate
            is_git = True
            break

    context_file: Path | None = None
    context_text = ""
    for name in _CONTEXT_FILENAMES:
        fp = root / name
        if fp.is_file():
            context_file = fp
            raw = fp.read_text(errors="replace")
            if len(raw) > _MAX_CONTEXT_CHARS:
                context_text = (
                    raw[:_MAX_CONTEXT_CHARS]
                    + f"\n\n... [context file truncated at "
                    f"{_MAX_CONTEXT_CHARS} chars — use the `read` "
                    f"tool on {name} for the full text]"
                )
            else:
                context_text = raw
            break

    return Project(
        root=root,
        is_git=is_git,
        context_file=context_file,
        context_text=context_text,
    )
