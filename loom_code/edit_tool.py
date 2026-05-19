"""Wrapped ``edit`` tool — surfaces post-edit file state so the
agent can self-correct when its replacement was malformed.

Why this exists: loomflow's stock ``edit_tool`` returns just
``"edited X bytes (Y → Z)"``. The agent has no visibility into
what the file ACTUALLY looks like after the edit, so when the
model writes a malformed ``new_string`` (e.g. accidentally
preserving the old code AND inserting new code, observed in a
real REPL session on ``silently_swallow`` → got two ``except``
blocks coexisting), the agent assumes success and moves on.

This wrapper forwards every call to the underlying edit_tool and
then appends an EDIT PREVIEW window — the edited region plus
±10 lines of surrounding context — to the tool result. Next-turn
the agent sees what the file looks like and can issue a
correcting edit instead of declaring victory.

Also runs an ``ast.parse`` check for ``.py`` files and surfaces
syntax errors as a clear warning (loomflow's edit_tool doesn't
validate; you can edit a Python file into a syntactically broken
state silently).
"""

from __future__ import annotations

import ast
from pathlib import Path

from loomflow import tool
from loomflow.tools import edit_tool as _loomflow_edit_tool
from loomflow.tools.registry import Tool

# How many lines of context to show around the edited region in
# the EDIT PREVIEW. Chosen so the model can usually see the whole
# function being edited; bigger windows just bloat the tool result.
_CONTEXT_LINES = 10


def _find_edit_region(
    old_text: str, new_text: str
) -> tuple[int, int]:
    """Return ``(start_line, end_line)`` (1-indexed, inclusive)
    of the region in ``new_text`` that differs from ``old_text``.

    Crude line-diff: walk both line lists in parallel, find the
    first divergence, then walk backward from the ends to find
    the last divergence. Good enough for the typical edit
    (single contiguous change). For multi-region edits this just
    picks the bounding window, which is the right thing to show
    anyway.
    """
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()

    # Find first divergence.
    start = 0
    while (
        start < len(old_lines)
        and start < len(new_lines)
        and old_lines[start] == new_lines[start]
    ):
        start += 1

    # Find last divergence — walk from the end inward.
    o_end = len(old_lines) - 1
    n_end = len(new_lines) - 1
    while (
        o_end >= start
        and n_end >= start
        and old_lines[o_end] == new_lines[n_end]
    ):
        o_end -= 1
        n_end -= 1
    return (start + 1, n_end + 1)


def _render_preview(
    file_path: Path, new_text: str, start_line: int, end_line: int
) -> str:
    """Render the edited region + ±N lines of context with line
    numbers. A ▸ marker on each line in the edit window so the
    agent can quickly see what changed."""
    lines = new_text.splitlines()
    if not lines:
        return ""
    show_start = max(1, start_line - _CONTEXT_LINES)
    show_end = min(len(lines), end_line + _CONTEXT_LINES)
    out: list[str] = [
        f"--- EDIT PREVIEW: {file_path.name} "
        f"(lines {show_start}-{show_end} of {len(lines)}) ---"
    ]
    for i in range(show_start, show_end + 1):
        line = lines[i - 1].rstrip()
        marker = "▸ " if start_line <= i <= end_line else "  "
        out.append(f"{marker}{i:4d} │ {line}")
    out.append("--- END EDIT PREVIEW ---")
    return "\n".join(out)


def _syntax_check_python(path: Path, text: str) -> str | None:
    """Return a warning string if ``text`` is invalid Python, else
    None. Only fires for ``.py`` files — silent for everything
    else.

    We don't BLOCK the edit on syntax failure because the agent
    sometimes intentionally edits a file into a transient broken
    state (e.g. mid-refactor), but we surface the error so it
    notices on this turn instead of debugging in 5 turns."""
    if path.suffix != ".py":
        return None
    try:
        ast.parse(text)
        return None
    except SyntaxError as exc:
        return (
            f"⚠ WARNING: file is no longer syntactically valid "
            f"Python after this edit. {type(exc).__name__}: "
            f"{exc.msg} (line {exc.lineno or '?'}). The edit was "
            "applied; consider correcting it on the next turn."
        )


def verifying_edit_tool(workdir: Path | str) -> Tool:
    """Build the loom-code edit tool.

    Same signature as loomflow's edit_tool:
        ``edit(path, old_string, new_string, replace_all=False)``

    Differences in the tool result:
        - On success: appends an ``EDIT PREVIEW`` block showing the
          edited region + ±10 lines of context with line numbers.
        - On Python syntax-break: appends a ``⚠ WARNING`` line so
          the agent immediately knows the file is broken.
        - On failure (file not found, no match, etc.): returns the
          underlying error verbatim. No preview, no extra noise.
    """
    root = Path(workdir).resolve()
    inner = _loomflow_edit_tool(workdir=root)

    async def edit(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Find-and-replace inside an existing file; returns the
        edit summary PLUS a preview of the file after the edit
        so the agent can self-correct malformed replacements.
        See module docstring for the full contract."""
        # Snapshot the file's pre-edit content for the diff bounds.
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return (
                f"edit: refusing to edit outside workdir: {target}"
            )
        if not target.is_file():
            # Defer to loomflow's error message for consistency.
            return await inner.fn(
                path=path,
                old_string=old_string,
                new_string=new_string,
                replace_all=replace_all,
            )
        old_text = target.read_text(
            encoding="utf-8", errors="replace"
        )

        # Delegate the actual replace + write.
        result = await inner.fn(
            path=path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )
        # loomflow's edit_tool signals errors via leading "ERROR:".
        # On error, return verbatim — no preview when the edit
        # didn't change anything.
        if str(result).startswith("ERROR"):
            return result

        # Read post-edit content + build preview.
        new_text = target.read_text(
            encoding="utf-8", errors="replace"
        )
        start_line, end_line = _find_edit_region(old_text, new_text)
        preview = _render_preview(
            target, new_text, start_line, end_line
        )
        warn = _syntax_check_python(target, new_text)
        parts: list[str] = [result, "", preview]
        if warn:
            parts.append("")
            parts.append(warn)
        return "\n".join(parts)

    return tool(
        name="edit",
        description=(
            "Find-and-replace inside an existing file. Same as "
            "loomflow's edit but the tool result includes an "
            "EDIT PREVIEW window showing the edited region + ±10 "
            "lines of context AFTER the edit, so you can verify "
            "the replacement looks right and self-correct on the "
            "next turn if it doesn't. Also warns when an edit "
            "leaves a .py file syntactically invalid. Args: path "
            "(relative), old_string (must match exactly), "
            "new_string, replace_all=False."
        ),
        # ``destructive=True`` matches loomflow's edit_tool default
        # so the ``ApprovalGate`` continues to fire before the
        # write. Without this, every edit auto-approves even when
        # the user opted into the gate — silently weakens the
        # destructive-action safety contract (caught by
        # tests/test_approval_integration.py).
        destructive=True,
    )(edit)
