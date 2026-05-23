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
import json
from pathlib import Path
from typing import Any

from loomflow import tool
from loomflow.tools import edit_tool as _loomflow_edit_tool
from loomflow.tools.registry import Tool

from .grep_tool import _as_bool

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
        # Coerce ``replace_all`` — the tool-call layer may send the
        # STRING "true"/"false" instead of a bool, and a non-empty
        # "false" string is truthy, which would silently replace
        # ALL occurrences when the model meant just one.
        replace_all = _as_bool(replace_all, default=False)
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


def _loads_lenient(text: str) -> Any:
    """Parse a model-serialised string into a Python object, tolerating
    BOTH JSON (double quotes) and Python-repr (single quotes) — weak
    models emit either, and ``json.loads`` rejects the single-quote
    form. Raises ``ValueError`` when neither parses."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(str(exc)) from exc


def _coerce_edits(value: Any) -> list[dict[str, str]] | str:
    """Coerce the model's serialisation of the ``edits`` list into
    a native list of ``{old_string, new_string}`` dicts. Returns
    the list, or an error string the tool returns verbatim.

    Weak models serialise list-of-objects args inconsistently — a
    JSON string, a list of JSON strings, a Python-repr (single-quote)
    string, a dict with an ``edits`` key — so we salvage every shape,
    same lenient approach ``plan_write`` uses for its ``steps`` arg.
    """
    if isinstance(value, str):
        try:
            value = _loads_lenient(value)
        except ValueError:
            return (
                "ERROR: `edits` must be a list of "
                "{old_string, new_string} objects (or a JSON "
                "string of one). Couldn't parse the string given."
            )
    if isinstance(value, dict):
        # ``{"edits": [...]}`` wrapper, or a single edit dict.
        value = value.get("edits", [value])
    if not isinstance(value, list):
        return (
            "ERROR: `edits` must be a list of "
            f"{{old_string, new_string}} objects. Got "
            f"{type(value).__name__}."
        )
    out: list[dict[str, str]] = []
    for i, item in enumerate(value):
        if isinstance(item, str):
            try:
                item = _loads_lenient(item)
            except ValueError:
                return (
                    f"ERROR: edit #{i + 1} is a string that isn't "
                    "valid JSON. Each edit must be an object with "
                    "`old_string` and `new_string`."
                )
        if not isinstance(item, dict):
            return (
                f"ERROR: edit #{i + 1} must be an object with "
                f"`old_string` + `new_string`, got "
                f"{type(item).__name__}."
            )
        if "old_string" not in item or "new_string" not in item:
            return (
                f"ERROR: edit #{i + 1} is missing `old_string` "
                "and/or `new_string`."
            )
        coerced_edit: dict[str, str] = {
            "old_string": str(item["old_string"]),
            "new_string": str(item["new_string"]),
        }
        # Preserve the optional per-edit ``replace_all`` flag (the
        # multi_edit applier coerces it to bool). Without carrying
        # it through, every edit defaults to single-replace.
        if "replace_all" in item:
            coerced_edit["replace_all"] = str(item["replace_all"])
        out.append(coerced_edit)
    if not out:
        return "ERROR: `edits` was empty — nothing to do."
    return out


def multi_edit_tool(workdir: Path | str) -> Tool:
    """Build the loom-code ``multi_edit`` tool — apply MANY edits to
    ONE file in a single ATOMIC call.

    Why it exists: making N separate ``edit`` calls to fix N things
    in one file is N round-trips, and a model that produces a
    slightly-off ``old_string`` retries the same edit repeatedly
    (observed: ~8 retries for one fix, burning a whole session's
    tokens). ``multi_edit`` collapses N changes into one call AND
    is region-targeted — it never reproduces unchanged code, so it
    scales to arbitrarily large files without the token blow-up (or
    the "# ... rest unchanged ..." laziness) of a whole-file
    rewrite.

    **Atomic**: every edit's ``old_string`` must match (exactly
    once, unless that edit sets ``replace_all``). If ANY edit fails
    to match, NOTHING is written and the tool reports which edit
    failed — so the file is never left half-changed / corrupted.
    The model fixes the offending edit and resubmits the batch.

    Model-facing signature:
        ``multi_edit(path, edits=[{old_string, new_string,
        replace_all?}, ...])``

    On success the result carries the same EDIT PREVIEW + Python
    syntax-break warning as ``edit``, so the model can verify the
    whole batch landed correctly in one look.
    """
    root = Path(workdir).resolve()

    async def multi_edit(path: str, edits: Any) -> str:
        """Apply a batch of find-and-replace edits to one file
        atomically. See the module/tool docstring for the full
        contract."""
        coerced = _coerce_edits(edits)
        if isinstance(coerced, str):
            return coerced  # error message, verbatim

        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return (
                f"multi_edit: refusing to edit outside workdir: "
                f"{target}"
            )
        if not target.is_file():
            return f"multi_edit: file not found: {path}"

        original = target.read_text(encoding="utf-8", errors="replace")
        working = original
        # Apply each edit to the in-memory working copy. Validate
        # match BEFORE mutating so a mid-batch failure leaves
        # ``working`` partially applied but we NEVER write it.
        for i, e in enumerate(coerced):
            old = e["old_string"]
            new = e["new_string"]
            replace_all = _as_bool(
                e.get("replace_all"), default=False
            )
            count = working.count(old)
            if count == 0:
                return (
                    f"ERROR: edit #{i + 1} old_string not found in "
                    f"{path} (after applying edits 1..{i}). It must "
                    "match EXACTLY (whitespace, indentation, line "
                    "breaks). NOTHING was written — fix this edit "
                    "and resubmit the whole batch."
                )
            if count > 1 and not replace_all:
                return (
                    f"ERROR: edit #{i + 1} old_string appears "
                    f"{count} times in {path}; add more surrounding "
                    "context to make it unique, or set "
                    "replace_all=true on that edit. NOTHING was "
                    "written."
                )
            working = (
                working.replace(old, new)
                if replace_all
                else working.replace(old, new, 1)
            )

        # All edits matched — write once.
        target.write_text(working, encoding="utf-8")

        start_line, end_line = _find_edit_region(original, working)
        preview = _render_preview(
            target, working, start_line, end_line
        )
        warn = _syntax_check_python(target, working)
        header = (
            f"multi_edit: ✓ applied {len(coerced)} edit"
            f"{'s' if len(coerced) != 1 else ''} to {path} "
            f"({len(original)} → {len(working)} bytes)"
        )
        parts = [header, "", preview]
        if warn:
            parts.append("")
            parts.append(warn)
        return "\n".join(parts)

    return tool(
        name="multi_edit",
        description=(
            "Apply MULTIPLE find-and-replace edits to ONE file in a "
            "single ATOMIC call. Prefer this over repeated `edit` "
            "calls when changing several things in the same file — "
            "one round-trip, and it scales to large files (only "
            "touches the changed regions, never rewrites the whole "
            "file). All edits must match or NONE apply (no half-"
            "edited file). Args: path, edits=[{old_string, "
            "new_string, replace_all?}, ...]. Result includes an "
            "EDIT PREVIEW + a syntax-break warning for .py files."
        ),
        destructive=True,
    )(multi_edit)
