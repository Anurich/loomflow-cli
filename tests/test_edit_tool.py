"""Tests for ``verifying_edit_tool`` — the loom-code edit wrapper.

Four things this file guards:

1. **Happy path appends an EDIT PREVIEW** — every successful edit
   adds a windowed view of the file post-edit so the agent can
   self-correct on the next turn instead of declaring victory.
2. **Python syntax-break surfaces a warning** — when an edit
   leaves a ``.py`` file syntactically broken the agent sees the
   warning immediately and can correct, rather than discovering
   the issue 5 turns later.
3. **No-match passes through loomflow's ERROR verbatim** — we
   don't paper over the underlying tool's failure path.
4. **Path traversal is refused** — same safety contract as the
   underlying tool.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loom_code.edit_tool import multi_edit_tool, verifying_edit_tool


def test_happy_path_appends_edit_preview(tmp_path: Path) -> None:
    """A successful edit returns loomflow's summary PLUS an
    EDIT PREVIEW block with line numbers and a ▸ marker on the
    edited line so the agent can verify the change visually."""
    f = tmp_path / "sample.py"
    f.write_text("def hello():\n    return 1\n")
    tool = verifying_edit_tool(tmp_path)

    result = asyncio.run(
        tool.fn(
            path="sample.py",
            old_string="return 1",
            new_string="return 99",
        )
    )
    # Underlying summary preserved.
    assert "edited sample.py" in result
    # Preview block + line markers present.
    assert "EDIT PREVIEW" in result
    assert "▸" in result
    assert "return 99" in result
    # END marker so the agent knows where the preview stops.
    assert "END EDIT PREVIEW" in result


def test_syntax_break_warns_but_still_applies(
    tmp_path: Path,
) -> None:
    """A Python edit that leaves the file invalid: edit IS
    applied (sometimes intentional mid-refactor) but a clear
    warning is appended so the agent immediately sees the
    problem and can correct on the next turn."""
    f = tmp_path / "sample.py"
    f.write_text("def hello():\n    return 1\n")
    tool = verifying_edit_tool(tmp_path)

    result = asyncio.run(
        tool.fn(
            path="sample.py",
            old_string="def hello():\n    return 1",
            new_string="def hello( :: invalid",
        )
    )
    assert "edited sample.py" in result
    assert "WARNING" in result
    assert "syntactically valid Python" in result
    # The bad content was actually written (mid-refactor pattern).
    assert "def hello( :: invalid" in f.read_text()


def test_no_warning_on_non_python_files(tmp_path: Path) -> None:
    """``ast.parse`` only fires for ``.py`` files; editing a
    Markdown or text file into 'invalid Python' is fine."""
    f = tmp_path / "notes.md"
    f.write_text("# header\nbody\n")
    tool = verifying_edit_tool(tmp_path)

    result = asyncio.run(
        tool.fn(
            path="notes.md",
            old_string="body",
            new_string="def hello( :: invalid",  # would break .py
        )
    )
    assert "WARNING" not in result


def test_no_match_passes_through_loomflow_error(
    tmp_path: Path,
) -> None:
    """Loomflow's edit_tool returns 'ERROR: old_string not found'
    on mismatch. We surface that verbatim — no preview, no
    interpretation, no swallowing."""
    f = tmp_path / "sample.py"
    f.write_text("def hello():\n    return 1\n")
    tool = verifying_edit_tool(tmp_path)

    result = asyncio.run(
        tool.fn(
            path="sample.py",
            old_string="nothing like this exists",
            new_string="whatever",
        )
    )
    assert result.startswith("ERROR")
    assert "EDIT PREVIEW" not in result
    # File is unchanged.
    assert f.read_text() == "def hello():\n    return 1\n"


def test_path_traversal_is_refused(tmp_path: Path) -> None:
    """Refuse traversal attempts to escape the workdir — agents
    shouldn't be able to edit ../../.."""
    (tmp_path / "sample.py").write_text("x\n")
    tool = verifying_edit_tool(tmp_path)

    result = asyncio.run(
        tool.fn(
            path="../../../etc/hosts",
            old_string="anything",
            new_string="whatever",
        )
    )
    assert "refusing" in result.lower() or "not found" in result.lower()


def test_preview_window_shows_context(tmp_path: Path) -> None:
    """For a small edit in a larger file the preview window
    shows the edited region + ~10 lines around it so the agent
    can see what the surrounding code looks like."""
    f = tmp_path / "big.py"
    f.write_text("\n".join(f"line_{i}" for i in range(1, 31)) + "\n")
    tool = verifying_edit_tool(tmp_path)

    result = asyncio.run(
        tool.fn(
            path="big.py",
            old_string="line_15",
            new_string="LINE_FIFTEEN",
        )
    )
    # The edit on line 15 should show lines roughly 5-25.
    assert "LINE_FIFTEEN" in result
    assert "line_10" in result  # context BEFORE the edit
    assert "line_20" in result  # context AFTER the edit
    # Lines way outside the context window are absent.
    assert "line_1\n" not in result
    assert "line_30" not in result


def test_tool_name_is_edit(tmp_path: Path) -> None:
    """The tool registers as ``edit`` so it cleanly replaces
    loomflow's ``edit_tool`` slot in the agent's tool surface."""
    tool = verifying_edit_tool(tmp_path)
    assert tool.name == "edit"


def test_replace_all_string_true_coerced(tmp_path: Path) -> None:
    """The tool-call layer can send replace_all='true' (string).
    A non-empty 'false' string is truthy, so without coercion a
    model meaning replace_all=False would silently replace ALL
    occurrences. Coercion must parse the string form."""
    f = tmp_path / "f.py"
    f.write_text("x = 1\nx = 1\nx = 1\n")
    tool = verifying_edit_tool(tmp_path)

    # replace_all="false" (string) → must replace only ONE.
    result = asyncio.run(
        tool.fn(
            path="f.py",
            old_string="x = 1",
            new_string="x = 2",
            replace_all="false",
        )
    )
    # loomflow's edit errors on multi-match without replace_all —
    # so a correctly-coerced "false" yields the multi-match error,
    # NOT a silent replace-all.
    assert "ERROR" in result and "appears" in result
    # File unchanged (the edit was rejected for ambiguity).
    assert f.read_text().count("x = 1") == 3


def test_replace_all_string_true_replaces_all(tmp_path: Path) -> None:
    """replace_all='true' (string) → coerced to True → all
    occurrences replaced."""
    f = tmp_path / "f.py"
    f.write_text("x = 1\nx = 1\nx = 1\n")
    tool = verifying_edit_tool(tmp_path)

    result = asyncio.run(
        tool.fn(
            path="f.py",
            old_string="x = 1",
            new_string="x = 2",
            replace_all="true",
        )
    )
    assert "ERROR" not in result
    assert f.read_text().count("x = 2") == 3


# ---- multi_edit: atomic batch edits to one file ---------------------


def test_multi_edit_applies_all_atomically(tmp_path: Path) -> None:
    """N edits in one call → all applied, single write, EDIT
    PREVIEW returned. The headline "all at once" win."""
    f = tmp_path / "m.py"
    f.write_text("a = 1\nb = 2\nc = 3\n")
    tool = multi_edit_tool(tmp_path)

    result = asyncio.run(
        tool.fn(
            path="m.py",
            edits=[
                {"old_string": "a = 1", "new_string": "a = 100"},
                {"old_string": "b = 2", "new_string": "b = 200"},
                {"old_string": "c = 3", "new_string": "c = 300"},
            ],
        )
    )
    assert "applied 3 edits" in result
    assert "EDIT PREVIEW" in result
    assert f.read_text() == "a = 100\nb = 200\nc = 300\n"


def test_multi_edit_atomic_failure_writes_nothing(
    tmp_path: Path,
) -> None:
    """If ANY edit's old_string doesn't match, NOTHING is written
    — the file is never left half-edited. The corruption guard."""
    f = tmp_path / "m.py"
    f.write_text("x = 1\ny = 2\n")
    tool = multi_edit_tool(tmp_path)

    result = asyncio.run(
        tool.fn(
            path="m.py",
            edits=[
                {"old_string": "x = 1", "new_string": "x = 9"},
                {"old_string": "NONEXISTENT", "new_string": "z"},
            ],
        )
    )
    assert result.startswith("ERROR")
    assert "edit #2" in result
    assert "NOTHING was written" in result
    # File completely unchanged — edit #1 was NOT applied either.
    assert f.read_text() == "x = 1\ny = 2\n"


def test_multi_edit_syntax_break_warns(tmp_path: Path) -> None:
    """A batch that leaves the .py file invalid still applies (the
    edits all matched) but warns — same contract as single edit."""
    f = tmp_path / "m.py"
    f.write_text("def f():\n    return 1\n")
    tool = multi_edit_tool(tmp_path)

    result = asyncio.run(
        tool.fn(
            path="m.py",
            edits=[
                {"old_string": "def f():", "new_string": "def f( :: broken"},
            ],
        )
    )
    assert "applied 1 edit" in result
    assert "WARNING" in result
    assert "syntactically valid Python" in result


def test_multi_edit_json_string_edits_arg(tmp_path: Path) -> None:
    """The tool-call layer often serialises the edits list as a
    JSON STRING. Must be coerced, not rejected."""
    f = tmp_path / "m.py"
    f.write_text("p = 1\n")
    tool = multi_edit_tool(tmp_path)

    result = asyncio.run(
        tool.fn(
            path="m.py",
            edits='[{"old_string": "p = 1", "new_string": "p = 2"}]',
        )
    )
    assert "applied 1 edit" in result
    assert f.read_text() == "p = 2\n"


def test_multi_edit_replace_all_per_edit(tmp_path: Path) -> None:
    """A single edit in the batch can set replace_all to hit every
    occurrence; without it, a multi-match edit fails the batch."""
    f = tmp_path / "m.py"
    f.write_text("v = 0\nv = 0\nv = 0\n")
    tool = multi_edit_tool(tmp_path)

    # Without replace_all → ambiguous → whole batch rejected.
    result = asyncio.run(
        tool.fn(
            path="m.py",
            edits=[{"old_string": "v = 0", "new_string": "v = 1"}],
        )
    )
    assert result.startswith("ERROR")
    assert "appears 3 times" in result
    assert f.read_text() == "v = 0\nv = 0\nv = 0\n"  # untouched

    # With replace_all=true → all three replaced.
    result = asyncio.run(
        tool.fn(
            path="m.py",
            edits=[
                {
                    "old_string": "v = 0",
                    "new_string": "v = 1",
                    "replace_all": "true",
                }
            ],
        )
    )
    assert "applied 1 edit" in result
    assert f.read_text().count("v = 1") == 3


def test_multi_edit_empty_edits_rejected(tmp_path: Path) -> None:
    f = tmp_path / "m.py"
    f.write_text("x\n")
    tool = multi_edit_tool(tmp_path)
    result = asyncio.run(tool.fn(path="m.py", edits=[]))
    assert result.startswith("ERROR")
    assert "empty" in result.lower()


def test_multi_edit_path_traversal_refused(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("x\n")
    tool = multi_edit_tool(tmp_path)
    result = asyncio.run(
        tool.fn(
            path="../../../etc/hosts",
            edits=[{"old_string": "a", "new_string": "b"}],
        )
    )
    assert "refusing" in result.lower() or "not found" in result.lower()


def test_multi_edit_tool_name_and_destructive(tmp_path: Path) -> None:
    tool = multi_edit_tool(tmp_path)
    assert tool.name == "multi_edit"
    assert tool.destructive is True
