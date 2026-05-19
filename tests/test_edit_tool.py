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

from loom_code.edit_tool import verifying_edit_tool


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
