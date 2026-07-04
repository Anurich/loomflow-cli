"""Doom-loop + environment-hint middleware (loom_code.loop_guard).

Terminal-Bench-backed behaviours locked down:
* 3rd+ edit to the same file in one turn → steering hint.
* Missing binary in bash output → PATH hint, once per binary.
* Same bash command run 4+ times in one turn → change-approach hint.
* Hints land APPENDED to the tool result string (post hooks run
  before the result is serialised, so the model sees them inline).
"""

from __future__ import annotations

import anyio
import pytest
from loomflow.core.types import ToolCall, ToolResult

from loom_code import loop_guard as lg


@pytest.fixture(autouse=True)
def _fresh():
    lg.reset()
    yield
    lg.reset()


# ---- edit repeats ----------------------------------------------------


def test_first_two_edits_are_silent() -> None:
    assert lg.hint_for("edit", {"path": "a.py"}, "") is None
    assert lg.hint_for("edit", {"path": "a.py"}, "") is None


def test_third_edit_same_file_fires() -> None:
    lg.hint_for("edit", {"path": "a.py"}, "")
    lg.hint_for("edit", {"path": "a.py"}, "")
    hint = lg.hint_for("edit", {"path": "a.py"}, "")
    assert hint and "edit #3" in hint and "a.py" in hint


def test_edit_counts_shared_across_edit_tools() -> None:
    # edit → write → multi_edit on one file is still a loop.
    lg.hint_for("edit", {"path": "a.py"}, "")
    lg.hint_for("write", {"path": "a.py"}, "")
    hint = lg.hint_for("multi_edit", {"path": "a.py"}, "")
    assert hint is not None


def test_different_files_do_not_cross_count() -> None:
    lg.hint_for("edit", {"path": "a.py"}, "")
    lg.hint_for("edit", {"path": "b.py"}, "")
    assert lg.hint_for("edit", {"path": "c.py"}, "") is None


def test_reset_clears_counters() -> None:
    for _ in range(3):
        lg.hint_for("edit", {"path": "a.py"}, "")
    lg.reset()
    assert lg.hint_for("edit", {"path": "a.py"}, "") is None


# ---- missing binary --------------------------------------------------


def test_bash_command_not_found_hints_path() -> None:
    hint = lg.hint_for(
        "bash",
        {"command": "pnpm install"},
        "bash: pnpm: command not found",
    )
    assert hint and "pnpm" in hint and "PATH" in hint


def test_windows_not_recognized_shape() -> None:
    hint = lg.hint_for(
        "bash",
        {"command": "choco -v"},
        "'choco' is not recognized as an internal or external command",
    )
    assert hint and "choco" in hint


def test_binary_hint_fires_once_per_turn() -> None:
    out = "bash: pnpm: command not found"
    assert lg.hint_for("bash", {"command": "pnpm i"}, out) is not None
    assert lg.hint_for("bash", {"command": "pnpm i -g"}, out) is None


def test_missing_path_arg_is_not_a_binary_hint() -> None:
    # errno text for a missing FILE hits similar patterns — a path
    # (contains /) must not trigger the PATH hint.
    hint = lg.hint_for(
        "bash",
        {"command": "cat /tmp/nope.txt"},
        "cat: /tmp/nope.txt: No such file or directory",
    )
    assert hint is None


# ---- repeated command ------------------------------------------------


def test_fourth_identical_command_fires() -> None:
    for _ in range(3):
        assert (
            lg.hint_for("bash", {"command": "pytest -q"}, "1 failed")
            is None
        )
    hint = lg.hint_for("bash", {"command": "pytest -q"}, "1 failed")
    assert hint and "4×" in hint


# ---- the post-tool hook mutates the live result ----------------------


def test_post_tool_appends_hint_to_string_output() -> None:
    async def go() -> None:
        call = ToolCall(
            id="t1", tool="bash", args={"command": "pnpm i"}
        )
        result = ToolResult(
            call_id="t1",
            ok=True,
            output="bash: pnpm: command not found",
        )
        await lg.post_tool(call, result)
        assert "[env]" in result.output
        assert result.output.startswith("bash: pnpm")  # original kept

    anyio.run(go)


def test_post_tool_leaves_non_string_output_alone() -> None:
    async def go() -> None:
        call = ToolCall(id="t1", tool="edit", args={"path": "a.py"})
        for _ in range(2):
            lg.hint_for("edit", {"path": "a.py"}, "")
        result = ToolResult(call_id="t1", ok=True, output={"k": 1})
        await lg.post_tool(call, result)
        assert result.output == {"k": 1}

    anyio.run(go)
