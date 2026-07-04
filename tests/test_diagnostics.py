"""Post-edit diagnostics (loom_code.diagnostics).

Contract: silence unless broken — findings appear only for real
per-file problems, never for healthy files, missing checkers, or
non-file tools; relative paths resolve against the PROJECT root."""

from __future__ import annotations

from pathlib import Path

import anyio
from loomflow.core.types import ToolCall, ToolResult

from loom_code import diagnostics as dg

# ---- detect_checker ---------------------------------------------------


def test_python_always_has_a_checker(tmp_path: Path) -> None:
    # ruff when installed, stdlib py_compile otherwise — either way,
    # .py files are never uncovered.
    argv = dg.detect_checker(tmp_path / "x.py")
    assert argv is not None


def test_unknown_extension_has_none(tmp_path: Path) -> None:
    assert dg.detect_checker(tmp_path / "x.xyz") is None
    assert dg.detect_checker(tmp_path / "Cargo.toml") is None


# ---- run_diagnostics ---------------------------------------------------


def test_healthy_python_file_is_silent(tmp_path: Path) -> None:
    f = tmp_path / "ok.py"
    f.write_text("x = 1\n")

    async def go() -> None:
        assert await dg.run_diagnostics(f) is None

    anyio.run(go)


def test_broken_python_file_reports(tmp_path: Path) -> None:
    f = tmp_path / "broken.py"
    f.write_text("def f(:\n")  # syntax error

    async def go() -> None:
        findings = await dg.run_diagnostics(f)
        assert findings is not None
        assert "broken.py" in findings

    anyio.run(go)


def test_missing_file_is_silent(tmp_path: Path) -> None:
    async def go() -> None:
        assert await dg.run_diagnostics(tmp_path / "gone.py") is None

    anyio.run(go)


# ---- the post-tool hook -------------------------------------------------


def _result(output: str = "edited ok") -> ToolResult:
    return ToolResult(call_id="t1", ok=True, output=output)


def test_hook_appends_findings_for_broken_edit(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def f(:\n")
    hook = dg.make_post_tool_hook(tmp_path)

    async def go() -> None:
        call = ToolCall(
            id="t1", tool="edit", args={"path": "broken.py"}
        )  # RELATIVE path — must resolve against root
        result = _result()
        await hook(call, result)
        assert "[diagnostics]" in result.output

    anyio.run(go)


def test_hook_silent_for_healthy_edit(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("x = 1\n")
    hook = dg.make_post_tool_hook(tmp_path)

    async def go() -> None:
        call = ToolCall(id="t1", tool="edit", args={"path": "ok.py"})
        result = _result()
        await hook(call, result)
        assert result.output == "edited ok"

    anyio.run(go)


def test_hook_ignores_non_edit_tools(tmp_path: Path) -> None:
    hook = dg.make_post_tool_hook(tmp_path)

    async def go() -> None:
        call = ToolCall(
            id="t1", tool="bash", args={"command": "ls broken.py"}
        )
        result = _result("listing")
        await hook(call, result)
        assert result.output == "listing"

    anyio.run(go)


def test_hook_skips_failed_edits(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def f(:\n")
    hook = dg.make_post_tool_hook(tmp_path)

    async def go() -> None:
        call = ToolCall(
            id="t1", tool="edit", args={"path": "broken.py"}
        )
        result = ToolResult(
            call_id="t1", ok=False, output="ERROR: no match"
        )
        await hook(call, result)
        assert result.output == "ERROR: no match"

    anyio.run(go)
