"""Post-edit diagnostics — surface syntax/lint errors to the model
immediately after every file mutation.

The MVP slice of "LSP-aware code intelligence" (named table-stakes for
2026 harnesses; Claude Code surfaces type errors after each edit, and
cline's LACK of it is a documented token-cost complaint): after each
``edit`` / ``multi_edit`` / ``write``, run a CHEAP per-file checker
and append any findings to the tool result, so the model fixes the
break in the same breath instead of discovering it three tool calls
later via a failing test.

Deliberately bounded — this is not an LSP client:

* per-FILE checks only (never whole-project ``tsc``/``cargo check``,
  which cost seconds-to-minutes per edit);
* only checkers that are actually PRESENT (never cause the #1 agent
  failure, "executable not found");
* hard timeout; on timeout/absence/success the hook is silent.

Rides the same native post-tool-hook mechanism as ``loop_guard`` —
the hint lands appended to the live ``ToolResult`` before the loop
serialises it, so it reaches the model inline.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import anyio

_EDIT_TOOLS = frozenset({"edit", "multi_edit", "write"})

_TIMEOUT_S = 4.0
_MAX_LINES = 12
_MAX_CHARS = 1200


def detect_checker(path: Path) -> list[str] | None:
    """The per-file checker argv for ``path``, or None.

    Preference order per language: the project's real linter when
    installed, else a stdlib/toolchain syntax check, else nothing.
    Every command here is single-file and sub-second."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        ruff = shutil.which("ruff")
        if ruff:
            # --no-cache: the agent edits fast; a stale cache dir in
            # odd cwds causes confusing misses. Still ~50ms/file.
            return [ruff, "check", "--no-cache", str(path)]
        return [sys.executable, "-m", "py_compile", str(path)]
    if suffix in (".js", ".mjs", ".cjs"):
        node = shutil.which("node")
        if node:
            return [node, "--check", str(path)]
        return None
    if suffix == ".go":
        gofmt = shutil.which("gofmt")
        if gofmt:
            # -e: report all (syntax) errors; -l alone is silent.
            return [gofmt, "-e", "-l", str(path)]
        return None
    if suffix in (".sh", ".bash"):
        bash = shutil.which("bash")
        if bash:
            return [bash, "-n", str(path)]
        return None
    return None


def _trim(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) > _MAX_LINES:
        lines = lines[:_MAX_LINES] + [
            f"… (+{len(lines) - _MAX_LINES} more lines)"
        ]
    out = "\n".join(lines)
    return out[:_MAX_CHARS]


async def run_diagnostics(path: Path) -> str | None:
    """Run the file's checker; return trimmed findings on FAILURE,
    None on success / no checker / timeout / any error. Silence is
    the contract — diagnostics may only ever add signal."""
    argv = detect_checker(path)
    # noqa rationale: one stat on a local path — cheaper than hopping
    # to a worker thread for it.
    if argv is None or not path.is_file():  # noqa: ASYNC240
        return None
    result: Any = None
    with anyio.move_on_after(_TIMEOUT_S):
        try:
            result = await anyio.run_process(argv, check=False)
        except Exception:  # noqa: BLE001 — silent by contract
            return None
    if result is None:  # timed out
        return None
    out = result.stdout.decode("utf-8", "replace")
    err = result.stderr.decode("utf-8", "replace")
    # gofmt -e -l prints the filename on stdout for UNFORMATTED files
    # even with rc=0 — only treat rc!=0 (real syntax/lint errors) as
    # a finding, matching "silence unless broken".
    if result.returncode == 0:
        return None
    findings = _trim(f"{out}\n{err}")
    return findings or None


def make_post_tool_hook(root: Path) -> Any:
    """Build the post-tool hook bound to ``root`` — edit tools take
    project-relative paths, so the hook must resolve them against the
    project root (mirroring ``paths.resolve_path``), not the process
    cwd."""

    async def post_tool(call: Any, result: Any) -> None:
        try:
            tool = str(getattr(call, "tool", ""))
            if tool not in _EDIT_TOOLS:
                return
            if not getattr(result, "ok", False):
                return  # the edit itself failed; don't pile on
            raw = str(
                dict(getattr(call, "args", {}) or {}).get("path", "")
            ).strip()
            if not raw:
                return
            # noqa rationale: pure string math, no disk I/O.
            p = Path(raw).expanduser()  # noqa: ASYNC240
            if not p.is_absolute():
                p = root / p
            findings = await run_diagnostics(p)
            if findings and isinstance(result.output, str):
                result.output = (
                    f"{result.output}\n\n[diagnostics] the edited "
                    "file now has problems — fix them before moving "
                    f"on:\n{findings}"
                )
        except Exception:  # noqa: BLE001 — never break a tool result
            return

    return post_tool


def attach(agent: Any, root: Path) -> None:
    """Register on ``agent``'s hook registry (same shape as
    loop_guard.attach). Best-effort."""
    try:
        agent._hooks.register_post_tool(  # noqa: SLF001
            make_post_tool_hook(root)
        )
    except Exception:  # noqa: BLE001
        return
