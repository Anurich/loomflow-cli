"""EXECUTION sweep of every slash command.

``test_repl_guards.test_every_defined_command_is_dispatched`` proves
each command has a dispatch branch; this file proves each branch
actually RUNS: every command in ``_COMMAND_DEFS`` is executed against
a real ``Repl`` (echo model, temp git project) and must complete
without raising, without printing "unknown command", and without
leaking a traceback to the user.

History: ``/checkpoints`` and ``/undo`` shipped advertised-but-unwired
(user-reported), and nothing exercised the other 29 handlers at all.

Interactive prompts are stubbed to CANCEL (``_select_menu`` /
``_prompt_line`` return ``None``) so menu-driven commands
(/set_model, /set_web, /resume, …) take their cancel path instead of
blocking on stdin.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from loom_code import render
from loom_code.project import Project
from loom_code.repl import _COMMAND_DEFS, Repl

pytestmark = pytest.mark.anyio


# Arguments that make each command exercise a REAL path (not just its
# usage message) where that's cheap and side-effect-safe. Commands not
# listed run bare.
_ARGS: dict[str, str] = {
    "/mode": "default",
    "/effort": "low",
    "/verify": "on",
    "/model": "echo",  # switch to the model we're already on
    "/set_continue_cap": "5",
    "/compress_token_length": "50000",
}

# Commands whose contract is to END the session (return False).
_EXITING = {"/exit"}


@pytest.fixture
def repl(tmp_path: Path) -> Any:
    """A real Repl on the echo model in a temp GIT project (git so the
    checkpoint/isolate family exercises its real path), interactive
    prompts stubbed to cancel, console captured."""
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True,
        capture_output=True,
    )
    (tmp_path / "main.py").write_text("print('hello')\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "-A"], check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "init"],
        check=True, capture_output=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
             "PATH": "/usr/bin:/bin"},
    )
    project = Project(
        root=tmp_path, is_git=True, context_file=None, context_text=""
    )
    r = Repl(project, "echo", use_tui=False)

    async def _cancel_menu(*a: Any, **k: Any) -> None:
        return None

    async def _cancel_line(*a: Any, **k: Any) -> None:
        return None

    r._select_menu = _cancel_menu  # type: ignore[method-assign]
    r._prompt_line = _cancel_line  # type: ignore[method-assign]
    return r


@pytest.fixture
def captured() -> io.StringIO:
    """Route the module-level console into a buffer for assertions."""
    buf = io.StringIO()
    render.set_console_target(Console(file=buf, width=100))
    yield buf
    render.reset_console_target()


@pytest.mark.parametrize(
    "cmd", [c for c, _d, _g in _COMMAND_DEFS] + ["/quit"]
)
async def test_command_executes_cleanly(
    repl: Any, captured: io.StringIO, cmd: str
) -> None:
    line = f"{cmd} {_ARGS.get(cmd, '')}".strip()
    result = await repl._handle_slash(line)

    out = captured.getvalue()
    assert "unknown command" not in out, f"{cmd} not dispatched:\n{out}"
    assert "Traceback" not in out, f"{cmd} leaked a traceback:\n{out}"
    if cmd in _EXITING or cmd == "/quit":
        assert result is False, f"{cmd} should end the session"
    else:
        assert result is True, f"{cmd} should keep the session alive"


async def test_isolate_review_discard_cycle(
    repl: Any, captured: io.StringIO
) -> None:
    """The worktree family as a REAL sequence: isolate → review →
    discard. (Bare parametrized runs only prove each is callable;
    the value is the cycle.)"""
    assert await repl._handle_slash("/isolate") is True
    assert await repl._handle_slash("/review") is True
    assert await repl._handle_slash("/discard") is True
    out = captured.getvalue()
    assert "Traceback" not in out


async def test_checkpoint_undo_cycle(
    repl: Any, captured: io.StringIO
) -> None:
    """Take a real checkpoint, mutate the tree, /undo restores it —
    the user-reported pair, as a behavioral test."""
    from loom_code import checkpoint as cp

    main = repl.project.root / "main.py"
    seq, err = cp.checkpoint(
        repl.project.root, summary="before test edit"
    )
    assert seq is not None, f"checkpoint failed: {err}"
    main.write_text("print('MUTATED')\n")
    assert await repl._handle_slash("/checkpoints") is True
    assert await repl._handle_slash("/undo") is True
    out = captured.getvalue()
    # Output FIRST: on failure this shows the actual /undo error
    # (git stderr etc.) instead of a bare content mismatch.
    assert "restored" in out, f"/undo did not restore. Output:\n{out}"
    assert "Traceback" not in out
    assert main.read_text() == "print('hello')\n", (
        f"content not reverted. Output:\n{out}"
    )
