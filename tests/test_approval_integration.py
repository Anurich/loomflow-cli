"""End-to-end approval-gate wiring test.

Before loomflow 0.10.17, the destructive flag dropped between
``Tool.to_def()`` and ``ToolCall``, so loom-code's ``ApprovalGate``
(179 LOC, fully built) silently never fired — every write/edit/bash
auto-approved. 0.10.17 fixes the framework's propagation chain;
this test pins the loom-code-side wiring so a future edit to
``build_workers`` can't silently drop ``permissions=`` or
``approval_handler=`` and re-introduce the auto-approve regression.

Targets the ``coder`` worker: the coordinator is read-only, so the
destructive tools (and their approval gate) live on ``coder``.

Doesn't run a real model or a TTY — uses ``ScriptedModel`` to
emit a canned ``edit`` tool call and a recording handler that
captures whether the gate was consulted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from loomflow import ScriptedModel, ScriptedTurn
from loomflow.core.types import ToolCall

from loom_code.project import Project
from loom_code.workers import build_workers

pytestmark = pytest.mark.anyio


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project(
        root=tmp_path,
        is_git=False,
        context_file=None,
        context_text="",
    )


def _coder(project: Project, scripted: ScriptedModel, handler: object):
    # build_workers takes a precomputed ``auto_compact_at_tokens`` (we
    # leave it None) so it never calls context_window_for on the
    # scripted model — which has no str name to lower().
    return build_workers(
        project,
        model=scripted,  # type: ignore[arg-type]
        approval_handler=handler,  # type: ignore[arg-type]
    )["coder"]


async def test_coder_consults_approval_handler_on_edit(
    project: Project, tmp_path: Path
) -> None:
    """The coder routes a destructive ``edit`` through the provided
    ``approval_handler``. Recorder captures the call; returning True
    approves and the edit actually runs."""
    target = tmp_path / "foo.py"
    target.write_text("def hello():\n    return 'world'\n")

    handler_calls: list[ToolCall] = []

    async def recording_handler(
        call: ToolCall, user_id: str | None = None
    ) -> bool:
        handler_calls.append(call)
        return True

    scripted = ScriptedModel(
        turns=[
            ScriptedTurn(
                text="",
                tool_calls=[
                    ToolCall(
                        tool="edit",
                        args={
                            "path": "foo.py",
                            "old_string": "world",
                            "new_string": "loomflow",
                        },
                    ),
                ],
            ),
            ScriptedTurn(text="done"),
        ]
    )
    coder = _coder(project, scripted, recording_handler)

    await coder.run("edit foo.py")

    # Gate consulted exactly once with the right call.
    assert len(handler_calls) == 1, (
        f"approval handler called {len(handler_calls)} times "
        "— wiring regressed; loom-code is auto-approving"
    )
    assert handler_calls[0].tool == "edit"
    # Approve → edit happened.
    assert "loomflow" in target.read_text()


async def test_coder_denial_blocks_the_edit(
    project: Project, tmp_path: Path
) -> None:
    """Handler returning False MUST prevent the tool from running.
    The gate has teeth: if denial doesn't actually skip execution,
    the gate is decorative and we'd be back to auto-approve."""
    target = tmp_path / "foo.py"
    original = "def hello():\n    return 'world'\n"
    target.write_text(original)

    async def deny_handler(
        call: ToolCall, user_id: str | None = None
    ) -> bool:
        return False

    scripted = ScriptedModel(
        turns=[
            ScriptedTurn(
                text="",
                tool_calls=[
                    ToolCall(
                        tool="edit",
                        args={
                            "path": "foo.py",
                            "old_string": "world",
                            "new_string": "loomflow",
                        },
                    ),
                ],
            ),
            ScriptedTurn(text="done"),
        ]
    )
    coder = _coder(project, scripted, deny_handler)

    await coder.run("edit foo.py")
    assert target.read_text() == original, (
        "denied edit still ran — gate is decorative, not enforcing"
    )


async def test_read_only_call_bypasses_handler(
    project: Project, tmp_path: Path
) -> None:
    """Sanity check: ``read`` is non-destructive — the gate stays
    silent so the user doesn't see prompts for every read."""
    target = tmp_path / "foo.py"
    target.write_text("hello")

    handler_calls: list[ToolCall] = []

    async def recorder(
        call: ToolCall, user_id: str | None = None
    ) -> bool:
        handler_calls.append(call)
        return True

    scripted = ScriptedModel(
        turns=[
            ScriptedTurn(
                text="",
                tool_calls=[
                    ToolCall(tool="read", args={"path": "foo.py"}),
                ],
            ),
            ScriptedTurn(text="done"),
        ]
    )
    coder = _coder(project, scripted, recorder)

    await coder.run("read foo.py")
    assert handler_calls == [], (
        "handler called for non-destructive read — would spam the "
        "user with prompts for every read/grep/find/ls"
    )
