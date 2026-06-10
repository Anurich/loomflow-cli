"""Active recall — ``_inject_learned_notes`` pushes proven notes into
the ``learned_notes`` working block before each turn.

The method only touches ``self.workspace`` and ``self.agent.memory``,
so it's exercised unbound with a stub ``self`` — no full Repl build.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from loomflow import InMemoryMemory
from loomflow.workspace import InMemoryWorkspace

from loom_code.repl import _USER_ID, Repl

pytestmark = pytest.mark.anyio


async def _block_text(memory: InMemoryMemory, name: str) -> str:
    blocks = await memory.working(user_id=_USER_ID)
    for b in blocks:
        if b.name == name:
            return b.content
    return ""


def _fake_repl(workspace: InMemoryWorkspace, memory: InMemoryMemory):
    return SimpleNamespace(
        workspace=workspace,
        agent=SimpleNamespace(memory=memory),
    )


async def _note(
    ws: InMemoryWorkspace, title: str, body: str, *, successes: int = 0
) -> str:
    note = await ws.write_note(
        author="coder", title=title, body=body, user_id=_USER_ID
    )
    for _ in range(successes):
        await ws.attribute_outcome(
            success=True, slugs=[note.slug], user_id=_USER_ID
        )
    return note.slug


async def test_proven_notes_land_in_block(tmp_path: Path) -> None:
    ws = InMemoryWorkspace()
    mem = InMemoryMemory()
    slug = await _note(
        ws,
        "auth retry fix",
        "Auth retries must use the shared backoff helper.",
        successes=2,
    )
    fake = _fake_repl(ws, mem)

    await Repl._inject_learned_notes(fake, "fix the auth retry bug")

    block = await _block_text(mem, "learned_notes")
    assert slug in block
    assert "worked 2x" in block
    # The header tells the agent how to keep the credit chain alive.
    assert "read_note" in block


async def test_uncredited_notes_are_excluded(tmp_path: Path) -> None:
    """A note that was never cited in an accepted turn is NOT injected
    — it stays discoverable via search, but doesn't get prompt space."""
    ws = InMemoryWorkspace()
    mem = InMemoryMemory()
    slug = await _note(
        ws,
        "auth unproven theory",
        "Auth might be broken because of cosmic rays.",
        successes=0,
    )
    fake = _fake_repl(ws, mem)

    await Repl._inject_learned_notes(fake, "fix the auth retry bug")

    block = await _block_text(mem, "learned_notes")
    assert slug not in block


async def test_block_cleared_when_nothing_relevant(
    tmp_path: Path,
) -> None:
    """Stale advice must not linger: when no proven note matches, the
    block is overwritten with empty content."""
    ws = InMemoryWorkspace()
    mem = InMemoryMemory()
    # Pre-seed a block from an earlier prompt.
    await mem.update_block(
        "learned_notes", "old stale advice", user_id=_USER_ID
    )
    fake = _fake_repl(ws, mem)

    await Repl._inject_learned_notes(fake, "completely unrelated task")

    assert await _block_text(mem, "learned_notes") == ""


async def test_injection_failure_never_raises(tmp_path: Path) -> None:
    """Memory/workspace I/O failing must not kill the turn."""

    class _Boom:
        async def search_notes(self, *a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("workspace down")

    fake = SimpleNamespace(
        workspace=_Boom(), agent=SimpleNamespace(memory=None)
    )
    await Repl._inject_learned_notes(fake, "anything")  # no raise
