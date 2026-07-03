"""Token-economy guards: greeting fast path + working-block dirty check.

A bare "hi" was observed costing 6,624 input tokens (solo) / 19,088
(team) because short prompts route to the full coordinator context.
And every turn rewrote identical working blocks, invalidating the
provider prompt-cache prefix. These tests lock the fixes down.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from loom_code.repl import Repl, _greeting_reply

# ---- greeting fast path ---------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    ["hi", "Hello", "HEY", "hi!", "hello there", "good morning", "yo."],
)
def test_pure_greetings_answered_locally(prompt: str) -> None:
    assert _greeting_reply(prompt) is not None


@pytest.mark.parametrize(
    "prompt",
    [
        "ok",            # moved-on feedback signal — must reach the model
        "thanks",        # same
        "hi, fix the failing test",  # greeting + task = a task
        "hello world program in C",  # content, not a greeting
        "fix it",        # anaphora — team path decides
        "",
    ],
)
def test_non_greetings_go_to_the_model(prompt: str) -> None:
    assert _greeting_reply(prompt) is None


# ---- working-block dirty check --------------------------------------------


class _RecordingMemory:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []

    async def update_block(
        self, name: str, content: str, *, user_id: str | None = None
    ) -> None:
        self.writes.append((name, content))


def _stub_with_memory() -> Any:
    stub = SimpleNamespace(
        _block_hashes={},
        agent=SimpleNamespace(memory=_RecordingMemory()),
    )
    stub._update_block_if_changed = (
        Repl._update_block_if_changed.__get__(stub)
    )
    return stub


@pytest.mark.anyio
async def test_unchanged_block_writes_once() -> None:
    stub = _stub_with_memory()
    await stub._update_block_if_changed("project_rules", "RULES v1")
    await stub._update_block_if_changed("project_rules", "RULES v1")
    await stub._update_block_if_changed("project_rules", "RULES v1")
    assert len(stub.agent.memory.writes) == 1  # cache prefix survives


@pytest.mark.anyio
async def test_changed_block_writes_again() -> None:
    stub = _stub_with_memory()
    await stub._update_block_if_changed("project_rules", "RULES v1")
    await stub._update_block_if_changed("project_rules", "RULES v2")
    assert len(stub.agent.memory.writes) == 2


@pytest.mark.anyio
async def test_blocks_are_tracked_independently() -> None:
    stub = _stub_with_memory()
    await stub._update_block_if_changed("loom_index", "MAP")
    await stub._update_block_if_changed("project_rules", "RULES")
    await stub._update_block_if_changed("loom_index", "MAP")
    assert len(stub.agent.memory.writes) == 2


@pytest.mark.anyio
async def test_failed_write_is_retried_next_turn() -> None:
    # A write that raises must NOT record the hash — else the block
    # would silently stay stale forever.
    stub = _stub_with_memory()

    class _FlakyMemory(_RecordingMemory):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next = True

        async def update_block(self, name, content, *, user_id=None):
            if self.fail_next:
                self.fail_next = False
                raise OSError("disk hiccup")
            await super().update_block(name, content, user_id=user_id)

    stub.agent = SimpleNamespace(memory=_FlakyMemory())
    with pytest.raises(OSError):
        await stub._update_block_if_changed("loom_index", "MAP")
    # Retry succeeds and actually writes.
    await stub._update_block_if_changed("loom_index", "MAP")
    assert stub.agent.memory.writes == [("loom_index", "MAP")]
