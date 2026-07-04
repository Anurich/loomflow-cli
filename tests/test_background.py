"""Background bash tools (loom_code.background).

Contract: spawn returns a handle immediately (never blocks on the
child), output/status read the spool by path, kill terminates the
whole process group, kill_all leaves no orphans, and bash_background
is destructive=True (same approval contract as bash)."""

from __future__ import annotations

import time
from pathlib import Path

import anyio
import pytest

from loom_code import background as bg


@pytest.fixture(autouse=True)
def _clean():
    bg.reset()
    yield
    bg.reset()


def _tools(root: Path) -> dict:
    return {t.name: t for t in bg.background_tools(root)}


def test_spawn_returns_immediately_with_handle(tmp_path: Path) -> None:
    tools = _tools(tmp_path)

    async def go() -> None:
        t0 = time.monotonic()
        out = await tools["bash_background"].fn(command="sleep 30")
        assert time.monotonic() - t0 < 2  # did NOT wait for the child
        assert "started bg" in out and "bash_output" in out

    anyio.run(go)


def test_output_shows_running_then_exit(tmp_path: Path) -> None:
    tools = _tools(tmp_path)

    async def go() -> None:
        started = await tools["bash_background"].fn(
            command="echo hello-from-bg"
        )
        handle = started.split()[1]  # "started bg1 (pid ...)"
        # Give the child a beat to run + flush.
        for _ in range(40):
            out = await tools["bash_output"].fn(handle=handle)
            if "EXITED rc=0" in out and "hello-from-bg" in out:
                break
            await anyio.sleep(0.05)
        assert "EXITED rc=0" in out
        assert "hello-from-bg" in out

    anyio.run(go)


def test_kill_terminates_long_runner(tmp_path: Path) -> None:
    tools = _tools(tmp_path)

    async def go() -> None:
        started = await tools["bash_background"].fn(command="sleep 60")
        handle = started.split()[1]
        out = await tools["bash_kill"].fn(handle=handle)
        assert "terminated" in out
        status = await tools["bash_output"].fn(handle=handle)
        assert "EXITED" in status

    anyio.run(go)


def test_unknown_handle_errors_helpfully(tmp_path: Path) -> None:
    tools = _tools(tmp_path)

    async def go() -> None:
        out = await tools["bash_output"].fn(handle="bg99")
        assert out.startswith("ERROR")
        out = await tools["bash_kill"].fn(handle="bg99")
        assert out.startswith("ERROR")

    anyio.run(go)


def test_kill_all_reaps_everything(tmp_path: Path) -> None:
    tools = _tools(tmp_path)

    async def go() -> None:
        await tools["bash_background"].fn(command="sleep 60")
        await tools["bash_background"].fn(command="sleep 60")
        assert bg.kill_all() == 2
        assert bg.kill_all() == 0  # idempotent

    anyio.run(go)


def test_bash_background_is_destructive(tmp_path: Path) -> None:
    # Runs arbitrary code → must carry the same approval contract as
    # bash. If this flips, the gate silently stops firing.
    tools = _tools(tmp_path)
    assert tools["bash_background"].destructive is True
    assert not getattr(tools["bash_output"], "destructive", False)


def test_empty_command_rejected(tmp_path: Path) -> None:
    tools = _tools(tmp_path)

    async def go() -> None:
        out = await tools["bash_background"].fn(command="   ")
        assert out.startswith("ERROR")

    anyio.run(go)
