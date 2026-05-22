"""Tests for the shell-command hook shim + the project-hook trust gate.

The hook shim (``loom_code.hooks``) turns ``settings.toml`` ``[[hooks]]``
entries into loomflow ``PreToolUse`` / ``PostToolUse`` callbacks and
REPL-lifecycle runners. The trust gate (``loom_code.trust``) gates
project-scope hooks behind explicit consent.

Hooks are real shell commands, so these tests run tiny inline shell
snippets (``echo`` / ``exit 2``) — offline, no network, POSIX shell.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from loomflow import Agent
from loomflow.core.types import ToolCall, ToolResult

from loom_code.extensions import Extensions, HookSpec
from loom_code.hooks import (
    _make_post_tool_hook,
    _make_pre_tool_hook,
    attach_tool_hooks,
    matches,
    run_repl_hooks,
)
from loom_code.trust import (
    filter_trusted_hooks,
    is_trusted,
    record_trust,
)

pytestmark = pytest.mark.anyio


# ---- matcher --------------------------------------------------------


def test_matches_wildcard_and_empty() -> None:
    assert matches("*", "bash")
    assert matches("", "anything")


def test_matches_pipe_list() -> None:
    assert matches("bash|edit", "edit")
    assert matches("bash|edit", "bash")
    assert not matches("bash|edit", "read")


def test_matches_regex() -> None:
    assert matches("web_.*", "web_fetch")
    assert not matches("web_.*", "read")


def test_matches_invalid_regex_fails_closed() -> None:
    assert not matches("(unclosed", "anything")


# ---- pre-tool hooks -------------------------------------------------


async def test_pre_tool_hook_blocks_on_exit_2(tmp_path: Path) -> None:
    spec = HookSpec(
        event="PreToolUse",
        command="echo 'rm forbidden' 1>&2; exit 2",
        matcher="bash",
        source="project",
    )
    hook = _make_pre_tool_hook(spec, cwd=tmp_path)
    call = ToolCall(id="c1", tool="bash", args={"command": "rm -rf /"})
    decision = await hook(call)
    assert decision is not None
    assert decision.decision == "deny"
    assert "rm forbidden" in (decision.reason or "")


async def test_pre_tool_hook_passes_on_matcher_miss(
    tmp_path: Path,
) -> None:
    spec = HookSpec(
        event="PreToolUse", command="exit 2", matcher="bash"
    )
    hook = _make_pre_tool_hook(spec, cwd=tmp_path)
    call = ToolCall(id="c2", tool="read", args={"path": "x"})
    assert await hook(call) is None


async def test_pre_tool_hook_rewrites_input(tmp_path: Path) -> None:
    spec = HookSpec(
        event="PreToolUse",
        command='echo \'{"updatedInput": {"command": "ls"}}\'',
        matcher="*",
    )
    hook = _make_pre_tool_hook(spec, cwd=tmp_path)
    call = ToolCall(id="c3", tool="bash", args={"command": "rm -rf /"})
    decision = await hook(call)
    assert decision is None  # rewrite, not block
    assert call.args == {"command": "ls"}


async def test_pre_tool_hook_block_via_json(tmp_path: Path) -> None:
    spec = HookSpec(
        event="PreToolUse",
        command='echo \'{"decision": "block", "reason": "nope"}\'',
        matcher="*",
    )
    hook = _make_pre_tool_hook(spec, cwd=tmp_path)
    call = ToolCall(id="c4", tool="bash", args={})
    decision = await hook(call)
    assert decision is not None and decision.decision == "deny"
    assert decision.reason == "nope"


# ---- post-tool hooks ------------------------------------------------


async def test_post_tool_hook_runs_command(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    spec = HookSpec(
        event="PostToolUse",
        command=f"touch {marker}",
        matcher="edit",
    )
    hook = _make_post_tool_hook(spec, cwd=tmp_path)
    call = ToolCall(id="c5", tool="edit", args={"path": "f"})
    result = ToolResult(call_id="c5", ok=True, output="done")
    await hook(call, result)
    assert marker.exists()


async def test_post_tool_hook_skips_on_matcher_miss(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "ran"
    spec = HookSpec(
        event="PostToolUse", command=f"touch {marker}", matcher="bash"
    )
    hook = _make_post_tool_hook(spec, cwd=tmp_path)
    call = ToolCall(id="c6", tool="read", args={})
    result = ToolResult(call_id="c6", ok=True, output="x")
    await hook(call, result)
    assert not marker.exists()


# ---- attach_tool_hooks ----------------------------------------------


def test_attach_registers_and_bumps_timeout(tmp_path: Path) -> None:
    agent = Agent("hi", model="echo")
    specs = [
        HookSpec(event="PreToolUse", command="true", timeout=30.0),
        HookSpec(event="PostToolUse", command="true", timeout=10.0),
    ]
    attach_tool_hooks(agent, specs, cwd=tmp_path)
    reg = agent._hooks  # noqa: SLF001
    assert len(reg.pre_tool_hooks) == 1
    assert len(reg.post_tool_hooks) == 1
    # timeout bumped to cover the slowest spec (30 + margin)
    assert reg.hook_timeout_s >= 31.0


def test_attach_noop_without_tool_hooks(tmp_path: Path) -> None:
    agent = Agent("hi", model="echo")
    before = agent._hooks.hook_timeout_s  # noqa: SLF001
    # only a REPL-lifecycle hook -> attach must not touch the registry
    attach_tool_hooks(
        agent,
        [HookSpec(event="UserPromptSubmit", command="x")],
        cwd=tmp_path,
    )
    reg = agent._hooks  # noqa: SLF001
    assert len(reg.pre_tool_hooks) == 0
    assert len(reg.post_tool_hooks) == 0
    assert reg.hook_timeout_s == before


# ---- REPL-lifecycle hooks -------------------------------------------


async def test_repl_hook_injects_context(tmp_path: Path) -> None:
    specs = [
        HookSpec(
            event="UserPromptSubmit",
            command='echo \'{"additionalContext": "be terse"}\'',
        )
    ]
    result = await run_repl_hooks(
        specs, "UserPromptSubmit", cwd=tmp_path, prompt="hello"
    )
    assert not result.blocked
    assert result.added_context == "be terse"


async def test_repl_hook_blocks_turn(tmp_path: Path) -> None:
    specs = [
        HookSpec(
            event="UserPromptSubmit",
            command="echo 'policy violation' 1>&2; exit 2",
        )
    ]
    result = await run_repl_hooks(
        specs, "UserPromptSubmit", cwd=tmp_path, prompt="do bad thing"
    )
    assert result.blocked
    assert "policy violation" in (result.reason or "")


async def test_repl_hook_only_fires_matching_event(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "ran"
    specs = [HookSpec(event="SessionEnd", command=f"touch {marker}")]
    # firing a DIFFERENT event must not run the SessionEnd hook
    await run_repl_hooks(specs, "SessionStart", cwd=tmp_path)
    assert not marker.exists()
    await run_repl_hooks(specs, "SessionEnd", cwd=tmp_path)
    assert marker.exists()


# ---- trust gate -----------------------------------------------------


def _project_hook() -> Extensions:
    return Extensions(
        hook_specs=[
            HookSpec(
                event="PreToolUse",
                command="./check.sh",
                matcher="bash",
                source="project",
            )
        ]
    )


def test_trust_deny_drops_project_hooks(tmp_path: Path) -> None:
    ext = _project_hook()
    out = filter_trusted_hooks(
        ext,
        project_root=tmp_path,
        prompt=lambda specs: False,
        trust_store=tmp_path / "trust.json",
    )
    assert out.hook_specs == []


def test_trust_approve_keeps_and_remembers(tmp_path: Path) -> None:
    ext = _project_hook()
    store = tmp_path / "trust.json"
    approved = filter_trusted_hooks(
        ext, project_root=tmp_path, prompt=lambda s: True, trust_store=store
    )
    assert len(approved.hook_specs) == 1
    assert store.exists()

    # second run: trusted -> prompt must NOT be called
    def boom(specs: object) -> bool:
        raise AssertionError("should not prompt once trusted")

    again = filter_trusted_hooks(
        ext, project_root=tmp_path, prompt=boom, trust_store=store
    )
    assert len(again.hook_specs) == 1


def test_trust_reprompts_when_command_changes(tmp_path: Path) -> None:
    store = tmp_path / "trust.json"
    filter_trusted_hooks(
        _project_hook(),
        project_root=tmp_path,
        prompt=lambda s: True,
        trust_store=store,
    )
    changed = Extensions(
        hook_specs=[
            HookSpec(
                event="PreToolUse",
                command="./DIFFERENT.sh",
                matcher="bash",
                source="project",
            )
        ]
    )
    asked = {"v": False}

    def prompt(specs: object) -> bool:
        asked["v"] = True
        return False

    filter_trusted_hooks(
        changed, project_root=tmp_path, prompt=prompt, trust_store=store
    )
    assert asked["v"] is True


def test_trust_user_hooks_always_pass(tmp_path: Path) -> None:
    # user-scope hooks are never gated; prompt must not be consulted
    ext = Extensions(
        hook_specs=[
            HookSpec(
                event="UserPromptSubmit", command="x", source="user"
            )
        ]
    )

    def boom(specs: object) -> bool:
        raise AssertionError("user hooks must not be gated")

    out = filter_trusted_hooks(
        ext,
        project_root=tmp_path,
        prompt=boom,
        trust_store=tmp_path / "trust.json",
    )
    assert len(out.hook_specs) == 1


def test_is_trusted_and_record_trust_roundtrip(tmp_path: Path) -> None:
    # The async-friendly helpers the desktop uses: check, record,
    # re-check. Mirrors filter_trusted_hooks' decision without the
    # sync prompt callback.
    store = tmp_path / "trust.json"
    hooks = [
        HookSpec(
            event="PreToolUse",
            command="./check.sh",
            matcher="bash",
            source="project",
        )
    ]
    assert is_trusted(tmp_path, hooks, trust_store=store) is False
    record_trust(tmp_path, hooks, trust_store=store)
    assert is_trusted(tmp_path, hooks, trust_store=store) is True
    # changing a command invalidates trust (re-prompt territory)
    changed = [
        HookSpec(
            event="PreToolUse",
            command="./OTHER.sh",
            matcher="bash",
            source="project",
        )
    ]
    assert is_trusted(tmp_path, changed, trust_store=store) is False


def test_is_trusted_empty_hooks_is_true(tmp_path: Path) -> None:
    # No project hooks -> nothing to gate -> trusted (and record is a
    # no-op).
    assert is_trusted(tmp_path, [], trust_store=tmp_path / "t.json")
    record_trust(tmp_path, [], trust_store=tmp_path / "t.json")
    assert not (tmp_path / "t.json").exists()
