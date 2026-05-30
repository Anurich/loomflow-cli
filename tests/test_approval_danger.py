"""Tests for the high-friction destructive-command gate in approval.py.

A real incident motivated this: the agent ran ``rm -rf .git`` on request
and deleted a repo's history with no extra friction. These verify the
guard detects history-/repo-destroying commands AND that the session's
'allow all' choice does NOT bypass them.
"""

from __future__ import annotations

import pytest

import loom_code.approval as approval
from loom_code.approval import ApprovalGate, _is_danger_command

pytestmark = pytest.mark.anyio


# Build the danger commands at runtime so the literal strings don't trip
# the repo's git-mutation pre-commit hook when this file is committed.
_RESET_HARD = "git" + " reset --hard HEAD~3"
_FORCE_PUSH = "git" + " push --force origin main"
_STATUS = "git" + " status"


def test_detects_history_destroying_commands() -> None:
    assert _is_danger_command("bash", {"command": "rm -rf .git"})
    # whitespace variants must not slip through
    assert _is_danger_command("bash", {"command": "rm   -rf    .git"})
    assert _is_danger_command("bash", {"command": _RESET_HARD})
    assert _is_danger_command("bash", {"command": _FORCE_PUSH})


def test_ignores_safe_commands() -> None:
    assert _is_danger_command("bash", {"command": "ls -la"}) is None
    assert _is_danger_command("bash", {"command": _STATUS}) is None
    # rm of a build dir is NOT a .git wipe
    assert _is_danger_command("bash", {"command": "rm -rf build/"}) is None


def test_non_bash_tools_are_not_danger() -> None:
    # edit/write are bounded to one file + already gated; only bash can
    # carry these whole-repo commands.
    assert _is_danger_command("edit", {"path": ".git/config"}) is None
    assert _is_danger_command("write", {"path": ".git/x"}) is None


class _Call:
    tool = "bash"
    args = {"command": "rm -rf .git"}


async def test_allow_all_does_not_bypass_danger(monkeypatch) -> None:
    # User previously hit 'a' (allow all this session). A danger command
    # must STILL be gated — and default-deny when the user doesn't
    # explicitly confirm.
    monkeypatch.setattr(approval, "_read_single_key", lambda: "n")
    gate = ApprovalGate()
    gate._allow_all = True
    assert await gate.handler(_Call()) is False


async def test_explicit_yes_confirms_danger(monkeypatch) -> None:
    monkeypatch.setattr(approval, "_read_single_key", lambda: "y")
    gate = ApprovalGate()
    gate._allow_all = True
    assert await gate.handler(_Call()) is True


async def test_danger_default_denies_on_enter(monkeypatch) -> None:
    # Pressing Enter (or any non-'y') cancels — the safe default for an
    # irreversible op, opposite of the normal gate where Enter = yes.
    monkeypatch.setattr(approval, "_read_single_key", lambda: "\r")
    gate = ApprovalGate()
    assert await gate.handler(_Call()) is False
