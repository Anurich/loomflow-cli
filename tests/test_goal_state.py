"""Durable /goal state (save / load / clear round-trip).

Contract: an active goal persists to .loom/goal.json and survives a
restart; load returns None for missing, corrupt, non-active, or
empty-task records; clear retires the file idempotently."""

from __future__ import annotations

import json
from pathlib import Path

from loom_code.repl import (
    _clear_goal_state,
    _load_goal_state,
    _save_goal_state,
)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    _save_goal_state(
        tmp_path,
        task="make all tests pass",
        condition="pytest exits 0",
        session_id="01ABCDEF",
        model="gpt-4.1-mini",
    )
    state = _load_goal_state(tmp_path)
    assert state is not None
    assert state["task"] == "make all tests pass"
    assert state["condition"] == "pytest exits 0"
    assert state["session_id"] == "01ABCDEF"
    assert state["status"] == "active"
    assert state["started_at"]  # timestamp recorded


def test_load_missing_is_none(tmp_path: Path) -> None:
    assert _load_goal_state(tmp_path) is None


def test_load_corrupt_is_none(tmp_path: Path) -> None:
    (tmp_path / "goal.json").write_text("{nope", encoding="utf-8")
    assert _load_goal_state(tmp_path) is None


def test_load_non_active_is_none(tmp_path: Path) -> None:
    (tmp_path / "goal.json").write_text(
        json.dumps({"task": "x", "status": "done"}), encoding="utf-8"
    )
    assert _load_goal_state(tmp_path) is None


def test_load_empty_task_is_none(tmp_path: Path) -> None:
    (tmp_path / "goal.json").write_text(
        json.dumps({"task": "  ", "status": "active"}),
        encoding="utf-8",
    )
    assert _load_goal_state(tmp_path) is None


def test_clear_retires_and_is_idempotent(tmp_path: Path) -> None:
    _save_goal_state(
        tmp_path,
        task="t",
        condition="c",
        session_id="s",
        model="m",
    )
    _clear_goal_state(tmp_path)
    assert _load_goal_state(tmp_path) is None
    _clear_goal_state(tmp_path)  # second clear: no error


def test_save_creates_loom_dir(tmp_path: Path) -> None:
    loom = tmp_path / ".loom"
    _save_goal_state(
        loom, task="t", condition="c", session_id="s", model="m"
    )
    assert _load_goal_state(loom) is not None
